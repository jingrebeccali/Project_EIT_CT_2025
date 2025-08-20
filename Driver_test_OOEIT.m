%% 1) Réglages de base
clear; close all; clc;
addpath(genpath('C:\Program Files\distmesh-master'))       % To adapt
addpath(genpath('C:\Program Files\OOEIT-main'))      
addpath(genpath('C:\Users\boulette\Desktop\Depot2\Project_EIT_CT_2025\mat_aatae'))



baseFolder  = 'meshes_mat_sujetss';
refSubject  = 's0011';
refSliceNum = 115;
Ncontour    = 400;  % nombre de points pour le sampling des points sur le contour
n_el = 16;
L=3.5;                     %% en Cm
NslicePerZone    = 4;    % nombre de slices par zone





%% charge reference 
% —> mesh + signal

fn      = sprintf('%s_mesh_slice_%d.mat', refSubject, refSliceNum);
S    = load(fullfile(baseFolder,refSubject,fn), 'g','H','sigma','edges','z','present_organs');

g_ref    = S.g;      H_ref = S.H+1;        % +1 because Python indexation is 0-based 
sig_ref  = S.sigma;  edges_ref=S.edges +1;  % same
z_ref = double(S.z);

if size(g_ref,2)>2
  g_ref = g_ref(:,1:2);      %get x,y only  from mesh.nodes of pyeit
end
 

E_ref = generateElectrodesSurfaces(g_ref, edges_ref, n_el, L);  %% a celle representing electrodes,format adapted to ForwardMesh1st

% build forward solver
fmesh_ref = ForwardMesh1st(g_ref,H_ref,E_ref);
solver_ref = EITFEM(fmesh_ref);

solver_ref.mode = 'current';

M = eye(n_el) - circshift(eye(n_el), [0, n_el/2]); 

solver_ref.Iel  = M(:);     %% opposed injecting electrodes, if  you want adjacent mode, comment this line

sig_real = Sig_update(S.present_organs,sig_ref);    %% Update the values of conductivity.If you want to modify the values, go to Sig_update.m

Umeas_ref = solver_ref.SolveForwardVec(sig_real);   %%Output of the forward solver:Measures on electrodes


[~, bnd_pts_A] = extract_ordered_boundary(g_ref, edges_ref);
dst = resample_contour_by_arclength(bnd_pts_A, Ncontour);        %% Exctratinhg boundary points of the mesh. Usuful for TPS.

figure;
subplot(2,1,1); plot(Umeas_ref); 
subplot(2,1,2);
plotMeshAndElectrodes(g_ref,H_ref,sig_real,E_ref)


fprintf('La slice de référence est prête\n')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%      preparing data -- Nothing to modify in this code -- 

slice_number = z_ref;

allFiles = struct('subject',{},'file',{},'sliceNum',[]);
cnt = 0;
subs = dir(baseFolder);
subs = subs([subs.isdir] & ~ismember({subs.name},{'.','..'}));
disp(numel(subs))
for i=1:numel(subs)
  subj = subs(i).name;
  mats = dir(fullfile(baseFolder,subj,'*_mesh_slice_*.mat'));
  for k=1:numel(mats)
    info = load(fullfile(baseFolder,subj,mats(k).name),'z');

    if ~isfield(info,'z')
      continue;  % ignore si pas là
    end
    cnt = cnt+1;
    allFiles(cnt).subject  = subj;
    allFiles(cnt).file     = mats(k).name;
    allFiles(cnt).sliceNum = double(info.z);
    % fprintf('un traité\n');

  end
  fprintf('le sujet %s traité\n',subj);
end


%% Pour chaque sujet, assigne une zone basée sur sliceNum -- Nothing to change in this code -- 
subjects = unique({allFiles.subject});
for i=1:numel(subjects)
  subj  = subjects{i};
  idx   = strcmp({allFiles.subject}, subj);
  sNums = [allFiles(idx).sliceNum];
  mn    = min(sNums);
  % zone = floor((sliceNum - mn)/NslicePerZone) + 1
  zones = floor((sNums - mn)/NslicePerZone) + 1;
  ids   = find(idx);
  for j=1:numel(ids)
    allFiles(ids(j)).zone = zones(j);
  end
end

%%    Save the name of slices to consider i the search. -- same -- 

isRefEntry = strcmp({allFiles.subject},refSubject) ...
           & ([allFiles.sliceNum] == z_ref);
assert(sum(isRefEntry)==1, 'Réf introuvable ou dupliquée');
refZone = allFiles(isRefEntry).zone;

mask     = ([allFiles.zone] == refZone);
filtered = allFiles(mask);
filtered = filtered(~strcmp({filtered.subject}, refSubject));

fprintf('Zone #%d retenue → %d slices\n', refZone, sum(mask));


%%  Construit la struct results attendue
subjectsF = unique({filtered.subject});
results   = struct('subject',{},'bestIdxSlices',{},'bestFileNames',{});


for i=1:numel(subjectsF)
  subj = subjectsF{i};
  sel  = strcmp({filtered.subject}, subj);
  fs   = filtered(sel);
  % trie par sliceNum
  [~,ord] = sort([fs.sliceNum]);
  fs      = fs(ord);
  results(i).subject       = subj;
  results(i).bestIdxSlices = [fs.sliceNum];
  results(i).bestFileNames = {fs.file};
end

%%    %%%%%%%%%%  Parallel Toolbox nécessaire pour ce calcule, sinon en séquentiel
tic
nW = 16;  % par ex. 16 workers
if exist('parpool','file')
  p = gcp('nocreate');
  if isempty(p) || p.NumWorkers~=nW
    if ~isempty(p), delete(p); end
    parpool('local', nW);
  end
else
  warning('Pas de Parallel Toolbox : tout sera séquentiel');
end

dq = parallel.pool.DataQueue;


warpedResults = warpAndComputeMeasurements( ...
               baseFolder, results, refSubject, refSliceNum, Ncontour,n_el,L,dq);
delete(gcp('nocreate'));

elapsed = toc;         

fprintf('%d slices par sujet sont transformées au contour de la slice de référence et sont résolues',NslicePerZone)
fprintf('Elapsed time: %.4f seconds\n', elapsed);

%%    Selecting the top Nsel slices

Nsel=15;  %% To adapt

M    = numel(warpedResults);
mseV = nan(M,1);

%  Calcul de la MSE pour chaque slice
for k = 1:M
    Iw = warpedResults(k).Imeas_wrapped;
    % Assurez‑vous qu'Iw et Imeas_ref ont la même taille
    if numel(Iw) ~= numel(Umeas_ref)
        error('Tailles incompatibles entre Imeas_wrapped (%d) et Imeas_ref (%d).', ...
               numel(Iw), numel(Imeas_ref));
    end
    mseV(k) = mean( (Iw - Umeas_ref).^2 );
end


%  On trie et on prend les Nsel plus petites MSE
[sortedMSE, sortIdx] = sort(mseV);
count                = min(Nsel, M);
bestIdx              = sortIdx(1:count);

fprintf('\n=== Top %d slices par MSE croissante ===\n', count);


%%

figure;
subplot(2,1,1);plot(Umeas_ref);
subplot(2,1,2);
h = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_real);
title(sprintf('Reference slice subject %s slice %d',refSubject,refSliceNum))
set(h, "facecolor", "interp");
set(h, "facealpha", 0.6);
set(h, "EdgeColor", "none");
axis equal;
grid on;
% caxis([0,2])

colorbar; view(2);


for i = 1:count
    idx    = bestIdx(i);
    subj   = warpedResults(idx).subject;
    sliceN = warpedResults(idx).slice;
    fprintf('%2d) Sujet %s, slice %3d → MSE = %.6e\n', ...
            i, subj, sliceN, sortedMSE(i));
    subjDir = fullfile(baseFolder, subj);
    fn      = sprintf('%s_mesh_slice_%d.mat', subj, sliceN);
    S       = load(fullfile(subjDir, fn));
    g = S.g;   H = S.H+1;   sig = S.sigma; edgesB=S.edges+1;

    if size(g,2) > 2
        g = g(:,1:2);
    end

    %  Contour cible et rééchantillonnage
    [~, bnd_pts_B] = extract_ordered_boundary(g, edgesB);
    src = resample_contour_by_arclength(bnd_pts_B, Ncontour);

    % 2.3) Alignement des contours
    [src_aligned, ~, ~] = align_contours(src, dst);

    % 2.4) Estimation de la TPS
    tps = ThinPlateSpline2D();
    tps = tps.fit(src_aligned, dst);

    % 2.5) Application du warp aux nœuds de référence
    nodesB_warp = tps.transform(g);


    figure('Name',sprintf('Top %d : %s',i,fn),'Position',[200 200 600 800]);
    subplot(2,1,1);
    plot(warpedResults(idx).Imeas_wrapped,'-o','LineWidth',1.5);
    title(sprintf('%s — Err = %.3e',fn,sortedMSE(i)));
    xlabel('Index d''électrode'); ylabel('Potentiel'); grid on;
    subplot(2,1,2);
    h = trimesh(H(:,1:3), nodesB_warp(:,1), nodesB_warp(:,2), warpedResults(idx).sigCell);
    title(sprintf('s%s slice %d— Err = %.3e',subj,sliceN,sortedMSE(i)));
    set(h, 'FaceColor','interp', 'FaceAlpha',0.6, 'EdgeColor','none');
    axis equal tight;  grid on;
    colorbar; view(2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%   calcul du vecteur de sigma moyen  de toutes les slices de la zone de refSlice

[~, bndA] = extract_ordered_boundary(g_ref, edges_ref);
dst       = resample_contour_by_arclength(bndA, 400);

sumSigmaB   = zeros(size(g_ref,1),1);
sigmaB_cell = {};
count = 0;

for i = 1:numel(results)
    subj = results(i).subject;
    fns  = results(i).bestFileNames;  % cellstr
    for k = 1:numel(fns)
        fn = fullfile(baseFolder, subj, fns{k});
        S  = load(fn, 'g','H','sigma','edges','z','present_organs','mask');

        if ~isfield(S,'g') || ~isfield(S,'sigma')
            warning('Missing g/sigma in %s — skipping.', fn);
            continue;
        end

        gB = double(S.g);
        if size(gB,2) > 2, gB = gB(:,1:2); end
        sigmaB = double(S.sigma);
        [~, bndB] = extract_ordered_boundary(gB, S.edges);
        src       = resample_contour_by_arclength(bndB, 400);
        [src_a,~,~] = align_contours(src, dst);
        tps       = ThinPlateSpline2D().fit(src_a, dst);
        nodesW    = tps.transform(gB(:,1:2));

        sigmaB = Sig_update(S.present_organs,sigmaB);

        % FE-consistent scatter interp: linear inside hull, nearest outside
        F = scatteredInterpolant(nodesW(:,1), nodesW(:,2), sigmaB, 'linear', 'nearest');

        sigmaB = F(g_ref(:,1), g_ref(:,2));  % resampled onto reference nodes

        % accumulate
        sumSigmaB = sumSigmaB + sigmaB;
        sigmaB_cell{end+1,1} = sigmaB; %#ok<*AGROW>
        count = count + 1;
    end
    fprintf('le sujet %s traité\n',subj);

end

if count == 0
    error('No slices were processed. Check `results` and file paths.');
end

avgSigmaB = sumSigmaB / count;

%%   Plot

% compute common color limits
allData = [sig_real; avgSigmaB];
cmin = min(allData);
cmax = max(allData);

figure;

% top subplot
ax1 = subplot(2,1,1);
h1 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_real);
set(h1, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig_real')
caxis([cmin cmax]);   % use common limits

% bottom subplot
ax2 = subplot(2,1,2);
h2 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), avgSigmaB);
set(h2, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig paper')
caxis([cmin cmax]);   % same limits
% 4) one colorbar on the right, spanning both
colormap(jet);
cb = colorbar('Position',[0.92  0.11  0.02  0.815]);  % [x y w h] in normalized figure units
cb.Label.String = 'Conductivity';


%%   solve forwarad problem 

solver = EITFEM(fmesh_ref);
solver.mode = 'current';
M = eye(n_el) - circshift(eye(n_el), [0, n_el/2]);  
solver.Iel  = M(:);       %% opposed injecting electrodes, if  you want adjacent mode, comment this line(AND solver.Iel = M(:); below)


sig_real = Sig_update(S.present_organs,sig_ref);

Umeas_ref = solver.SolveForwardVec(sig_real);


%%  Inverse sovler with avgSigmaB as starting point

solver.Iel = M(:);
solver.Uel = Umeas_ref;

hbar = mean_edge_length(g_ref,H_ref);

TV1   = PriorTotalVariation(g_ref, H_ref, 0.30, 1, 1e-3);

Pos   = PriorPositivityParabolic(1e-6, 0.2);   

mu1   = avgSigmaB;                
var1  = (1.0)^2;                  
cor1  = 2*hbar;
SM1   = PriorSmoothness(g_ref, cor1, var1, mu1);

sigma0 = avgSigmaB;

inv1 = SolverGN({ solver, TV1, SM1, Pos });
inv1.maxIter = 5; inv1.maxIterInLine = 10;
inv1.eStep = 1e-6; inv1.eStop = 5e-5;
inv1.plotIterations = true; inv1.plotter = Plotter(g_ref,H_ref);

sigma_final = inv1.Solve(sigma0);






%% Plot 
allData = [sig_real; sigma1];
cmin = min(allData);
cmax = max(allData);

figure;

ax1 = subplot(2,1,1);
h1 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_real);
set(h1, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig_real')
caxis([cmin cmax]);   % use common limits

ax2 = subplot(2,1,2);
h2 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sigma1);
set(h2, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig paper')
caxis([cmin cmax]);   % same limits
% 4) one colorbar on the right, spanning both
colormap(jet);
cb = colorbar('Position',[0.92  0.11  0.02  0.815]);  % [x y w h] in normalized figure units
cb.Label.String = 'Conductivity';




%%     Inverse problem with NOSER

TVPrior = PriorTotalVariation(g_ref, H_ref,100000);
invSolver = SolverGN({ solver; TVPrior });
invSolver.maxIter =1;            % a handful of GN steps
invSolver.eStep   = 1e-6;
invSolver.eStop   = 1e-6;
invSolver.plotIterations = true;
solver.Iel = M(:);
solver.Uel = Umeas_ref;

plotter = Plotter(g_ref, H_ref);
invSolver.plotter = plotter;


invSolver.maxIterInLine = 70;
N       = numel(sig_real);
onesSig = ones(N,1);
U1      = solver.SolveForwardVec(onesSig); 



rho_b   = (U1'*Umeas_ref)/(U1'*U1);

%U1 = U1(mask);

sigma0  = (1/rho_b)*onesSig;

sigma_final_paper = invSolver.Solve(sigma0);
%% Plot 

% compute common color limits
allData = [sig_real; sigma_final_paper];
cmin = min(allData);
cmax = max(allData);

figure;

% top subplot
ax1 = subplot(2,1,1);
h1 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_real);
set(h1, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig_real')
caxis([cmin cmax]);   % use common limits

% 3) bottom subplot
ax2 = subplot(2,1,2);
h2 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sigma_final_paper);
set(h2, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig paper')
caxis([cmin cmax]);   % same limits
% 4) one colorbar on the right, spanning both
colormap(jet);
cb = colorbar('Position',[0.92  0.11  0.02  0.815]);  % [x y w h] in normalized figure units
cb.Label.String = 'Conductivity';

%%             Prepare a random slice to be used as starting point
fn = sprintf('%s_mesh_slice_%d.mat', 's0011', 113);   %% To adapt
S_1  = load(fullfile(baseFolder,'s0011',fn), 'g','H','sigma','edges','z','present_organs','mask');  %%To adapt

mask_ref_1 = S_1.mask;

g_ref_1    = S_1.g;      H_ref_1 = S_1.H+1;        %
sig_ref_1  = S_1.sigma;  edges_ref_1=S_1.edges+1;
sig_ref_1 = Sig_update(S_1.present_organs,sig_ref_1);

% 
if size(g_ref_1,2)>2
  g_ref_1 = g_ref_1(:,1:2);      %get x,y only  from mesh.nodes of pyeit
end


%% Transfom to refslice domain with TPS
[~, bndB] = extract_ordered_boundary(g_ref_1, edges_ref_1);
src       = resample_contour_by_arclength(bndB, 400);
[src_a,~,~] = align_contours(src, dst);
tps       = ThinPlateSpline2D().fit(src_a, dst);
nodesW    = tps.transform(g_ref_1(:,1:2));
%%   Matching vectors of conductivty
F = scatteredInterpolant(nodesW(:,1), nodesW(:,2), sig_ref_1, ...
                         'linear', 'nearest');

% evaluate at the new nodes
sigma_rand = F(g_ref(:,1), g_ref(:,2));

%%  Inverse problem (I dont really know what to do here, many parameters seem to have a good result if the staring slice if close enough)




hbar = mean_edge_length(g_ref,H_ref);

TV1   = PriorTotalVariation(g_ref, H_ref, 0.30, 1, 1e-3);

Pos   = PriorPositivityParabolic(1e-6, 0.2);   

mu1   = avgSigmaB;                
var1  = (1.0)^2;                  
cor1  = 2*hbar;
SM1   = PriorSmoothness(g_ref, cor1, var1, mu1);

sigma0 = avgSigmaB;

inv = SolverGN({ solver, TV1, SM1, Pos });
inv.maxIter = 5; inv.maxIterInLine = 10;
inv.eStep = 1e-6; inv.eStop = 5e-5;
inv.plotIterations = true; inv.plotter = Plotter(g_ref,H_ref);



sigma_final_random = inv.Solve(sigma_rand);
%%

% 1) compute common color limits
allData = [sig_real;sigma_final_random];
cmin = min(allData);
cmax = max(allData);

figure;

% 2) top subplot
ax1 = subplot(2,1,1);
h1 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_real);
set(h1, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig_real')
caxis([cmin cmax]);   

ax2 = subplot(2,1,2);
h2 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sigma_final_random);
set(h2, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig paper')
caxis([cmin cmax]);  
colormap(jet);
cb = colorbar('Position',[0.92  0.11  0.02  0.815]);  % [x y w h] in normalized figure units
cb.Label.String = 'Conductivity';


%% 
% 1) compute common color limits
allData = [sig_real; sigma_final;sigma_final_paper;sigma_final_random];
cmin = min(allData);
cmax = max(allData);

figure;

% 2) top subplot
ax1 = subplot(4,1,1);
h1 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_real);
set(h1, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig_real')
caxis([cmin cmax]);   % use common limits

% 3) bottom subplot
ax2 = subplot(4,1,2);
h2 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sigma_final);
set(h2, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig averge')
caxis([cmin cmax]);   % same limits

ax3 = subplot(4,1,3);
h3 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sigma_final_paper);
set(h3, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig paper')
caxis([cmin cmax]);   % same limits

ax4 = subplot(4,1,4);
h4 = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sigma_final_random);
set(h4, 'facecolor','interp', 'facealpha',0.6, 'edgecolor','none');
axis equal off; grid on; view(2);
title('sig_random')
caxis([cmin cmax]);   % same limits

% 4) one colorbar on the right, spanning both
colormap(jet);
cb = colorbar('Position',[0.92  0.11  0.02  0.815]);  % [x y w h] in normalized figure units
cb.Label.String = 'Conductivity';













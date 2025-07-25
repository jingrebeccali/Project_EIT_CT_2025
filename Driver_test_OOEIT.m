%% 1) Réglages de base
clear; close all; clc;
addpath(genpath('C:\Program Files\distmesh-master'))       % To adapt
addpath(genpath('C:\Program Files\OOEIT-main'))      
addpath(genpath('C:\Users\boulette\Desktop\Depot2\Project_EIT_CT_2025\mat_aatae'))

% dossier « référence » (un seul sujet dont on veut afficher mesh+signal)



% Dir='meshes_mat_sujetss/s0059';
% 
% refDir    = 'meshes_mat_sujetss/s0011';  %% choose subject and slice
% % dossier « bibliothèque » de tous les meshes à comparer
% 
% % nom du fichier référence (vous pouvez aussi prendre le premier trouvé automatiquement)
% refFile   = 's0011_mesh_slice_190.mat';  
% 
%  %%%%%%%%% conductivity values, in order of :
%  %%%%%%%%% Soft_tissue,lungs,heart,aorta,trachea,esophagus,ribs,vertebraes,scapula,
% 
% b = [0.3, 0.15, 0.01, 0.2, 0.6, 0.8, 0.1, 0.25, 0.05];  
% 
%  %%%%%%%%%  (TODO : need get the list of organs present in the slice from
%  %%%%%%%%% python)
% 
% %% charge reference 
% % —> mesh + signal
% S = load(fullfile(refDir,refFile));
% g_ref    = S.g;      H_ref = S.H+1;        % +1 si vos H étaient 0-based  
% sig_ref  = S.sigma;  
% 
% 
% c = zeros(size(sig_ref)); % on prépare le vecteur résultat
% 
% for i = 1:9
%     % on trouve toutes les positions où v == i
%     idx = (sig_ref == i);
%     % et on y place b(i)
%     c(idx) = b(i);
% end
% 
% %sig_ref=c;
% 
% if size(g_ref,2)>2
%   g_ref = g_ref(:,1:2);      %get x,y only  from mesh.nodes of pyeit
% end
% 
% 
% n_el = 16;       % needs to be the same as pyeit( TODO : get it from pyeit as well)
% 
% % recreate  Electrodes surface elementst
% E_ref = cell(n_el,1);
% for L = 1:n_el
%     fld = sprintf('E%d', L);
% 
%     E_ref{L} = S.(fld) + 1;  
% end
% 
% 
% 
% % build forward solver
% fmesh_ref = ForwardMesh1st(g_ref,H_ref,E_ref);
% solver_ref = EITFEM(fmesh_ref);
% solver_ref.mode = 'current';
% 
% Imeas_ref = solver_ref.SolveForwardVec(sig_ref);
% 
% 
% 
% 
% %%%%%%%%%%%%  uncomment if you want to remove some measurments like in your
% %%%%%%%%%%%%  driver.Then have to cuncomment this part on line 107 as well
% %%%%%%%%%%%%  %%%
% % validIdx = ~isnan(Imeas_ref);
% % for n=0:fmesh_ref.nEl-1
% %   block = Imeas_ref(1+n*fmesh_ref.nEl : (n+1)*fmesh_ref.nEl);
% %   ii = find(block<0);
% %   for j=ii'
% %     idx = j + n*fmesh_ref.nEl;
% %     Imeas_ref(idx)                          = nan;
% %     Imeas_ref(mod(idx, fmesh_ref.nEl)+1)   = nan;
% %     Imeas_ref(mod(idx-2, fmesh_ref.nEl)+1) = nan;
% %   end
% % end
% 
% figure;
% subplot(2,1,1); plot(Imeas_ref); 
% subplot(2,1,2)
% h = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_ref);
% set(h, "facecolor", "interp");
% set(h, "facealpha", 0.6);
% set(h, "EdgeColor", "none");
% axis equal;
% grid on;
% % caxis([0,2])
% 
% colorbar; view(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



baseFolder  = 'meshes_mat_sujetss';
refSubject  = 's0011';
refSliceNum = 96;
Ncontour    = 400;  % nombre de points pour le sampling des points sur le contour
n_el = 8;
L=27;                     %% en mm
NslicePerZone    = 15;    % nombre de slices par zone









%% charge reference 
% —> mesh + signal

fn      = sprintf('%s_mesh_slice_%d.mat', refSubject, refSliceNum);
S    = load(fullfile(baseFolder,refSubject,fn), 'g','H','sigma','edges','z','present_organs');

g_ref    = S.g;      H_ref = S.H+1;        % +1 si vos H étaient 0-based  
sig_ref  = S.sigma;  edges_ref=S.edges;
z_ref = double(S.z);
% 
% c = zeros(size(sig_ref)); % on prépare le vecteur résultat
% 
% for i = 1:9
%     % on trouve toutes les positions où v == i
%     idx = (sig_ref == i);
%     % et on y place b(i)
%     c(idx) = b(i);
% end
% 
% %sig_ref=c;
% 
if size(g_ref,2)>2
  g_ref = g_ref(:,1:2);      %get x,y only  from mesh.nodes of pyeit
end
 

E_ref = generateElectrodesSurfaces(g_ref, edges_ref, n_el, L);

% build forward solver
fmesh_ref = ForwardMesh1st(g_ref,H_ref,E_ref);
solver_ref = EITFEM(fmesh_ref);
solver_ref.mode = 'current';
M = eye(n_el) - circshift(eye(n_el), [0, n_el/2]);   %%%% TO DO : add more injecting protocles, in the function warpAndComputeMeasurements as well
solver_ref.Iel  = M(:);      % vecteur colonne

sig_real = Sig_update(S.present_organs,sig_ref);

Imeas_ref = solver_ref.SolveForwardVec(sig_real);

% [Eref, Imeas_ref] = computeMeasurementsOverRotations( ...
%     g_ref, H_ref, edges_ref, sig_ref, n_el, L, M(:));
[~, bnd_pts_A] = extract_ordered_boundary(g_ref, edges_ref);
dst = resample_contour_by_arclength(bnd_pts_A, Ncontour);
% figure;
% % subplot(2,1,1); plot(Imeas_set); 
% subplot(1,1,1)
% h = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_ref);
% set(h, "facecolor", "interp");
% set(h, "facealpha", 0.6);
% set(h, "EdgeColor", "none");
% axis equal;
% grid on;
% % caxis([0,2])

% colorbar; view(2);

fprintf('La slice de référence est prête\n')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 




%% 

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

%% 3) Pour chaque sujet, assigne une zone basée sur sliceNum
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

%% 

isRefEntry = strcmp({allFiles.subject},refSubject) ...
           & ([allFiles.sliceNum] == z_ref);
assert(sum(isRefEntry)==1, 'Réf introuvable ou dupliquée');
refZone = allFiles(isRefEntry).zone;

mask     = ([allFiles.zone] == refZone);
filtered = allFiles(mask);
filtered = filtered(~strcmp({filtered.subject}, refSubject));

fprintf('Zone #%d retenue → %d slices\n', refZone, sum(mask));


%% 5) Construit la struct results attendue
subjectsF = unique({filtered.subject});
results   = struct('subject',{},'bestIdxSlices',{},'bestFileNames',{});

%% 6) Construire la struct "results" pour warpAndComputeMeasurements

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

%% 
tic
nW = 16;  % par ex. 8 workers
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
%afterEach(dq, @(tk) fprintf('Slice %3d of %s done\n', tk.slice, tk.subject));






warpedResults = warpAndComputeMeasurements( ...
               baseFolder, results, refSubject, refSliceNum, Ncontour,n_el,L,dq);
delete(gcp('nocreate'));

elapsed = toc;         

fprintf('%d slices par sujet sont transformées au contour de la slice de référence et sont résolues',NslicePerZone)
fprintf('Elapsed time: %.4f seconds\n', elapsed);

%% 

% %  Exemples d'accès aux résultats
% for k = 1:numel(warpedResults)
%   fprintf('Sujet %s, slice %3d → %d mesures\n', ...
%     warpedResults(k).subject, ...
%     warpedResults(k).slice, ...
%     numel(warpedResults(k).Imeas_wrapped));
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nsel=5;
M    = numel(warpedResults);
mseV = nan(M,1);

% 1) Calcul de la MSE pour chaque slice
for k = 1:M
    Iw = warpedResults(k).Imeas_wrapped;
    % Assurez‑vous qu'Iw et Imeas_ref ont la même taille
    if numel(Iw) ~= numel(Imeas_ref)
        error('Tailles incompatibles entre Imeas_wrapped (%d) et Imeas_ref (%d).', ...
               numel(Iw), numel(Imeas_ref));
    end
    mseV(k) = mean( (Iw - Imeas_ref).^2 );
end

% 
% bestR = nan(M,1);  % optionnel : index de la rotation qui réalise cette MSE
% 
% for k = 1:M
%     Iset = warpedResults(k).Imeas_wrapped;  % cell-array de mesures
%     nrot = numel(Iset);
%     mse_rot = nan(nrot,1);
% 
%     for r = 1:nrot
%         Iw = Iset{r};  % vecteur de mesures pour la rotation r
% 
%         % vérification de la taille
%         if numel(Iw)~=numel(Imeas_ref{r})
%             error('Tailles incompatibles : %d vs %d', numel(Iw), numel(Imeas_ref{r}));
%         end
% 
%         % calcul de la MSE pour cette rotation
%         mse_rot(r) = mean( (Iw - Imeas_ref{r}).^2 );
%     end
% 
%     % pour le slice k, on prend la rotation qui minimise la MSE
%     [mseV(k), idx_min] = min(mse_rot);
%     bestR(k) = idx_min;
% end


% 2) On trie et on prend les Nsel plus petites MSE
[sortedMSE, sortIdx] = sort(mseV);
count                = min(Nsel, M);
bestIdx              = sortIdx(1:count);



% 3) Affichage des résultats

fprintf('\n=== Top %d slices par MSE croissante ===\n', count);
%% 


figure;
subplot(2,1,1); plot(Imeas_ref); 
subplot(2,1,2)
h = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_real);
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
    g = S.g;   H = S.H+1;   sig = S.sigma; edgesB=S.edges;

    if size(g,2) > 2
        g = g(:,1:2);
    end

    % 2.2) Contour cible et rééchantillonnage
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
    set(h, 'FaceColor','interp', 'FaceAlpha',0.6, 'EdgeColor','none');
    axis equal tight;  grid on;
    title('Mesh ');
    colorbar; view(2);
end









% % 
% % 
% %   c = zeros(size(sig)); % on prépare le vecteur résultat
% % 
% %   for i = 1:9
% %       % on trouve toutes les positions où v == i
% %       idx = (sig_ref == i);
% %       % et on y place b(i)
% %       c(idx) = b(i);
% %   end
% % 
% %   % sig=c;
% % 
% % 
% % 
% % 
% %   fn = fieldnames(S);
% % 
% % 
% %   if size(g,2)>2
% %     g = g(:,1:2);      %get x,y from mesh.nodes of pyeit
% %   end
% % 
% %   E = cell(n_el,1);
% %   for L = 1:n_el
% %      fld = sprintf('E%d', L);
% %      E{L} = S.(fld) + 1;  
% %   end
% % 
% % 
% % 
% % 
% %   fmesh = ForwardMesh1st(g,H,E);
% %   solver = EITFEM(fmesh);
% %   solver.mode = 'current';
% % 
% % 
% %   Imeas = solver.SolveForwardVec(sig);
% %   % for n=0:fmesh.nEl-1
% %   %   block = Imeas(1+n*fmesh.nEl : (n+1)*fmesh.nEl);
% %   %   ii = find(block<0);
% %   %   for j=ii'
% %   %     idx = j + n*fmesh.nEl;
% %   %     Imeas(idx)                          = nan;
% %   %     Imeas(mod(idx, fmesh.nEl)+1)       = nan;
% %   %     Imeas(mod(idx-2, fmesh.nEl)+1)     = nan;
% %   %   end
% %   % end
% % 
% %   allIm{k} = Imeas;
% % 
% %   err(k) = norm( allIm{k} - Imeas_ref );               %%%%%%%
% % 
% %   fprintf('%3d/%3d %s → err = %.3e\n', k,nF,files(k).name,err(k));
% % end
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
% % 
% % 
% % % on prend les 3 plus petits
% % [~,best] = mink(err,3);
% % for i=1:3
% %   k = best(i);
% %   S = load(fullfile(Dir,files(k).name));
% %   g = S.g;   H = S.H+1;   sig = S.sigma;
% % 
% %   c = zeros(size(sig)); % on prépare le vecteur résultat
% % 
% %   for i = 1:9
% %       % on trouve toutes les positions où v == i
% %       idx = (sig == i);
% %       % et on y place b(i)
% %       c(idx) = b(i);
% %   end
% %   % sig=c
% %   figure('Name',sprintf('Top %d : %s',i,files(k).name),'Position',[200 200 600 800]);
% %   subplot(2,1,1);
% %     plot(allIm{k},'-o','LineWidth',1.5);
% %     title(sprintf('%s — Err = %.3e',files(k).name,err(k)));
% %     xlabel('Index d''électrode'); ylabel('Potentiel'); grid on;
% %   subplot(2,1,2);
% %     h = trimesh(H(:,1:3), g(:,1), g(:,2), sig);
% %     set(h, 'FaceColor','interp', 'FaceAlpha',0.6, 'EdgeColor','none');
% %     axis equal tight;  grid on;
% %     title('Mesh ');
% %     colorbar; view(2);
% % end


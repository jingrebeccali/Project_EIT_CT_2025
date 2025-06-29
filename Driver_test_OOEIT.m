%% 1) Réglages de base
clear; close all; clc;
addpath(genpath('C:\Program Files\distmesh-master'))
addpath(genpath('C:\Program Files\OOEIT-main'))

% dossier « référence » (un seul sujet dont on veut afficher mesh+signal)
refDir    = 'meshes_mat_sujets';  
% dossier « bibliothèque » de tous les meshes à comparer
libDir    = 'meshes_mat';          

% nom du fichier référence (vous pouvez aussi prendre le premier trouvé automatiquement)
refFile   = 's0011_mesh.mat';  

 %%%%%%%%% conductivity values, in order of :
 %%%%%%%%% Soft_tissue,lungs,heart,aorta,trachea,esophagus,ribs,vertebraes,scapula,

b = [0.3, 0.15, 0.01, 0.2, 0.6, 0.8, 0.1, 0.25, 0.05];  

 %%%%%%%%%  (TODO : need get the list of organs present in the slice from
 %%%%%%%%% python)

%% charge reference 
% —> mesh + signal
S = load(fullfile(refDir,refFile));
g_ref    = S.g;      H_ref = S.H+1;        % +1 si vos H étaient 0-based  
sig_ref  = S.sigma;  


c = zeros(size(sig_ref)); % on prépare le vecteur résultat

for i = 1:9
    % on trouve toutes les positions où v == i
    idx = (sig_ref == i);
    % et on y place b(i)
    c(idx) = b(i);
end

sig_ref=c;

if size(g_ref,2)>2
  g_ref = g_ref(:,1:2);      %get x,y only  from mesh.nodes of pyeit
end


n_el = 16;       % needs to be the same as pyeit( TODO : get it from pyeit as well)

% recreate  Electrodes surface elementst
E_ref = cell(n_el,1);
for L = 1:n_el
    fld = sprintf('E%d', L);
    
    E_ref{L} = S.(fld) + 1;  
end



% build forward solver
fmesh_ref = ForwardMesh1st(g_ref,H_ref,E_ref);
solver_ref = EITFEM(fmesh_ref);
solver_ref.mode = 'current';

Imeas_ref = solver_ref.SolveForwardVec(sig_ref);




%%%%%%%%%%%%  uncomment if you want to remove some measurments like in your
%%%%%%%%%%%%  driver.Then have to cuncomment this part on line 107 as well
%%%%%%%%%%%%  %%%
% validIdx = ~isnan(Imeas_ref);
% for n=0:fmesh_ref.nEl-1
%   block = Imeas_ref(1+n*fmesh_ref.nEl : (n+1)*fmesh_ref.nEl);
%   ii = find(block<0);
%   for j=ii'
%     idx = j + n*fmesh_ref.nEl;
%     Imeas_ref(idx)                          = nan;
%     Imeas_ref(mod(idx, fmesh_ref.nEl)+1)   = nan;
%     Imeas_ref(mod(idx-2, fmesh_ref.nEl)+1) = nan;
%   end
% end

figure;
subplot(2,1,1); plot(Imeas_ref); 
subplot(2,1,2)
h = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_ref);
set(h, "facecolor", "interp");
set(h, "facealpha", 0.6);
set(h, "EdgeColor", "none");
axis equal;
grid on;
% caxis([0,2])

colorbar; view(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%charge meshs from library

files = dir(fullfile(libDir,'*_mesh.mat'));
nF    = numel(files);
err   = nan(nF,1);
allIm = cell(nF,1);

for k=1:nF
  
  S = load(fullfile(libDir,files(k).name));
  g = S.g; H = S.H+1;  sig = S.sigma;

  c = zeros(size(sig)); % on prépare le vecteur résultat

  for i = 1:9
      % on trouve toutes les positions où v == i
      idx = (sig_ref == i);
      % et on y place b(i)
      c(idx) = b(i);
  end

  sig=c;




  fn = fieldnames(S);


  if size(g,2)>2
    g = g(:,1:2);      %get x,y from mesh.nodes of pyeit
  end

  E = cell(n_el,1);
  for L = 1:n_el
     fld = sprintf('E%d', L);
     E{L} = S.(fld) + 1;  
  end



  
  fmesh = ForwardMesh1st(g,H,E);
  solver = EITFEM(fmesh);
  solver.mode = 'current';

  
  Imeas = solver.SolveForwardVec(sig);
  % for n=0:fmesh.nEl-1
  %   block = Imeas(1+n*fmesh.nEl : (n+1)*fmesh.nEl);
  %   ii = find(block<0);
  %   for j=ii'
  %     idx = j + n*fmesh.nEl;
  %     Imeas(idx)                          = nan;
  %     Imeas(mod(idx, fmesh.nEl)+1)       = nan;
  %     Imeas(mod(idx-2, fmesh.nEl)+1)     = nan;
  %   end
  % end

  allIm{k} = Imeas;
  
  err(k) = norm( allIm{k} - Imeas_ref );               %%%%%%%

  fprintf('%3d/%3d %s → err = %.3e\n', k,nF,files(k).name,err(k));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù


% on prend les 3 plus petits
[~,best] = mink(err,3);
for i=1:3
  k = best(i);
  S = load(fullfile(libDir,files(k).name));
  g = S.g;   H = S.H+1;   sig = S.sigma;

  c = zeros(size(sig)); % on prépare le vecteur résultat

  for i = 1:9
      % on trouve toutes les positions où v == i
      idx = (sig == i);
      % et on y place b(i)
      c(idx) = b(i);
  end
  sig=c
  figure('Name',sprintf('Top %d : %s',i,files(k).name),'Position',[200 200 600 800]);
  subplot(2,1,1);
    plot(allIm{k},'-o','LineWidth',1.5);
    title(sprintf('%s — Err = %.3e',files(k).name,err(k)));
    xlabel('Index d''électrode'); ylabel('Potentiel'); grid on;
  subplot(2,1,2);
    h = trimesh(H(:,1:3), g(:,1), g(:,2), sig);
    set(h, 'FaceColor','interp', 'FaceAlpha',0.6, 'EdgeColor','none');
    axis equal tight;  grid on;
    title('Mesh ');
    colorbar; view(2);
end
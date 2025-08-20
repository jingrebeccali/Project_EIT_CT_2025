%%  Réglages de base
clear; close all; clc;
addpath(genpath('C:\Program Files\distmesh-master'))       % To adapt
addpath(genpath('C:\Program Files\OOEIT-main'))      
addpath(genpath('C:\Users\boulette\Desktop\Depot2\Project_EIT_CT_2025\mat_aatae'))


baseFolder  = 'meshes_mat_sujetss';


refSubject  = 's0011';
refSliceNum = 0;
n_el =16;
L=3.5;                     %% en Cm




%% charge reference 

fn      = sprintf('%s_mesh_slice_%d.mat', refSubject, refSliceNum);
S    = load(fullfile(baseFolder,refSubject,fn), 'g','H','sigma','edges','z','present_organs');

g_ref    = S.g;      H_ref = S.H+1;        % +1 because of python indexation is  0-based  
sig_ref  = S.sigma;  edges_ref=S.edges+1;
z_ref = double(S.z);

% 
if size(g_ref,2)>2
  g_ref = g_ref(:,1:2);      %get x,y only  from mesh.nodes of pyeit
end
%% electrodes placement
 
E= generateElectrodesSurfaces(g_ref,edges_ref, n_el, L);


%%
figure;
plotMeshAndElectrodes(g_ref,H_ref,sig_ref,E)

%%  build forward solver


fmesh_ref = ForwardMesh1st(g_ref,H_ref,E);
solver_ref = EITFEM(fmesh_ref);
solver_ref.mode = 'current';

M = eye(n_el) - circshift(eye(n_el), [0, n_el/2]);      %%% opposed injecting electrodes 


solver_ref.Iel  = M(:);     

sig_real = Sig_update(S.present_organs,sig_ref);   %% update values of conductivity, See Sig_update if you want to change the values.

Umeas = solver_ref.SolveForwardVec(sig_real);     %% solve
%% plot 

figure;
subplot(2,1,1); plot(Umeas); 
subplot(2,1,2)
plotMeshAndElectrodes(g_ref,H_ref,sig_ref,E)


%%  Exzmapl of TPS

fn = sprintf('%s_mesh_slice_%d.mat', 's0065', 110);
S_1  = load(fullfile(baseFolder,'s0065',fn), 'g','H','sigma','edges','z','present_organs','mask');

mask_ref_1 = S_1.mask;

g_ref_1    = S_1.g;      H_ref_1 = S_1.H+1;        % +1 si vos H étaient 0-based  
sig_ref_1  = S_1.sigma;  edges_ref_1=S_1.edges+1;

% 
if size(g_ref_1,2)>2
  g_ref_1 = g_ref_1(:,1:2);      %get x,y only  from mesh.nodes of pyeit
end

E_1= generateElectrodesSurfaces(g_ref_1,edges_ref_1, n_el, L);

Ncontour = 400;
[~, bnd_pts_A] = extract_ordered_boundary(g_ref, edges_ref);
dst = resample_contour_by_arclength(bnd_pts_A, Ncontour);

[~, bndB] = extract_ordered_boundary(g_ref_1, edges_ref_1);
src       = resample_contour_by_arclength(bndB, Ncontour);

[src_a,~,~] = align_contours(src, dst);
tps       = ThinPlateSpline2D().fit(src_a, dst);
nodesW    = tps.transform(g_ref_1(:,1:2));
E_11= generateElectrodesSurfaces(nodesW,edges_ref_1, n_el, L);

%%

figure;
subplot(3,1,1)
plotMeshAndElectrodes(g_ref_1,H_ref_1,sig_ref_1,E_1)

subplot(3,1,2)
plotMeshAndElectrodes(g_ref,H_ref,sig_ref,E)


subplot(3,1,3)

plotMeshAndElectrodes(nodesW,H_ref_1,sig_ref_1,E_1)

%%
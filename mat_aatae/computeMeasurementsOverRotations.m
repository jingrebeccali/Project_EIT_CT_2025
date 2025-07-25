function [Eset, Imeas_set] = computeMeasurementsOverRotations( ...
    g_ref, H_ref, edges_ref, sig_ref, n_el, L, Iel_vec)
%COMPUTEMEASUREMENTSOVERROTATIONS
% Calcule Imeas pour chaque config d'électrodes tournée jusqu'au premier
% overlap. (TO DO : controler le nbre de config, en pourcentage par
% exemple)
%
% [Eset, Imeas_set] = computeMeasurementsOverRotations( ...
%     g_ref, H_ref, edges_ref, sig_ref, n_el, L, Iel_vec)
%
% INPUTS
%   g_ref      : matrice des nœuds du maillage de réf.
%   H_ref      : connectivité du maillage (triangles +1‑based)
%   edges_ref  : liste des arêtes de contour 
%   sig_ref    : conductivités pour la slice de référence
%   n_el       : nombre d'électrodes
%   L          : taille d'électrode en nombre d'arêtes
%   Iel_vec    : vecteur  d'injection courante
%
% OUTPUTS
%   Eset       : cell-array chaque élément est une cell-array N_el×1
%                de matrices L×2 listant les arêtes de cette config
%   Imeas_set  : cell-array  vecteurs de mesures SolveForwardVec(sig_ref)
%
% EXEMPLE
%   M   = eye(n_el) - circshift(eye(n_el),[0,n_el/2]);
%   Iel = M(:);
%   [Eset, Imeas_set] = computeMeasurementsOverRotations( ...
%        g_ref, H_ref, edges_ref, sig_ref, n_el, L, Iel);

  % donne toutes les config possible
  Eset    = rotateElectrodesUntilOverlap(g_ref, edges_ref, n_el, L);
  nConfigs = numel(Eset);

  
  Imeas_set = cell(nConfigs,1);

  
  for k = 1:nConfigs
    E_k = Eset{k};

    
    fmesh_k  = ForwardMesh1st(g_ref, H_ref, E_k);
    solver_k = EITFEM(fmesh_k);
    solver_k.mode = 'current';
    solver_k.Iel  = Iel_vec;

    
    Imeas_set{k} = solver_k.SolveForwardVec(sig_ref);
  end
end

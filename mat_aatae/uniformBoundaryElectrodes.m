function electrodeIdx = uniformBoundaryElectrodes(g, edges, N_el)
%UNIFORMBOUNDARYELECTRODES  Sélectionne N_el électrodes uniformément sur le bord
%   electrodeIdx = uniformBoundaryElectrodes(g, edges, N_el)
% 
%   g      : matrice des coordonnées de tous les nœuds
%   edges  : listes d'arêtes du contour en indices de nœuds
%   N_el   : nombre d'électrodes souhaité
%
%   electrodeIdx : vecteur (N_el×1) d'indices dans g, répartis uniformément

    % Récupérer le contour ordonné
    [bnd_idx, ~] = extract_ordered_boundary(g, edges);
    M = numel(bnd_idx);
    
    % Calculer N_el positions uniformes en indices
    % On génère N_el+1 points de coupe puis on enlève la dernière
    cut = round(linspace(1, M, N_el+1));
    cut(end) = [];
    
    % Mappe sur les indices originaux des nœuds
    electrodeIdx = bnd_idx(cut(:));
end

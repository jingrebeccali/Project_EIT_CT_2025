function E = generateElectrodesSurfaces(g, edges, N_el, L)
%GENERATEELECTRODESSURFACES  Crée N_el électrodes de L arêtes uniformes
%   E = generateElectrodesSurfaces(g, edges, N_el, L)
% 
%   g      : (NA×2) coords des nœuds
%   edges  : (Nb×2) arêtes du contour (indices de nœuds)
%   N_el   : nombre d'électrodes souhaité
%   L      : taille d'électrode en nombre d'arêtes
%
%   E est un cell-array N_el×1, chaque élément L×2 

  % 1) récupère le contour ordonné
  [bnd_idx, ~] = extract_ordered_boundary(g, edges);
  M = numel(bnd_idx);

  maxL = floor(M / N_el);
  if L > maxL
    error(['Taille L = %d trop grande : %d électrodes de longueur %d ' ...
           'ne peuvent pas être placées sur un contour de %d arêtes. ' ...
           'Réduisez L ≤ %d.'], L, N_el, L, M, maxL);
  end

  % centre de chaque électrode
  centers = uniformBoundaryElectrodes(g, edges, N_el);

  halfL = floor(L/2);

  % construction des arêtes pour chaque électrode
  E = cell(N_el,1);
  for i = 1:N_el
    % trouver la position du centre dans bnd_idx
    pos = find(bnd_idx == centers(i), 1);
    if isempty(pos)
      error('Centre electrode non trouvé sur le contour');
    end

    % on prend L arêtes autour de pos : de (pos-halfL) à (pos+halfL)
    nodes_sel = zeros(L+1,1);
    for k = 0:L
      idx = mod((pos - halfL - 1) + k, M) + 1;  
      nodes_sel(k+1) = bnd_idx(idx);
    end

    % chaque arête est une paire consécutive dans nodes_sel
    E{i} = [ nodes_sel(1:L), nodes_sel(2:L+1) ];
  end
end

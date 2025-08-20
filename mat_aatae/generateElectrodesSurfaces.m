function E = generateElectrodesSurfaces(g, edges, N_el, L_cm)
%GENERATEELECTRODESSURFACES  Place N_el electrodes of physical length L_cm (cm)
%   E = generateElectrodesSurfaces(g, edges, N_el, L_cm)
%
%   g      : (NA×2) node coordinates (meters)
%   edges  : (Nb×2) boundary edges (node indices)
%   N_el   : number of electrodes
%   L_cm   : electrode physical length (centimeters)
%
%   E      : N_el×1 cell, each is (K×2) list of boundary edges (node pairs)

  % ---- 1) Ordered boundary and per-segment lengths ----
  [bnd_idx, ~] = extract_ordered_boundary(g, edges);  % M×1 ordered node ids
  M = numel(bnd_idx);
  pts   = g(bnd_idx,:);                 % M×2
  % seg i connects node i -> i+1 (with wrap M->1)
  seglen = sqrt(sum(diff([pts; pts(1,:)],1,1).^2, 2));  % M×1
  totalLen = sum(seglen);

  % ---- 2) Target electrode length (meters) + sanity check ----
  L_m = L_cm / 100;
  maxLen = totalLen / N_el;
  if L_m > maxLen
    error('Electrode %.2f cm too long: max ≈ %.2f cm for %d electrodes.', ...
          L_cm, 100*maxLen, N_el);
  end

  % ---- 3) Electrode centers (indices into bnd_idx) ----
  centers = uniformBoundaryElectrodes(g, edges, N_el);  % node ids
  % map center node ids to their position along bnd_idx
  center_pos = zeros(N_el,1);
  for i = 1:N_el
    pos = find(bnd_idx == centers(i), 1);
    if isempty(pos)
      error('Electrode center not found on ordered boundary.');
    end
    center_pos(i) = pos;
  end

  % ---- 4) Build each electrode by arc-length growth around center ----
  E = cell(N_el,1);

  for i = 1:N_el
    pos = center_pos(i);  % 1..M, node index on boundary

    % Forward half-length: add segments pos, pos+1, ... until >= L_m/2
    len_f = 0;
    kf = pos;
    while len_f < L_m/2
      len_f = len_f + seglen(kf);
      kf = wrap(kf + 1, M);  
    end

    % Backward half-length: add segments pos-1, pos-2, ... until >= L_m/2
    len_b = 0;
    kb = wrap(pos - 1, M);
    while len_b < L_m/2
      len_b = len_b + seglen(kb);
      kb = wrap(kb - 1, M);
    end

    % First node index in nodes_sel is the node AFTER kb
    firstNode = wrap(kb + 1, M);
    lastNode  = kf;

    % Collect nodes from firstNode to lastNode along the boundary (with wrap)
    if firstNode <= lastNode
      nodes_sel = bnd_idx(firstNode:lastNode);
    else
      nodes_sel = [bnd_idx(firstNode:end); bnd_idx(1:lastNode)];
    end

    % Turn consecutive nodes into edge pairs
    if numel(nodes_sel) < 2
      error('Electrode node selection produced <2 nodes — check geometry.');
    end
    E{i} = [nodes_sel(1:end-1), nodes_sel(2:end)];
  end
end

% --- Helpers ---
function k = wrap(k, M)
  % wrap index into 1..M
  k = mod(k-1, M) + 1;
end









% function E = generateElectrodesSurfaces(g, edges, N_el, L)
% %GENERATEELECTRODESSURFACES  Crée N_el électrodes de L arêtes uniformes
% %   E = generateElectrodesSurfaces(g, edges, N_el, L)
% % 
% %   g      : (NA×2) coords des nœuds
% %   edges  : (Nb×2) arêtes du contour (indices de nœuds)
% %   N_el   : nombre d'électrodes souhaité
% %   L      : taille d'électrode en nombre d'arêtes
% %
% %   E est un cell-array N_el×1, chaque élément L×2 
% 
%   % 1) récupère le contour ordonné
%   [bnd_idx, ~] = extract_ordered_boundary(g, edges);
%   M = numel(bnd_idx);
% 
%   maxL = floor(M / N_el);
%   if L > maxL
%     error(['Taille L = %d trop grande : %d électrodes de longueur %d ' ...
%            'ne peuvent pas être placées sur un contour de %d arêtes. ' ...
%            'Réduisez L ≤ %d.'], L, N_el, L, M, maxL);
%   end
% 
%   % centre de chaque électrode
%   centers = uniformBoundaryElectrodes(g, edges, N_el);
% 
%   halfL = floor(L/2);
% 
%   % construction des arêtes pour chaque électrode
%   E = cell(N_el,1);
%   for i = 1:N_el
%     % trouver la position du centre dans bnd_idx
%     pos = find(bnd_idx == centers(i), 1);
%     if isempty(pos)
%       error('Centre electrode non trouvé sur le contour');
%     end
% 
%     % on prend L arêtes autour de pos : de (pos-halfL) à (pos+halfL)
%     nodes_sel = zeros(L+1,1);
%     for k = 0:L
%       idx = mod((pos - halfL - 1) + k, M) + 1;  
%       nodes_sel(k+1) = bnd_idx(idx);
%     end
% 
%     % chaque arête est une paire consécutive dans nodes_sel
%     E{i} = [ nodes_sel(1:L), nodes_sel(2:L+1) ];
%   end
% end

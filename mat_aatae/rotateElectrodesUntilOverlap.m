function Erot = rotateElectrodesUntilOverlap(g, edges, N_el, L)
%ROTATEELECTRODESUNTILOVERLAP  
%   Erot = rotateElectrodesUntilOverlap(g, edges, N_el, L)
%
%   g      : coords des nœuds
%   edges  : table des arêtes du contour
%   N_el   : nombre d'électrodes
%   L      : taille de chaque électrode en nb d'arêtes
%
%   Erot   : cell-array K×1 (K = nombre de rotations avant overlap),
%            chacune est une cell-array N_el×1 de matrices L×2.

  % Configuration de référence
  E0 = generateElectrodesSurfaces(g, edges, N_el, L);

  % Contour ordonné
  [bnd_idx, ~] = extract_ordered_boundary(g, edges);
  M = numel(bnd_idx);

  % positions "centre" pour E0 dans bnd_idx
  centers0 = uniformBoundaryElectrodes(g, edges, N_el);
  pos0     = arrayfun(@(c)find(bnd_idx==c,1), centers0);
  halfL    = floor(L/2);

  % on balaye les décalages successifs
  Erot = {};     
  for shift = 0:(M-1)
    if shift == 0
      E = E0;     
    else
      posS = mod(pos0-1 + shift, M) + 1;
      E    = cell(N_el,1);
      for i = 1:N_el
        p    = posS(i);
        idxs = mod((p-halfL-1) + (0:L), M) + 1;  % indices autour du centre
        nodes= bnd_idx(idxs);                    
        segs = [nodes(1:L), nodes(2:L+1)];       
        segs = sort(segs,2);                     % pour normaliser [a b] vs [b a]
        E{i} = segs;
      end
    end

    % Pour shift>0 : test d'overlap avec E0
    if shift > 0
      overl = false;
      for i = 1:N_el
        for j = 1:N_el
          % si la i‑ème électrode tourné = j‑ème électrode de base
          if isequal(sortrows(E{i}), sortrows(E0{j}))
            overl = true;
            break
          end
        end
        if overl, break, end
      end
      if overl
        break   % on s'arrête quand on trouve un overlap
      end
    end

    Erot{end+1,1} = E;
  end
end

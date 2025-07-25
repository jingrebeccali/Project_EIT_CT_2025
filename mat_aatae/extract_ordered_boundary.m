function [bnd_idx, bnd_pts] = extract_ordered_boundary(node, bars)
% EXTRACT_ORDERED_BOUNDARY  
    %%%
    % Construit la liste ordonnée des nœuds sur un contour, définit par la
    % liste des arrêts bars

    % node : tableau des coordonnées des sommets d'un mesh
    % bars : liste des arrêts du contour du mesh.
    % bnd_idx : tableau des indices des points sur le contour (dans l'ordre
    % CCW)
    % bnd_pts : tableau des coordonnées des points sur le contour( dans
    % l'ordre CCW)

    %%%
    
    % les edges viennent de Python, ajoutez 1 pour passer à l'indexation
    % MATLAB
    if min(bars(:)) == 0
        bars = bars + 1;
    end
    %construis le dictionnaire d'adjacence sur le contour
    %chaque noeud y apparaît exactement dans 2 arêtes 
    uniq = unique(bars(:));
    neigh = containers.Map(uniq, repmat({[]}, numel(uniq),1));
    for k = 1:size(bars,1)
        i = bars(k,1); j = bars(k,2);
        neigh(i) = [neigh(i), j];
        neigh(j) = [neigh(j), i];
    end


    %parcours la boucle
    start = bars(1,1);
    boundary = start;
    prev = NaN;
    curr = start;
    while true
        nbrs = neigh(curr);
        
        if nbrs(1) ~= prev  % choisis le suivant : celui qui n'est pas le précédent
            nxt = nbrs(1);
        else
            nxt = nbrs(2);
        end
        if nxt == start
            break;
        end
        boundary(end+1,1) = nxt; %#ok<AGROW>
        prev = curr;
        curr = nxt;
    end

    bnd_idx = boundary;
    bnd_pts  = node(bnd_idx, 1:2);
end

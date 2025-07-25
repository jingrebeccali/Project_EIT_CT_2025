function d = hausdorff_distance(ptsA, ptsB)
%HAUSDORFF_DISTANCE   Distance de Hausdorff entre deux maillages
%   d = hausdorff_distance(meshA, meshB) calcule la distance de Hausdorff
%   entre les nuages de points meshA.node et meshB.node.
%



    % Matrice des distances euclidiennes entre chaque paire de points
    D = pdist2(ptsA, ptsB);  
    % distance dirigée A→B : pour chaque point de A, la plus proche distance à B
    d_ab = max( min(D, [], 2) );  
    % distance dirigée B→A : pour chaque point de B, la plus proche distance à A
    d_ba = max( min(D, [], 1) ); 
    % Hausdorff : le plus grand des deux
    d = max(d_ab, d_ba);
end

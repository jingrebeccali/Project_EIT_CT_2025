% resample_contour_by_arclength.m
function new_pts = resample_contour_by_arclength(pts, N)
% new_pts = resample_contour_by_arclength(pts, N)
%   pts : K×2 points fermés (le 1er et le dernier peuvent différer)
%   N   : nombre de points désiré 
%
%   Retourne N points uniformément répartis le long du contour du mesh
%   Cette fontion est uniquement utilisé avant ThinPlateSpline2D.
%   N doit être relativement grand pour capturer le plus de détails possible du
%   contour.(N=400 suffit)
%   L'utilité de cette fonction est d'obtenir un meme nbre de points
%   representant deux contours, car ceci est indispensable pour
%   ThinPlateSpline2D.
%   
%   L'interpolation est faite sur la longeur du périmétre.(il faut peut
%   être mieux le faire en angle, mais cela nécessite to trouver le
%   centroide du domaine, plus de calcul..)
%
%

    % fermer la boucle
    if ~isequal(pts(1,:), pts(end,:))
        pts = [pts; pts(1,:)];
    end
    
    % longueurs des segments
    seg_vecs    = diff(pts,1,1);           
    seg_lengths = sqrt(sum(seg_vecs.^2,2)); 
    
    cumdist = [0; cumsum(seg_lengths)];    
    total_length = cumdist(end);
    
    sample_d = linspace(0, total_length, N+1)';
    sample_d(end) = [];  % on ne répète pas le dernier ( ni le premier)
    
    new_pts = zeros(N,2);
    for i = 1:N
        d = sample_d(i);
        idx = find(cumdist <= d, 1, 'last');  %trouver où tombe le i-ième sample, sur le contour
        if idx == length(cumdist)
            idx = idx - 1;                    %Récuperer l'indice du point
        end
        t = (d - cumdist(idx)) / seg_lengths(idx);
        new_pts(i,:) = (1-t)*pts(idx,:) + t*pts(idx+1,:);  %Interpolation entre le ponit trouvé et le suivant
    end
end

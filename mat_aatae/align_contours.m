% align_contours.m
function [src_aligned, best_shift, reversed_flag] = align_contours(src, dst)
% [src_aligned, shift, rev] = align_contours(src, dst)
%   src, dst :  contours échantillonnés, du meme nbre de points Nx2
%
%   src_aligned : src tourné ou inversé (ou les deux
%   best_shift  : nombre de pas de rotation circulaire(si il y en a)
%   reversed_flag : true si inversion src(end:-1:1,:)
%
%   Cette fonction aligne les deux nuages de points src et dst dans le meme
%   sense, et assure que l'on commence à partir du meme point.Ceci est
%   indispensable pou rassurer une bonne transformation à
%   ThinPlateSpline2D.

%   Le principe est de minimiser la somme des distances au carré, point à point.


    N = size(src,1);
    best_err = inf;
    best_shift = 0;
    reversed_flag = false;
    
    for rev = [false, true]
        if ~rev
            cand = src;
        else
            cand = flipud(src);
        end
        for k = 0:N-1
            rolled = circshift(cand, -k, 1);
            err = sum(sum((rolled - dst).^2));
            if err < best_err
                best_err = err;
                best_shift = k;
                reversed_flag = rev;
            end
        end
    end
    
    if reversed_flag
        src = flipud(src);
    end
    src_aligned = circshift(src, -best_shift, 1);
end

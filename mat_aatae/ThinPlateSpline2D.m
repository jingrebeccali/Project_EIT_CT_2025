% ThinPlateSpline2D.m
classdef ThinPlateSpline2D
    %%%% 
    % Implémentation de Thin Plate Spline interpolation, largement inspiré la version python de de https://github.com/djeada/Numerical-Methods/blob/master/notes/6_regression/thin_plate_spline_interpolation.md
    %Pour plus d'informations : https://pypi.org/project/thin-plate-spline/

    %%%%
    properties
        w   % coefficients non-affines (Mx2)
        a   % coefficients affines (3x2)
        src % points source (Mx2)
    end
    
    methods
        function obj = ThinPlateSpline2D()
            obj.w   = [];
            obj.a   = [];
            obj.src = [];
        end
        
        function K = kernel(~, r)
            K = r.^2 .* log(r);
            K(isnan(K) | isinf(K)) = 0;
        end
        
        function obj = fit(obj, src, dst)
            % src et dst sont M×2
            M = size(src,1);
            obj.src = src;
            
            % calcul manuel de la matrice des distances D entre src et src
            XX = sum(src.^2, 2);              
            D2 = bsxfun(@plus, XX, XX') ...   
               - 2*(src*src');
            D2(D2<0) = 0;                     % pour éviter les qqe négatifs
            D  = sqrt(D2);
           
            
            K = obj.kernel(D);
            P = [ones(M,1), src];            
            
            A = zeros(M+3);
            A(1:M,1:M)       = K;
            A(1:M,M+1:M+3)   = P;
            A(M+1:M+3,1:M)   = P';
            
            Y = [dst; zeros(3,2)];
            sol = A \ Y;
            
            obj.w = sol(1:M, :);
            obj.a = sol(M+1:end, :);
        end
        
        function pts_t = transform(obj, pts)
            % 
            N = size(pts,1);
            M = size(obj.src,1);
            
            %
            XX = sum(pts.^2, 2);             
            YY = sum(obj.src.^2, 2)';        
            D2 = bsxfun(@plus, XX, YY) ...    
               - 2*(pts * obj.src');
            D2(D2<0) = 0;
            D  = sqrt(D2);
            
            
            U = obj.kernel(D);
            non_affine = U * obj.w;           
            
            P = [ones(N,1), pts];            
            affine = P * obj.a;               
            
            pts_t = non_affine + affine;
        end
    end
end

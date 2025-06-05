clear all;

format compact;

addpath distmesh-master/
 
load ct_mat.mat;

Slices_folder_name = ['Slices_folder'];

nz = size(ct_mat.img,3);

% Start and end slice
ns = 201;
nf = 210;




for iz= ns:nf
    
    slice_tmp = squeeze(ct_mat.img(:,:,iz));

%     figure; mesh(slice_tmp); colorbar;

    slice  = slice_tmp;
    
    % Keep region of interest above threshhold 
    min_color = -500;
    BW = (slice_tmp > min_color);
    
    % Get the different 8-connected components
    L = bwlabel(BW, 8);
    
    % Get and keep largest connected compenent
    counts = histcounts(L(L>0), 1:(max(L(:))+1));
    [~, idxMax] = max(counts);
    mainBlob = (L == idxMax);
    
    % Fill holes
    mainBlob_filled = imfill(mainBlob, 'holes');

    premask = mainBlob_filled;

    % Resize for quicker processing
    mask = imresize(premask,0.25,'nearest');
    mask = logical(mask);
    
    % Resize the sigma values
    slice_new = imresize(slice,0.25);
    
    % Make the sigma values positive
    slice_pos = (slice_new - (min(min(slice_new))));
    
    % Make the sigma values between 0 and 1
    slice_norm = slice_pos/(max(max(slice_pos)));
    
    % Apply the mask
    slice_clean = zeros(size(slice_new));
    slice_clean(mask==1) = (slice_norm(mask==1)+1);
    
    slice_clean2 = slice_clean;

%     figure; mesh(mask); colorbar;
    
    % First triangulation attempt
    [jj,kk] = find(mask);

    shp = alphaShape(kk, jj);

    facets = shp.boundaryFacets;
    points = shp.Points;

%     figure;
%     plot(points(facets(:,1),1), points(facets(:,1),2), 'ro');

    % Trying to add a smooth layer of boundary points, take the exisitng
    % alphashape boundary points and do an interpolation in the angle using
    % high order Fourier interpolation. Angle measured by making the origin
    % nx/2, ny/2.
    ii = facets(:,1);

    nx = size(mask,1);
    ny = size(mask,2);

    ang = atan2(points(ii,2)-nx/2,points(ii,1)-ny/2)/pi;

    fx_angle = points(ii,1)-nx/2;
    fy_angle = points(ii,2)-ny/2;

    ang_new = ang;

    % Evalute the smooth interpolation at the angles, multiply the distance
    % from the center (nx/2, ny/2) by 1.1. Adds a smooth boundary, making
    % the size pf the radius 10 % bigger.
    fitx = fit(ang,fx_angle,"fourier8");
    fx_fit = 1.1*fitx(ang_new)+nx/2;

    fity = fit(ang,fy_angle,"fourier8");
    fy_fit = 1.1*fity(ang_new)+ny/2;
 
    % Pass the smooth boundary pv into Distmesh
    pv = [fx_fit,fy_fit];

   

    fd = { 'l_dpolygon', [], pv };
    fh = @(p) ones(size(p,1),1);

    % p and t are the nodes and triangles of the triangulation
    [p,t] = distmesh( fd, fh, 2, [0,0; nx,ny], pv );

%     figure; patch( 'vertices', p, 'faces', t, 'facecolor', [.9, .9, .9] );
    
    % Look for the boundary of the triangluation, easier to use the Matlab
    % functions
    TR = triangulation(t,p);
    aa = TR.edgeAttachments(TR.edges);
    all_edges = TR.edges;
    ne = length(aa);
    % Edge is a boundary edge if it only has one triangle attached to it
    bd_edges = [];
    for ie = 1:ne
        if (length(aa{ie})==1)
            bd_edges = [bd_edges; all_edges(ie,1:2)];
        end
    end

    % Want to order the boundary edges by connection
    points = TR.Points;
    facets = bd_edges;

    facets_ord = [facets(1,1:2)];

    nf = size(facets,1);

    for ifa = 2:nf

        kk = find(facets_ord(ifa-1,2) == facets(1:end,1));
        if (length(kk) == 1)
            facets_next = facets(kk,1:2);
            facets(kk,1:2) = [nan,nan];
        elseif (length(kk) >= 2)
            disp('wrong 2');
            stop
        elseif (length(kk) == 0)
            kk = find(facets_ord(ifa-1,2) == facets(1:end,2));
            if (length(kk) == 1)
                facets(kk,:) = facets(kk,2:-1:1);
                facets_next = facets(kk,1:2);
                facets(kk,1:2) = [nan,nan];
            else
                disp('wrong 1');
                stop
            end

        end

        facets_ord = [facets_ord;facets_next];

    end
    
    % facets_ord is an ordered set of edges, ne1 is the number of
    % electrodes
    nel = 16;

    facets = facets_ord;
    nfa = size(facets,1);
    ii = facets(:,1);
    
    % vec is the starting node of each electrode
    vec = floor(linspace(1,nfa,nel+1));
    % Each electrode has 4 segments. We need to be able to let user decide
    % how many segments in each electrode as long as they do not overlap
    % with other electrodes. Each electrode segment should correspond to a
    % physiscal length that the user inputs. To be done. 
    for ie = 1:nel
        elfaces{ie} = facets(vec(ie):vec(ie)+4,1:2);
    end

    % Interpolating the Cartesian sigma from the slice to the nodes in the
    % triangulation.
    % Hardwired the physical lengths of the resized slice. Needs to be
    % changed. 
    xmin = 1;
    xmax = 78;
    ymin = 1;
    ymax = 78;
    
    % xsamp and ysamp are the triangulation nodes
    xsamp = points(:,1);
    ysamp = points(:,2);

    nx = size(slice,1);
    ny = size(slice,2);

    xvec = linspace(xmin,xmax,nx);
    yvec = linspace(ymin,ymax,ny);
    
    % Making slice_tmp2 between 0 and 1 again, maybe logical conversion to
    % double. Maybe need to change slice at the start of the code to
    % double.
    slice_tmp = slice - min(min(slice));
    slice_tmp2 = double(slice_tmp)/double(max(max(slice_tmp)));
    
    % Interpolated value at the triangulation nodes. Consider using
    % 'linear' instead of 'nearest'.
    Vq = interp2(xvec,yvec,slice_tmp2,points(:,1),points(:,2),'nearest');

    np = size(points,1);

    % sigma is defined at the interpolated values
    sig = Vq;
    
    % Here, trying to make sure sigma around the boundary is a constant
    % value corresponding to a consistent value for the boundary. Makes the
    % EIT solver more robust. We hardwire the gap size to be 3. Needs a
    % physical length. Hardwire the value to be 0.5, needs to adapt to the
    % sigma near the boundary. Could set it to the average all sigma values 
    % around the boundary.
    nfa = size(facets,1);
    bdy_gap = 3;

    for ifa = 1:nfa
        dist_fa = sqrt((points(:,1)-points(facets(ifa,1),1)).^2+(points(:,2)-points(facets(ifa,1),2)).^2);
        ll = find(dist_fa <= bdy_gap);
        sig(ll) = 0.5;
    end

   
    
    g = points;
    H = TR.ConnectivityList;

    figure;
    h = trimesh(H(:,1:3), points(:,1), points(:,2),sig);
    set(h, "facecolor", "interp");
    set(h, "facealpha", 0.6);
    set(h, "EdgeColor", "b");
    axis equal;
    grid on;

    colorbar;
    % caxis([0,1])
    view(2)

    % Nodes g, triangles H, electrode edges elfaces, and conductivity sig
    % are saved for each slice, to be used in EIT solver.
    file_name = [Slices_folder_name,'\slice_OOEIT_',num2str(iz)];
    save(file_name,'g','H','elfaces','sig');

end



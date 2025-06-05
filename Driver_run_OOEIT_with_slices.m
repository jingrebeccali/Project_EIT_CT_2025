clear all;
close all;

format compact;
addpath distmesh-master/
 

Slices_folder_name = ['Slices_folder'];



%Base slice taken as measurement
iz_sol = 209;

iz = iz_sol;

%Get the slice from the slice folder where g, H, elfaces, and sig are
%located
file_name = [Slices_folder_name,'\slice_OOEIT_',num2str(iz)];
load(file_name);

%Create a mesh-object from the loaded variables g, H and elfaces 
mesh = ForwardMesh1st(g, H, elfaces);

%Initialize the forward problem solver (FPS).
solver = EITFEM(mesh);

%Set the injection mode to potential injection. 
solver.mode = 'potential';

%Get the conductivity distribution
sigma = sig;

%Compute the forward problem which serves as the measurement
Imeas = solver.SolveForwardVec(sigma);

%Set where the electrodes are injected current to NaN as well as the left
%and right neighbors to focus on the relevant measurements (disregard the
%negative spikes and their neighbors).
for n = 0:15
    ii = find(Imeas(1+n*16:16+n*16)<0);
    iip = ii+1;
    iim = ii-1;

    iipp = mod(iip-1, 16)+1;
    iimm = mod(iim-1, 16)+1;

    Imeas(ii+n*16) = nan;
    Imeas(iipp+n*16) = nan;
    Imeas(iimm+n*16) =nan;
end

%Set the solution elements
signal_sol = Imeas;
nodes_sol=g;
elem_sol=H;
cond_sol = sig;

%Take slices to compare
zvec = [201:210];

nnz = length(zvec);

%Load each slice, create the mesh, initialize the FPS, get the conductivity
%distribution, and compute the forward problem.
for izz = 1:nnz

    iz = zvec(izz);
    file_name = [Slices_folder_name,'\slice_OOEIT_',num2str(iz)];
    
    load(file_name);

    mesh = ForwardMesh1st(g, H, elfaces);

    solver = EITFEM(mesh);

    solver.mode = 'potential';

    sigma = sig; 

    Imeas = solver.SolveForwardVec(sigma);

    signal{iz} = Imeas;
    nodes{iz}=g;
    elem{iz}=H;
    cond{iz} = sig;


end

%Initialize the error vector to be 0
err_vec = zeros(nnz,1);

%Get indices where we are taking the relevant measurments from
innan = ~isnan(signal_sol);

%For each slice we are comparing, only keep relevant measurements by
%setting where the current is injected to NaN
for izz = 1:nnz

    iz = zvec(izz);

    Imeas = signal{iz};

    for n = 0:15
        ii = find(Imeas(1+n*16:16+n*16)<0);
        iip = ii+1;
        iim = ii-1;

        iipp = mod(iip-1, 16)+1;
        iimm = mod(iim-1, 16)+1;

        Imeas(ii+n*16) = nan;
        Imeas(iipp+n*16) = nan;
        Imeas(iimm+n*16) =nan;
    end

    %Set the value of the error to the two norm of the difference between
    %the slice and the measurement
    err_vec(izz,1) = norm(Imeas(innan)-signal_sol(innan));

end

figure;
plot(zvec,err_vec,'x');
title('Norm 2 Errors of the slices in the library');

grid on;

%Number of closest fits to consider
nbest = 3;

%Get the error indices of the smallest errors
[values, smallest_n] = mink(err_vec, nbest);

%For each of the nbest closest errros, get the slice number, and plot the
%signal and mesh
for index = 1:length(smallest_n)
    
    isol = smallest_n(index);

    izsol = zvec(isol);

    g = nodes{izsol};
    H = elem{izsol};
    Imeas = signal{izsol};
    sig = cond{izsol};

    figure;
    subplot(2,1,1); plot(Imeas); 
    subplot(2,1,2)
    h = trimesh(H(:,1:3), g(:,1), g(:,2), sig);
    set(h, "facecolor", "interp");
    set(h, "facealpha", 0.6);
    set(h, "EdgeColor", "none");
    axis equal;
    grid on;
    % caxis([0,2])

    colorbar; view(2);

end

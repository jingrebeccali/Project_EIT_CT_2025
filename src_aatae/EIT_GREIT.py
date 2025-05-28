import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib.pyplot as plt
import napari
from scipy.ndimage import binary_closing, binary_fill_holes,label
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from Extract_skin_mask import *
import pyeit.mesh as mesh
from pyeit.eit.fem    import EITForward
import pyeit.eit.protocol as protocol
import pyeit.eit.greit    as greit
from pyeit.visual.plot    import create_mesh_plot, create_plot
from pyeit.mesh.shape import *





#To adapt
base_dir=r'Data_set'
case_id="s0011"    
organ_parts= [
    "lung_lower_lobe_left.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_middle_lobe_right.nii.gz",
    "lung_upper_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz",
]
    

mask_eit,organ_mask,body_mask,outside_mask,body,ct= extract_skin_mask(case_id, organ_parts, z=340)

body_2D= body[:,:,340] 


show_masks_eit(mask_eit) #show the mask in 2D
show_masks_eit(body_2D)  #show the body mask in 2D



# ###########   extract the contour of the body mask(view Exract_skin_mask.py for details)  #################
n_el = 8  # Number of electrodes, for some reason, the number of electrodes must be a multiple of 4
p_fix=Electrodes_position(body_2D,n_el)


######################  Create the mesh and the protocol ####################
############ Same code structure as in the folder examples in pyeit ################


mesh_obj = mesh.create(
    n_el    = n_el ,
    h0      = 0.05,   # resolution of the mesh(less it is, more triangles you have)
    p_fix   = p_fix,         # points fixed on the contour
    fd      = circle,         
    fh      = area_uniform,
)
pts      = mesh_obj.node     
tri      = mesh_obj.element   
tri_centers = mesh_obj.elem_centers 

def world2pix(xy, shape):
    u = ((xy[:,0] + 1)/2) * (shape[1] - 1)
    v = ((1 - (xy[:,1] + 1)/2)) * (shape[0] - 1)
    return np.vstack([v, u]).T

pix    = world2pix(tri_centers, mask_eit.shape).astype(int)
labels = mask_eit[pix[:,0], pix[:,1]]

# Choice of conductivity values 
sigma_map = {0: -10, 1: 0.1, 2:10} # Outside of body : -10, body : 0.1, lungs: 10   respectivement
perm0 = np.ones(mesh_obj.element.shape[0])       # sigma reference values

perm = np.array([sigma_map[l] for l in labels])


# simulation with greit
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
fwd = EITForward(mesh_obj, protocol_obj)
eit = greit.GREIT(mesh_obj, protocol_obj)
eit.setup(p=0.20, lamb=0.01, perm=1.0, jac_normalized=True)

#it is important to set the permittivity of the mesh object before solving
mesh_obj.perm = perm
delta_perm = np.real(mesh_obj.perm - perm0)

v0 = fwd.solve_eit(perm=perm0)
v1 = fwd.solve_eit(perm=perm)

ds   = eit.solve(v1, v0, normalize=True)   ##the parameter normalize=True gives arbitrary units(to visualize), if you want to have the real values, do normalize=False

# 2D grid crossing
xg, yg, grid_ds = eit.mask_value(ds, mask_value=np.nan)


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,5))

#plots the mesh and the electrodes
create_mesh_plot(ax0, mesh_obj,
                 electrodes=mesh_obj.el_pos,
                 coordinate_labels="radiological",
                 marker_text_kwargs={ "color": "red", "fontsize": 6 })
ax0.set_title("mesh + électrodes")
ax0.axis("off")

# plot the reconstructed GREIT conductivity
im = ax1.imshow(
    np.real(grid_ds),
    origin="lower",
    extent=[xg.min(), xg.max(), yg.min(), yg.max()],
    interpolation="none",
    cmap=plt.cm.viridis
)
ax1.set_title("GREIT reconstruit Δσ")
ax1.axis("off")
ax1.set_aspect("equal")


div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes("right", size="5%", pad=0.05)
cb0  = plt.colorbar(ax0.collections[0], cax=cax0)
cb0.set_label("Element Value")


div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.05)
cb1  = plt.colorbar(im, cax=cax1)
cb1.set_label("σ (S/m)")

plt.tight_layout()
plt.show()
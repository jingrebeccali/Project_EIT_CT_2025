

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes,label
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point
from skimage import measure
import imageio.v2 as imageio

import pyeit.mesh as mesh
from pyeit.eit.fem    import EITForward
from pyeit.mesh.mesh_img import groundtruth_IMG_based
from pyeit.eit.interp2d import sim2pts

import pyeit.eit.protocol as protocol
import pyeit.eit.greit    as greit
import pyeit.eit.bp       as bp
import pyeit.eit.jac as jac
from pyeit.visual.plot    import create_mesh_plot, create_plot
from pyeit.mesh.shape import *

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from Extract_skin_mask import *
from help_functions import *
from EIT_sim import *

########### Example usage of the Test_BP function: maks are already generated, and placed in library_masks.##########

case_id    = "s0011"
output_dir = os.path.join("library_masks", case_id)
z_slice    = 272  # Check library masks for available slices


png_name = f"mask_{case_id}_z{z_slice:03d}.png"
png_path = os.path.join(output_dir, png_name)

img_rgba = imageio.imread(png_path)  
mask_eit = (img_rgba ).astype(np.uint8)



#### ######""  if you want to  create your mask    ####### ####
# #To adapt
# base_dir=r'Data_set'
# case_id="s0011"    
# organ_parts = [
#     "lung_lower_lobe_left.nii.gz",
#     "lung_lower_lobe_right.nii.gz",
#     "lung_middle_lobe_right.nii.gz",
#     "lung_upper_lobe_left.nii.gz",
#     "lung_upper_lobe_right.nii.gz",
# ]
# skin=["skin.nii.gz"]

# skin_mask,outside,lungs_mask=Create_skin_mask_bis(case_id,organ_parts)
# mask_eit=Create_mask_2D([lungs_mask,skin_mask],340)



protocol_obj = set_protocol(n_el=16, dist_exc=1, step_meas=1) # To adpt

mask_eit,mesh_obj,ds,perm0,v0, v1,eit,protocol_obj=Test_BP("BP",mask_eit, condu_body=0.3, condu_lung=0.15, n_el=16, h0=0.06, protocol_obj=protocol_obj) ## To adapt

view_EIT_BP(mask_eit,mesh_obj,ds,perm0,v0, v1,eit,protocol_obj) 

display_v0_v1(mesh_obj, protocol_obj, perm0, mesh_obj.perm)  # Display the potentials v0 and v1 for the mesh and protocol
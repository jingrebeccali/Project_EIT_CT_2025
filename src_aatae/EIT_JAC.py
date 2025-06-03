


import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import napari
from scipy.ndimage import binary_closing, binary_fill_holes,label
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point
from skimage import measure

import pyeit.mesh as mesh
from pyeit.eit.fem    import EITForward
from pyeit.mesh.mesh_img import groundtruth_IMG_based

import pyeit.eit.protocol as protocol
import pyeit.eit.greit    as greit
import pyeit.eit.bp       as bp
import pyeit.eit.jac as jac
from pyeit.visual.plot    import create_mesh_plot, create_plot
from pyeit.mesh.shape import *

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyeit.eit.interp2d import sim2pts


from Extract_skin_mask import *
from help_functions import *

#To adapt
base_dir=r'Data_set'
case_id="s0011"    
organ_parts = [
    "lung_lower_lobe_left.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_middle_lobe_right.nii.gz",
    "lung_upper_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz",
]
skin=["skin.nii.gz"]

skin_mask,outside,lungs_mask=Create_skin_mask_bis(case_id,organ_parts)

mask_eit=Create_mask_2D([lungs_mask,skin_mask],340)

#show_masks_eit(mask_eit)




def Test_JAC(maks_eit,n_el, h0):
        """
        Test the BP method on a given mask and mesh parameters.
        :param mask_eit: 2D mask for EIT
        :param n_el: Number of electrodes

        :param h0: Resolution of the mesh
        :return: None, plots the result of the simulation
        """


        # ###########   extract the contour of the body mask(view Exract_skin_mask.py for details)  #################
        # Number of electrodes, for some reason, the number of electrodes must be a multiple of 4
        body_poly = Extract_contour(mask_eit)
        fd_body = make_fd_body(body_poly)

        ######################  Create the mesh and the protocol ####################
        ############ Same code structure as in the folder examples in pyeit ################


        
        mesh_obj = mesh.create(
            n_el    = n_el ,
            h0      = h0,   # resolution of the mesh(less it is, more triangles you have)
            fd      = fd_body,         
            fh      = area_uniform,
        )
        
        pts      = mesh_obj.node      
        tri      = mesh_obj.element  
        tri_centers = mesh_obj.elem_centers 




        labels_elems = compute_element_labels(mask_eit, pts, tri)
        Ne = tri.shape[0]
        cond_body = 10.0
        cond_lung = 0.001
        perm0 = np.ones(Ne)
        perm = perm0.copy()
        perm[labels_elems == 2] = cond_lung 
        perm[labels_elems == 1] = cond_body
        #no label_elems = 0; ignore outside the body
        mesh_obj.perm = perm 



        protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1)
        fwd = EITForward(mesh_obj, protocol_obj)

        v0 = fwd.solve_eit(perm=perm0)  # initial potential with perm0
        v1 = fwd.solve_eit(perm=mesh_obj.perm)


        eit = jac.JAC(mesh_obj, protocol_obj)
        eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
 


        ds = eit.solve(v1, v0, normalize=True) ##the parameter normalize=True gives arbitrary units(to visualize), if you want to have the real values, do normalize=False

        ds_n = sim2pts(pts, tri, np.real(ds))

        fig, axes = plt.subplots(1, 2, figsize=(12,5))

        ax0=axes[0]



        #plots the mesh and the electrodes
        create_mesh_plot(ax0, mesh_obj,
                        electrodes=mesh_obj.el_pos,
                        coordinate_labels="radiological",
                        marker_text_kwargs={ "color": "red", "fontsize": 6 })
        ax0.set_title("mesh + électrodes")
        ax0.axis("off")

        # ax0.axis("equal")
        # ax0.set_title(r"Input $\Delta$ Conductivities")
        # im = ax0.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")



        #plot the reconstructed BP conductivity
        ax1=axes[1]

        im = ax1.tripcolor(pts[:,0], pts[:,1], tri,
                                ds_n,shading='flat', cmap='jet')

        ax1.set_title("BP reconstruit Δσ")
        ax1.axis("off")
        ax1.set_aspect("equal")

        fig.colorbar(im, ax=axes.ravel().tolist())
        # fig.savefig('../doc/images/demo_bp.png', dpi=96)
        plt.show()



Test_JAC(mask_eit, n_el=8, h0=0.1)  # h0 is the resolution of the mesh









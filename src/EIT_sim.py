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

def Test_BP(solver:str,mask_eit, condu_body, condu_lung, n_el, h0, protocol_obj):
        """
        Test the BP method on a given mask and mesh parameters.
        :param mask_eit: 2D mask for EIT
        :param n_el: Number of electrodes
        :param h0: Resolution of the mesh
        :param condu_body: Conductivity of the body
        :param condu_lung: Conductivity of the lungs
        :param protocol_obj: Protocol object for EIT : defiend before calling this function


        :return:
            mesh_obj: Mesh object created from the mask
            ds_bp: BP reconstruction result
            perm0: Initial conductivity distribution
            v0: Initial potential
            v1: Potential after applying the BP method
            eit_bp: EIT BP object
            protocol_obj: Protocol object used for EIT
        """


        # ###########   extract the contour of the body mask(view Exract_skin_mask.py for details)  #################
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

        perm,perm0=set_condu_eit(mesh_obj,condu_body=condu_body, condu_lung=condu_lung, labels_elems=labels_elems)

        protocol_obj = protocol_obj
        fwd = EITForward(mesh_obj, protocol_obj)
        v0 = fwd.solve_eit(perm=perm0)  # initial potential with perm0
        v1 = fwd.solve_eit(perm=mesh_obj.perm)
        if solver.upper() == "BP":
            
            eit = bp.BP(mesh_obj, protocol_obj)
            eit.setup(weight="none")  


            ds = eit.solve(v1, v0, normalize=True)   #the parameter normalize=True gives arbitrary units(to visualize), if you want to have the real values, do normalize=False
        if solver.upper() == "JAC":
                eit = jac.JAC(mesh_obj, protocol_obj)
                eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
 


                ds = eit.solve(v1, v0, normalize=True)

                ds = sim2pts(pts, tri, np.real(ds))

        return mask_eit,mesh_obj,ds,perm0,v0, v1,eit,protocol_obj


def view_EIT_BP(mask_eit,mesh_obj, ds,perm0, v0, v1, eit, protocol_obj):
        """ 
        Visualize the EIT BP results.
        :param mesh_obj: Mesh object created from the mask
        :param ds_bp: BP reconstruction result
        :param perm0: Initial conductivity distribution
        :param v0: Initial potential
        :param v1: Potential after applying the BP method
        :param eit_bp: EIT BP object
        :param protocol_obj: Protocol object used for EIT
        :return: None
        """

        delta_perm = np.real(mesh_obj.perm - perm0)
        
        pts      = mesh_obj.node      
        tri      = mesh_obj.element  
        tri_centers = mesh_obj.elem_centers 


        fig, axes = plt.subplots(1, 3, figsize=(12,5))

        axes[0].imshow(mask_eit, cmap='viridis')
        axes[0].set_title("Masque (0=fond,1=jaune,2=vert)")
        axes[0].axis('off')


        ax1=axes[1]
        #plots the mesh and the electrodes
        create_mesh_plot(ax1, mesh_obj,
                        electrodes=mesh_obj.el_pos,
                        coordinate_labels="radiological",
                        marker_text_kwargs={ "color": "red", "fontsize": 6 })
        ax1.set_title("mesh + électrodes")
        ax1.axis("off")

        # ax0.axis("equal")
        # ax0.set_title(r"Input $\Delta$ Conductivities")
        # im = ax0.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")



        #plot the reconstructed BP conductivity
        ax2=axes[2]
        im = ax2.tripcolor(pts[:,0], pts[:,1], tri,ds,shading='flat', cmap='jet')
        ax2.set_title("BP reconstruit Δσ")
        ax2.axis("off")
        ax2.set_aspect("equal")
        fig.colorbar(im, ax=axes.ravel().tolist())
        # fig.savefig('../doc/images/demo_bp.png', dpi=96)
        plt.show()

######## -  some measurements functions  -   ###
def display_v0_v1(mesh_obj, protocol_obj, perm0, perm_true):
    """
    Calcule et affiche les matrices v0 et v1 (tensions aux électrodes)
    pour perm0 (milieu homogène) et perm_true (milieu réel).

    Paramètres
    ----------
    mesh_obj : PyEITMesh
        Maillage PyEIT déjà construit.
    protocol_obj : PyEITProtocol
        Protocole EIT déjà créé (ex_mat, meas_mat, etc.).
    perm0 : array_like, shape (n_elem,)
        Vecteur de conductivité de référence (milieu homogène).
    perm_true : array_like, shape (n_elem,)
        Vecteur de conductivité “réel”.

    Comportement
    ------------
    - Calcule v0 = fwd.solve_eit(perm=perm0) et v1 = fwd.solve_eit(perm=perm_true).
    - Détermine n_exc = nombre d’électrodes, n_meas = len(v0) // n_exc.
    - Remet en forme v0_mat, v1_mat de shape (n_exc, n_meas).
    - Imprime v0_mat, v1_mat et leurs différences.
    - Trace v0_mat[i], v1_mat[i], et Δv_mat[i] pour i=0 (pattern 0) en deux sous‐plots.

    """
    from pyeit.eit.fem import EITForward

    # 1) Calcul des tensions v0 et v1
    fwd = EITForward(mesh_obj, protocol_obj)

    v0 = fwd.solve_eit(perm=perm0)
    mesh_obj.perm = perm_true
    v1 = fwd.solve_eit(perm=perm_true)

    n_exc = mesh_obj.el_pos.size
    total_meas = v0.shape[0]
    n_meas = total_meas // n_exc

    v0_mat = v0.reshape(n_exc, n_meas)
    v1_mat = v1.reshape(n_exc, n_meas)
    dv_mat = v1_mat - v0_mat

    np.set_printoptions(precision=4, suppress=True)
    # print(f"v0_mat shape = {v0_mat.shape}")
    # print(v0_mat, "\n")
    # print(f"v1_mat shape = {v1_mat.shape}")
    # print(v1_mat, "\n")
    # print("Δv_mat = v1_mat - v0_mat\n", dv_mat)

    # 5) Tracé : pattern 0 (i=0)
    i = 4   
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # (a) v0 vs v1 pour pattern 0
    axes[0].plot(v0_mat[i, :], '-o', label="v0 (réf)")
    axes[0].plot(v1_mat[i, :], '-s', label="v1 (réel)")
    axes[0].set_title("Tensions v0 et v1 (pattern 0)")
    axes[0].set_ylabel("Tension")
    axes[0].legend(loc="upper right", fontsize="small")
    axes[0].grid(True)

    # (b) Δv pour pattern 0
    axes[1].plot(dv_mat[i, :], '-^', label="Δv = v1-v0", color="red")
    axes[1].set_title("Δv (pattern 0)")
    axes[1].set_xlabel("Indice de mesure")
    axes[1].set_ylabel("ΔTension")
    axes[1].legend(loc="upper right", fontsize="small")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()





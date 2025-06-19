
import numpy as np
from pyeit.mesh import PyEITMesh
import pyeit.mesh as mesh
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.protocol import PyEITProtocol
from pyeit.visual.plot    import create_mesh_plot, create_plot
from pyeit.mesh.shape import *
from pyeit.eit.fem    import EITForward,calculate_ke,assemble

import sys, os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from Extract_skin_mask import *
from help_functions import *
from EIT_sim import *

from pyeit.mesh.utils import edge_list
from pyeit.mesh import PyEITMesh




def build_and_store_meshes(
    case_id,            # liste de strings e.g. ["s0011","s0012",...]
    base_dir,            # dossier racine où sont Data_set/{case_id}/ct.nii.gz
    organs,         # liste de noms de segmentation à charger
    slice_index,         # coupe à extraire
    n_el,                # nb électrodes pour pyeit.mesh.create
    h0,                  # résolution pour pyeit.mesh.create
    output_dir           # où écrire les .pkl
):
    print (len(organs))
    os.makedirs(output_dir, exist_ok=True)

    print(f"==> Traitement de {case_id}")
    soft_tissue,outside,_=Create_skin_mask_bis(case_id,[x for sous_liste in organs for x in sous_liste])
    mask3d = [outside,soft_tissue]
    for organ_parts in organs :
        organ_mask = Create_organ_mask(case_id, organ_parts)
        mask3d.append(organ_mask)
        
    mask2d = Create_mask_2D(mask3d, slice_index)


    print("mask crée !")
    # 2) contour et fonction distance
    body_poly = Extract_contour(mask2d)
    fd_body   = make_fd_body(body_poly)

    # 3) création du mesh PyEIT en coords normalisées
    mesh_obj = mesh.create(
         n_el=n_el, h0=h0, fd=fd_body, fh=area_uniform
    )

    # 4) passe en coordonnées réelles (mm)
    mesh_obj = scale_pyeit_mesh_to_mm(mesh_obj, case_id, slice_index, mask2d)

    # 5) étiquettes éléments & conductivités
        
    perm0 = set_condu_eit(mask2d,mesh_obj,case_id,slice_index,len(organs))
    mesh_obj.perm = perm0


    perm=set_condu_nodes(mask2d, mesh_obj, case_id, slice_index,len(organs))

     

    # 6) conversion mesh.node mm→m pour la simulation SI
    mesh_obj.node[:,0] /= 1000.0
    mesh_obj.node[:,1] /= 1000.0


    print("mesh géneré !! ")
    # 7) sérialisation
    out_path = os.path.join(output_dir, f"{case_id}_mesh.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "mask":      mask2d,
            "node":    mesh_obj.node,
            "element": mesh_obj.element,
            "perm":    mesh_obj.perm,
            "perm_OOEIT":    perm,
            "el_pos":  mesh_obj.el_pos,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  → Mesh enregistré dans {out_path}\n")
    
    

if __name__=="__main__":
    sujet = "s0011"               # exemple
    zslice_values=[318,361,92,324,359,338,180,330,346,325]
    CASES_LIBRARY =["s0028","s0667","s0691","s0708","s0720","s0723","s0070","s0648","s0613","s0561"]
    ORGANS = [
        ["lung_lower_lobe_left.nii.gz",
        "lung_lower_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_upper_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",],
        ["heart.nii.gz"],
        ["aorta.nii.gz"],
        ["trachea.nii.gz"],
        ["esophagus.nii.gz"],
        ["rib_left_1.nii.gz",
        "rib_left_2.nii.gz",
        "rib_left_3.nii.gz",
        "rib_left_4.nii.gz",
        "rib_left_5.nii.gz",
        "rib_left_6.nii.gz",
        "rib_left_7.nii.gz",
        "rib_left_8.nii.gz",
        "rib_left_9.nii.gz",
        "rib_left_10.nii.gz",
        "rib_left_11.nii.gz",
        "rib_left_12.nii.gz",
        "rib_right_1.nii.gz",
        "rib_right_2.nii.gz",
        "rib_right_3.nii.gz",
        "rib_right_4.nii.gz",
        "rib_right_5.nii.gz",
        "rib_right_6.nii.gz",
        "rib_right_7.nii.gz",
        "rib_right_8.nii.gz",
        "rib_right_9.nii.gz",
        "rib_right_10.nii.gz",
        "rib_right_11.nii.gz",
        "rib_right_12.nii.gz"],
        ["vertebrae_C1.nii.gz",
        "vertebrae_C2.nii.gz",
        "vertebrae_C3.nii.gz",
        "vertebrae_C4.nii.gz",
        "vertebrae_C5.nii.gz",
        "vertebrae_C6.nii.gz",
        "vertebrae_C7.nii.gz",
        "vertebrae_L1.nii.gz",
        "vertebrae_L2.nii.gz",
        "vertebrae_L3.nii.gz",
        "vertebrae_L4.nii.gz",
        "vertebrae_L5.nii.gz",
        "vertebrae_S1.nii.gz",
        "vertebrae_T1.nii.gz",
        "vertebrae_T2.nii.gz",
        "vertebrae_T3.nii.gz",
        "vertebrae_T4.nii.gz",
        "vertebrae_T5.nii.gz",
        "vertebrae_T6.nii.gz",
        "vertebrae_T7.nii.gz",
        "vertebrae_T8.nii.gz",
        "vertebrae_T9.nii.gz",
        "vertebrae_T10.nii.gz",
        "vertebrae_T11.nii.gz",
        "vertebrae_T12.nii.gz"],
        ["scapula_left.nii.gz", "scapula_right.nii.gz"]]
    build_and_store_meshes(                         ##                       
        case_id=sujet,
        base_dir="Data_set",
        organs=ORGANS,
        slice_index=344,
        n_el=16,
        h0=0.04,
        output_dir="meshes_cache_sujets"
    )
    for case_id, slice_index in zip(CASES_LIBRARY,zslice_values):

        build_and_store_meshes(                         ##
            case_id=case_id,
            base_dir="Data_set",
            organs=ORGANS,
            slice_index=slice_index,
            n_el=16,
            h0=0.04,
            output_dir="meshes_cache"
        )




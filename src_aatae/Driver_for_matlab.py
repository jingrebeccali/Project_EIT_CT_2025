import os, pickle
import numpy as np
from scipy.io import savemat
from pyeit.mesh.utils import edge_list

def export_meshes_to_mat(cache_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".pkl"):
            continue
        name = fname.replace(".pkl","")
        with open(os.path.join(cache_dir, fname), "rb") as f:
            data = pickle.load(f)
        node    = data["node"]      # (Nnodes,2)
        element = data["element"]   # (Nelements,3)
        el_pos  = data["el_pos"]    # (n_el,) indices   
        sigma_OOEIT = data["perm_OOEIT"]  
        boundary=data["boundary_edges"]          # (Nelements,) conductivités
        present_organs=data["present_organs"]
        slice_number = data["slice_number"]
        z_min=data["z_min"]
        z_max=data["z_max"]
        
        # print(boundary)
        # Reconstruire E comme précédemment
        # boundary = edge_list(element)

        # E = []
        # for center in el_pos:
        #     segs = [e for e in boundary if center in e]
        #     E.append(np.vstack(segs))

        # --- Ici, on construit matdict et on y injecte sigma ---
        matdict = {
            "g":       node,
            "H":       element,
            "sigma":   data["perm_OOEIT"],  # <— on ajoute σ par élément
            "edges":   boundary,
            "present_organs": present_organs,
            "z":slice_number,
            "z_min":z_min,
            "z_max":z_max,
            "mask" : data["mask"],

        }
        # Ajout dynamique des E1, E2, ..., En
        # for ℓ, segs in enumerate(E, start=1):
        #     matdict[f"E{ℓ}"] = segs

        # Sauvegarde
        out_path = os.path.join(out_dir, f"{name}.mat")
        savemat(out_path, matdict, oned_as="column")
        print(f"→ Exporté {out_path}")

if __name__=="__main__":
    export_meshes_to_mat(cache_dir="scratch/aboulette/meshes_cache_sujets/s0011", out_dir="meshes_mat_sujetss/s0011")
    

    
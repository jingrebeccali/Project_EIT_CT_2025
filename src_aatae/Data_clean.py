
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import numpy as np
import pandas as pd
import nibabel as nib
from skimage import measure
from nibabel.affines import apply_affine
import trimesh
from visualise_ALPHA import *
from visualise_MC import *




def touches_border(row):
        # Input: row: tolerance to each direction, in mm
        # Output: 'True' if the organe touches the bordrers of the CT,fasle if not  
        return (
            np.isclose(row["min_x"], 0, atol=tol) or np.isclose(row["max_x"], row["bound_x"], atol=tol) or
            np.isclose(row["min_y"], 0, atol=tol) or np.isclose(row["max_y"], row["bound_y"], atol=tol) or
            np.isclose(row["min_z"], 0, atol=tol) or np.isclose(row["max_z"], row["bound_z"], atol=tol)
        )




#list of organs
organs = [
    "rib_left_1.nii.gz","rib_left_2.nii.gz","rib_left_3.nii.gz"]

data_dir = r'Data_set'
meta_csv = r'meta.csv'
meta_df = pd.read_csv(meta_csv, sep=";")
meta_df.set_index("image_id",inplace=True)

all_subjects = sorted(os.listdir(data_dir))
tol=84

for organ_name in organs:
    print(f"Traitement:{organ_name}")
    results=[]
    for case_id in all_subjects:
        seg_path=os.path.join(data_dir, case_id, "segmentations", organ_name)
        if not os.path.exists(seg_path):
            continue
        try:
            mesh,verts_mm,faces,volume_cm3,surface_cm2=make_mesh_MC(case_id, organ_name,0.5)
            min_xyz=verts_mm.min(axis=0)
            max_xyz=verts_mm.max(axis=0)
            img=nib.load(seg_path)
            data=img.get_fdata()
            shape=data.shape
            zooms=img.header.get_zooms()
            bounds=np.array(shape) * np.array(zooms)
            # Age & gender
            if case_id in meta_df.index:
                age=meta_df.loc[case_id, "age"]
                gender=meta_df.loc[case_id, "gender"]
            else:
                age=None
                gender=None
            results.append({
                "subject": case_id,
                "volume_cm3": volume_cm3,
                "surface_cm2": surface_cm2,
                "min_x": min_xyz[0], "max_x": max_xyz[0],
                "min_y": min_xyz[1], "max_y": max_xyz[1],
                "min_z": min_xyz[2], "max_z": max_xyz[2],
                "bound_x": bounds[0], "bound_y": bounds[1], "bound_z": bounds[2],
                "age": age, "gender": gender
            })
        except Exception as e:
            print(f"Erreur pour {case_id}: {e}")


df = pd.DataFrame(results)
df["touches_border"] = df.apply(touches_border, axis=1)
filtered_df = df[~df["touches_border"]].copy()
print(f"Sujets valides après filtre 'bord': {len(filtered_df)}/{len(results)}")
filtered_df.to_csv(f"subjects_{organ_name.replace('.nii.gz','')}_filtered.csv", index=False)
print("Liste sauvegardée pour", organ_name)

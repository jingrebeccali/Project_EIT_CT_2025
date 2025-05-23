import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import nibabel as nib
from skimage import measure
import trimesh
import matplotlib.pyplot as plt
import seaborn as sns
from nibabel.affines import apply_affine

from  visualise_MC import *


#To adapt
data_dir=r'Data_set'
meta_csv=r'meta.csv'
organ_parts=["lung_lower_lobe_left.nii.gz","lung_upper_lobe_left.nii.gz","lung_middle_lobe_right.nii.gz","lung_lower_lobe_right.nii.gz","lung_upper_lobe_right.nii.gz"] #To adapt

meta=pd.read_csv(meta_csv, sep=";")
meta.set_index("image_id",inplace=True)

results=[]
for subject in sorted(os.listdir(data_dir)):
    
    volume_total=0.0
    surface_total=0.0

    try:
        for part in organ_parts:
            
            _,_,_,volume_cm3,surface_cm2=make_mesh_MC(subject,part,level=0.5)
            volume_total+=volume_cm3/1000
            surface_total+=surface_cm2/100


        if subject in meta.index:
            age=meta.loc[subject,"age"]
            gender=meta.loc[subject,"gender"]
            results.append({"subject": subject,"age": age,"gender": gender,"volume_cm3": volume_total,"surface_cm2": surface_total})

    except Exception as e:  #throwing exception in case of unvalid images, there are quite some, or does Level needs to be well adjusted?
        print(f"Error processing{subject}:{e}")

df = pd.DataFrame(results)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
palette={"m": "blue", "f": "red"}
sns.scatterplot(data=df, x="age", y="volume_cm3", hue="gender", palette=palette)
stats = df.groupby("gender")["volume_cm3"].agg(["mean", "std"]).round(1)

y_max=df["volume_cm3"].max()
x_min=df["age"].min()

text_x=x_min + 1
text_y=y_max - 100

for i, gender in enumerate(["m", "f"]):
    if gender in stats.index:
        mean_val=stats.loc[gender, "mean"]
        std_val=stats.loc[gender, "std"]
        gender_name="Male" if gender == "m" else "Female"
        color=palette[gender]
        plt.text(text_x,text_y-i*150,f"{gender_name}:mean={mean_val}cm³\net ={std_val}cm³",color=color,fontsize=10,ha="left",va="top")

plt.title("Lung Volume vs Age ")
plt.xlabel("Age")
plt.ylabel("Volume (cm³)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="age", y="surface_cm2", hue="gender", palette={"m": "blue", "f": "red"})
stats = df.groupby("gender")["surface_cm2"].agg(["mean", "std"]).round(1)
y_max_surf=df["surface_cm2"].max()
x_min_surf=df["age"].min()
text_x=x_min_surf+1
text_y=y_max_surf-100
for i, gender in enumerate(["m", "f"]):
    if gender in stats.index:
        mean_val=stats.loc[gender, "mean"]
        std_val=stats.loc[gender, "std"]
        gender_name="Male" if gender == "m" else "Female"
        color=palette[gender]
        plt.text(text_x,text_y-i*150,f"{gender_name}:mean={mean_val}cm²\net ={std_val}cm²",color=color,fontsize=10,ha="left",va="top")
plt.title("Lung Surface Area vs Age ")
plt.xlabel("Age")
plt.ylabel("Surface Area (cm²)")
plt.tight_layout()
plt.show()

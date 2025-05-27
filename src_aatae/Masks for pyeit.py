import nibabel as nib
import numpy as np
import napari
from scipy.ndimage import binary_closing, binary_fill_holes,label

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

#To adapt
base_dir=r'Data_set'
case_id="s0011"    
organ_name="rib_left_1.nii.gz"
organ_parts= [
    "lung_lower_lobe_left.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_middle_lobe_right.nii.gz",
    "lung_upper_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz",
]
    
   
 
   
ct_path=os.path.join(base_dir,case_id,"ct.nii.gz")
subject_path = os.path.join(base_dir, case_id)

shape = None
mask_total = None

for part in organ_parts:
    seg_path=os.path.join(subject_path, "segmentations",part)
    if not os.path.exists(seg_path):
        print(f"{seg_path} n'existe pas, on saute.")
        continue
    mask = nib.load(seg_path).get_fdata().astype(np.uint8)
    if mask_total is None:
        mask_total = np.zeros_like(mask)
    mask_total = np.logical_or(mask_total, mask)



ct=nib.load(ct_path).get_fdata()

lungs_mask=mask_total.astype(np.uint8)


# we condider that the outside has small density, so we can use a threshold
body_mask=(ct>-300)  # could be adjusted based on the subject


labeled,num = label(body_mask)
#We nly keep the largest connected component, which should be the body in a CT scan
sizes=np.bincount(labeled.ravel())
sizes[0]=0
largest=sizes.argmax()
body_mask=(labeled==largest)


body_mask=body_mask.astype(np.uint8)

body_mask_filled = binary_fill_holes(body_mask).astype(np.uint8)  #some zones inside the body may not be filled,in case we only work one or few organs


rest_of_body_mask_filled=(body_mask_filled & (~lungs_mask)).astype(np.uint8)

rest_of_body_mask_filled=binary_fill_holes(rest_of_body_mask_filled).astype(np.uint8)       #Same

oustide_mask_filled=(~np.logical_or(rest_of_body_mask_filled,lungs_mask)).astype(np.uint8)

#  lungs_mask: mask of the lungs
#  rest_of_body_mask_filled: mask of the rest of the body, filled
#  oustide_mask_filled : mask of the outside of the body
viewer=napari.Viewer()

viewer.add_image(ct, name="CT of "+case_id, colormap="gray", contrast_limits=[-200, 500])
viewer.add_labels(lungs_mask,name="lungs",opacity=0.5)        
viewer.add_labels(rest_of_body_mask_filled,name="rest_of_body",opacity=0.5) 
viewer.add_labels(oustide_mask_filled ,name="Outside of body_filled",opacity=0.5) 

napari.run()

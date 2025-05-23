import nibabel as nib
import numpy as np
import napari
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

#To adapt
base_dir=r'Data_set'
case_id="s0011"    
organ_name="rib_left_1.nii.gz"




ct_path=os.path.join(base_dir,case_id,"ct.nii.gz")
subject_path = os.path.join(base_dir, case_id)
seg_path=os.path.join(subject_path, "segmentations", organ_name)




ct=nib.load(ct_path).get_fdata()
mask=nib.load(seg_path).get_fdata().astype(np.uint8)

viewer=napari.Viewer()
viewer.add_image(ct, name="CT of "+case_id, colormap="gray", contrast_limits=[-200, 500])
viewer.add_labels(mask,name=organ_name,opacity=0.5)        #Optionnel
napari.run()


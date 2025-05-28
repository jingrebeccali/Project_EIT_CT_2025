import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib.pyplot as plt
import napari
from scipy.ndimage import binary_closing, binary_fill_holes,label
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


##Help function 
def keep_largest_component(mask):
    """
    Conserve uniquement la plus grande composante connexe d'un masque binaire 2D.
    """
    labeled, num = label(mask)
    if num == 0:
        return mask  # rien à garder
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # Ignore le fond
    largest = sizes.argmax()
    mask_clean = (labeled == largest)
    return mask_clean.astype(np.uint8)

def extract_skin_mask(case_id, organ_parts, z):
    base_dir=r'Data_set'   
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
    body_mask=(ct>-300)# we condider that the outside has small density, so we can use a threshold
                            # could be adjusted based on the subject

    body_mask=keep_largest_component(body_mask).astype(np.uint8)

    body_mask_filled = binary_fill_holes(body_mask).astype(np.uint8)  #some zones inside the body may not be filled,in case we only work one or few organs
    rest_of_body_mask_filled=(body_mask_filled & (~lungs_mask)).astype(np.uint8)
    rest_of_body_mask_filled=binary_fill_holes(rest_of_body_mask_filled).astype(np.uint8)       #Same
    outside_mask_filled=(~np.logical_or(rest_of_body_mask_filled,lungs_mask)).astype(np.uint8)


    outside_mask_clean = keep_largest_component(outside_mask_filled)
    rest_of_body_clean = (~np.logical_or(outside_mask_clean,lungs_mask)).astype(np.uint8)


    mask_eit = np.zeros_like(lungs_mask[:,:,z])

    mask_eit[rest_of_body_clean[:,:,z] > 0] = 1
    mask_eit[lungs_mask[:,:,z] > 0] = 2 

    return mask_eit, lungs_mask,rest_of_body_clean,outside_mask_clean,ct

  

def show_masks_eit(mask_eit):
    plt.figure(figsize=(6,6))
    plt.imshow(mask_eit, interpolation='nearest')
    plt.title('Tranche EIT (mask_eit)')
    plt.axis('off')
    cbar = plt.colorbar(ticks=[0,1,2])
    cbar.ax.set_yticklabels(['extérieur','corps','poumons'])
    plt.show()

def show_masks_3D(organ_mask,body_mask,outside_mask,ct, case_id):
    viewer = napari.Viewer()
    viewer.add_image(ct, name="CT of "+case_id, colormap="gray", contrast_limits=[-200, 500])

    viewer.add_labels(organ_mask, name='lungs mask', opacity=0.5)
    viewer.add_labels(body_mask, name='body mask Mask', opacity=0.5)

    viewer.add_labels(outside_mask, name='outside Mask', opacity=0.5)


    napari.run()
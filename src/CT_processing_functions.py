import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import nibabel as nib
import numpy as np
from skimage import measure

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.ndimage import binary_closing, binary_fill_holes,binary_opening,label
from shapely.geometry import Polygon, Point
from scipy.ndimage import binary_closing, binary_fill_holes,binary_opening



##  Helper ##
def keep_largest_component(mask):
    """
    Keep the largest connected component in a binary mask.
    Parameters
    ----------
    mask : (H, W) array
        Binary mask where 1 indicates the object and 0 indicates the background.
    Returns
    -------
    mask_clean : (H, W) array
        Binary mask with only the largest connected component kept.
    """
    labeled, num = label(mask)
    if num == 0:
        return mask  # rien à garder
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # Ignore le fond
    largest = sizes.argmax()
    mask_clean = (labeled == largest)
    return mask_clean.astype(np.uint8)
## ##

def Create_skin_mask(case_id):

    """
    Create a skin mask from the CT data of a given case.
    Parameters
    ----------
    case_id : str
        The ID of the case for which to create the skin mask.
    Returns
    -------
    skin_mask : (H, W, D) array
        The skin mask extracted from the CT data.
    outside_mask : (H, W, D) array
        The mask of the outside region, excluding the skin.
        
    """         
    base_dir=r'Data_set'   
    ct_path=os.path.join(base_dir,case_id,"ct.nii.gz")
    subject_path = os.path.join(base_dir, case_id)
    organ_parts = [
    "adrenal_gland_left.nii.gz",
    "adrenal_gland_right.nii.gz",
    "aorta.nii.gz",
    "atrial_appendage_left.nii.gz",
    "autochthon_left.nii.gz",
    "autochthon_right.nii.gz",
    "brachiocephalic_trunk.nii.gz",
    "brachiocephalic_vein_left.nii.gz",
    "brachiocephalic_vein_right.nii.gz",
    "brain.nii.gz",
    "clavicula_left.nii.gz",
    "clavicula_right.nii.gz",
    "colon.nii.gz",
    "common_carotid_artery_left.nii.gz",
    "common_carotid_artery_right.nii.gz",
    "costal_cartilages.nii.gz",
    "duodenum.nii.gz",
    "esophagus.nii.gz",
    "femur_left.nii.gz",
    "femur_right.nii.gz",
    "gallbladder.nii.gz",
    "gluteus_maximus_left.nii.gz",
    "gluteus_maximus_right.nii.gz",
    "gluteus_medius_left.nii.gz",
    "gluteus_medius_right.nii.gz",
    "gluteus_minimus_left.nii.gz",
    "gluteus_minimus_right.nii.gz",
    "heart.nii.gz",
    "hip_left.nii.gz",
    "hip_right.nii.gz",
    "humerus_left.nii.gz",
    "humerus_right.nii.gz",
    "iliac_artery_left.nii.gz",
    "iliac_artery_right.nii.gz",
    "iliac_vena_left.nii.gz",
    "iliac_vena_right.nii.gz",
    "iliopsoas_left.nii.gz",
    "iliopsoas_right.nii.gz",
    "inferior_vena_cava.nii.gz",
    "kidney_cyst_left.nii.gz",
    "kidney_cyst_right.nii.gz",
    "kidney_left.nii.gz",
    "kidney_right.nii.gz",
    "liver.nii.gz",
    "lung_lower_lobe_left.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_middle_lobe_right.nii.gz",
    "lung_upper_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz",
    "pancreas.nii.gz",
    "portal_vein_and_splenic_vein.nii.gz",
    "prostate.nii.gz",
    "pulmonary_vein.nii.gz",
    "rib_left_1.nii.gz",
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
    "rib_right_12.nii.gz",
    "sacrum.nii.gz",
    "scapula_left.nii.gz",
    "scapula_right.nii.gz",
    "skull.nii.gz",
    "small_bowel.nii.gz",
    "spinal_cord.nii.gz",
    "spleen.nii.gz",
    "sternum.nii.gz",
    "stomach.nii.gz",
    "subclavian_artery_left.nii.gz",
    "subclavian_artery_right.nii.gz",
    "superior_vena_cava.nii.gz",
    "thyroid_gland.nii.gz",
    "trachea.nii.gz",
    "urinary_bladder.nii.gz",
    "vertebrae_C1.nii.gz",
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
    "vertebrae_T12.nii.gz"
    ]

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


    mask_total=mask_total.astype(np.uint8)

    ct=nib.load(ct_path).get_fdata()
    body_mask=(ct>-200)# we condider that the outside has small density, so we can use a threshold
                            # could be adjusted based on the subject                        
    body_mask=keep_largest_component(body_mask).astype(np.uint8)

    body_mask_filled = binary_fill_holes(body_mask).astype(np.uint8)  #some zones inside the body may not be filled,in case we only work one or few organs
    rest_of_body_mask_filled=(body_mask_filled & (~mask_total)).astype(np.uint8)
    rest_of_body_mask_filled=binary_fill_holes(rest_of_body_mask_filled).astype(np.uint8)       #Same
    outside_mask_filled=(~np.logical_or(rest_of_body_mask_filled,mask_total)).astype(np.uint8)


    outside_mask_clean = keep_largest_component(outside_mask_filled)
    rest_of_body_clean = (~np.logical_or(outside_mask_clean,mask_total)).astype(np.uint8)
    body=np.logical_or(rest_of_body_clean,mask_total).astype(np.uint8)

    return rest_of_body_clean,outside_mask_clean,ct



def Extrat_skin(case_id):
    """
    Extract the skin mask from the CT data of a given case and save it as a NIfTI file.
    Parameters
    ----------
    case_id : str
        The ID of the case for which to extract the skin mask.  
    Returns
    -------
    skin_mask : (H, W, D) array
        The skin mask extracted from the CT data.
    outside_mask_clean : (H, W, D) array
        The cleaned mask of the outside region, excluding the skin.
    """
    base_dir=r'Data_set'   
    ct_path=os.path.join(base_dir,case_id,"ct.nii.gz")
    subject_path = os.path.join(base_dir, case_id,"segmentations")
    output_path = os.path.join(subject_path, "skin.nii.gz")
    ct=nib.load(ct_path)
    ct_data = ct.get_fdata()
    header = ct.header.copy()
    affine = nib.load(ct_path).affine


    skin_mask,outside_mask_clean,_= Create_skin_mask(case_id)
    ct_skin = nib.Nifti1Image(skin_mask, affine, header=header)  # Create a new Nifti1Image with the skin mask
    nib.save(ct_skin, output_path)
    return skin_mask,outside_mask_clean



def Create_organ_mask(case_id, organ_parts):
    """
    Create a mask for a specific organ from the segmentation files of a given case.
    Parameters
    ----------
    case_id : str
        The ID of the case for which to create the organ mask.
    organ_parts : list of str
        List of organ part filenames to consider for the mask.
    Returns
    -------
    mask_total : (H, W, D) array
        The combined mask of the specified organ parts.

    """
    base_dir    = "Data_set"
    subject_dir = os.path.join(base_dir, case_id, "segmentations")

    mask_total     = None
    any_part_found = False

    for part in organ_parts:
        seg_path = os.path.join(subject_dir, part)
        if not os.path.exists(seg_path):
            # fichier absent, on ignore
            continue

        seg_vol = nib.load(seg_path).get_fdata()
        # recadrage Z
        seg_crop = seg_vol

        if np.any(seg_crop > 0):
            any_part_found = True
            part_mask = (seg_crop > 0).astype(np.uint8)
            if mask_total is None:
                # initialisation à la forme de la première partie valide
                mask_total = np.zeros_like(part_mask, dtype=np.uint8)
            # fusion OR
            mask_total = np.logical_or(mask_total, part_mask)

    if not any_part_found:
        # aucune partie n'a de voxel : on déclenche l'erreur
        raise ValueError(f"Empty segmentation for all parts of organ ({organ_parts}) in case {case_id}")

    return mask_total.astype(np.uint8)



def Create_skin_mask_bis(case_id,organ_parts):
    """
    Create a skin mask and outside mask from the CT data of a given case.
    Parameters
    ----------
    case_id : str

        The ID of the case for which to create the skin mask.
    organ_parts : list of str
        List of organ part filenames to consider for the mask.
    Returns
    -------
    rest_of_body_clean : (H, W, D) array
        The cleaned mask of the rest of the body, excluding the skin.
    outside_mask_clean : (H, W, D) array
        The cleaned mask of the outside region, excluding the skin.
    mask_total : (H, W, D) array    
        The combined mask of the specified organ parts.
    -------
    What changes from Create_skin_mask: here we consider each organ not present in organ_parts as skin(soft-tissue).
    """


    base_dir=r'Data_set'   
    ct_path=os.path.join(base_dir,case_id,"ct.nii.gz")
    subject_path = os.path.join(base_dir, case_id)
    

    mask_total = Create_organ_mask(case_id,organ_parts)



    mask_total=mask_total.astype(np.uint8)

    ct=nib.load(ct_path)
    ct_data=ct.get_fdata()

    affine=nib.load(ct_path).affine
    
    body_mask=(ct_data>-300)# we condider that the outside has small density, so we can use a threshold
                            # could be adjusted based on the subject                        
    body_mask=keep_largest_component(body_mask).astype(np.uint8)

    body_mask_filled = binary_fill_holes(body_mask).astype(np.uint8)  #some zones inside the body may not be filled,in case we only work one or few organs
    rest_of_body_mask_filled=(body_mask_filled & (~mask_total)).astype(np.uint8)
    radius =1
    size = 2 * radius + 1  # 9

    # créer les grilles de coordonnées centrées en 0
    # np.ogrid est plus mémoire-efficace que meshgrid pour ce cas
    Z, Y, X = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]

    # masque sphérique : inclusion si distance <= radius
    structure = (X**2 + Y**2 + Z**2) <= radius**2

    rest_of_body_mask_filled = binary_closing(rest_of_body_mask_filled, structure=structure).astype(np.uint8)  #Ferme les trous
    rest_of_body_mask_filled=binary_fill_holes(rest_of_body_mask_filled).astype(np.uint8)       #Same
    outside_mask_filled=(~np.logical_or(rest_of_body_mask_filled,mask_total)).astype(np.uint8)



    outside_mask_clean = keep_largest_component(outside_mask_filled)
    outside_mask_clean =binary_opening(outside_mask_clean, structure=structure).astype(np.uint8)  #Ferme les trous
    rest_of_body_clean = (~np.logical_or(outside_mask_clean,mask_total)).astype(np.uint8)

    
    
    # body=np.logical_or(rest_of_body_clean,mask_total).astype(np.uint8)
    return rest_of_body_clean,outside_mask_clean,mask_total


def Get_all_masks(case_id, organs_presnt, ORGANS):
    """
    Create a list of masks for the specified organs present in the case.
    Parameters
    ----------
    case_id : str
        The ID of the case for which to create the masks.
    organs_presnt : list of str
        List of organs present in the case.
    ORGANS : dict
        Dictionary mapping organ names to their corresponding segmentation files.
    Returns
    -------
    mask3d : list of (H, W, D) arrays
        List of masks for the specified organs, including skin and outside masks.

    """
    selected_files = [
    fname
    for struct in organs_presnt
    for fname in ORGANS.get(struct, [])
    ]
    soft_tissue, outside, _ = Create_skin_mask_bis(case_id,selected_files)
    mask3d=[outside,soft_tissue]
    for organ in organs_presnt :
        organ_mask = Create_organ_mask(case_id, ORGANS[organ])
        mask3d.append(organ_mask)
    return mask3d

def Create_mask_2D(masks,z):
    """
    Create a 2D mask from a list of 3D masks at a specific z-slice.
    Parameters
    ----------
    masks : list of (H, W, D) arrays
        List of 3D masks for different organs.
    z : int
        The z-slice index to extract from the 3D masks.
    Returns
    -------
    mask_eit : (H, W) array
        The 2D mask for the specified z-slice, where each organ is represented by
        a unique integer value corresponding to its index in the masks list.    
    """
    mask_eit = np.zeros_like(masks[0][:,:,z])
    for i,mask in zip(range(0,len(masks)),masks):
        mask_eit[mask[:,:,z]>0]=i
    # mask_closed = binary_closing(mask_eit, structure=disk(3))
    # mask_filled = binary_fill_holes(mask_closed)
    return mask_eit.astype(np.uint8)


#### Crop en Z ####

def compute_z_bounds(ct_path, seg_dir, top, bottom, margin):
    """
    Compute the Z bounds for cropping the CT data based on segmentation files.
    Parameters
    ----------
    ct_path : str
        Path to the CT NIfTI file.
    seg_dir : str   
        Directory containing segmentation files.
    top : str   
        Name of the segmentation file for the top boundary.
    bottom : str
        Name of the segmentation file for the bottom boundary.
    margin : int
        Margin to add to the computed Z bounds.
    Returns
    -------
    crop_ct : (H, W, D) array
        The cropped CT data.
    z_min : int
        The minimum Z index before cropping.
        z_max : int 
        The maximum Z index before cropping.
        ct : (H, W, D) array
        The original CT data.
        
    """
    # 
    ct_img   = nib.load(ct_path)
    ct       = ct_img.get_fdata()       
    nz       = ct.shape[2]

    
    bottom_mask_z = np.zeros(nz, dtype=bool)
    for fname in os.listdir(seg_dir):
        if fname.startswith(bottom) and fname.endswith(".nii.gz"):
            seg = nib.load(os.path.join(seg_dir, fname)).get_fdata()
            # True pour chaque coupe Z contenant bottom
            bottom_mask_z |= np.any(seg > 0, axis=(0,1))
    if not bottom_mask_z.any():
        raise RuntimeError(f"Aucun repère bottom `{bottom}` trouvé.")
    z_min = bottom_mask_z.argmax()       # premier True

    
    top_mask_z = np.zeros(nz, dtype=bool)
    for fname in os.listdir(seg_dir):
        if fname.startswith(top) and fname.endswith(".nii.gz"):
            seg = nib.load(os.path.join(seg_dir, fname)).get_fdata()
            top_mask_z |= np.any(seg > 0, axis=(0,1))
    if not top_mask_z.any():
        raise RuntimeError(f"Aucun repère top `{top}` trouvé.")
    z_max = nz - 1 - top_mask_z[::-1].argmax()  # dernier True

    
    z_min = max(z_min - margin, 0)
    z_max = min(z_max + margin, nz - 1)
    if z_min >= z_max:
        raise RuntimeError(f"z_min ({z_min}) >= z_max ({z_max})")

    
    crop_ct = ct[:, :, z_min : z_max + 1]
    return crop_ct, z_min, z_max,ct

def organs_present_in_crop(seg_dir, z_min, z_max, structures):
    """
    Check which organs are present in the cropped segmentation files.
    Parameters
    ----------
    seg_dir : str
        Directory containing segmentation files.
    z_min : int
        Minimum Z index of the crop.
    z_max : int
        Maximum Z index of the crop.
    structures : dict
        Dictionary mapping organ names to their corresponding segmentation file names.
    Returns
    -------
    present : list of str
        List of organ names that are present in the cropped segmentation files.

    
    """
    present = []
    
    for organ, parts in structures.items():
        found = False
        for part in parts:
            seg_path = os.path.join(seg_dir, part)
            if not os.path.exists(seg_path):
                continue
            seg = nib.load(seg_path).get_fdata()
            
            crop_seg = seg[:, :, z_min:z_max+1]
            if np.any(crop_seg > 0):
                found = True
                break
        if found:
            present.append(organ)
    return present

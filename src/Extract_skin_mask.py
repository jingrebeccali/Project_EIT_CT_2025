import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from skimage import measure

import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes,label
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

###import napari      # Uncomment if you have napari installed, but it is not necessary for the main functionality

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

    """
    INPUT:
    - case_id: ID du cas,
    - organ_parts: liste des parties de l'organe à considérer,
    - z: indice de la tranche à extraire (par exemple, 340 pour la tranche 340).
    OUTPUT:
    - lungs_mask: lungs mask,     (en 3D)
    - rest_of_body_clean: masque du reste du corps,(en 3D)
    - outside_mask_clean: masque de l'extérieur,(en 3D)
    - body: masque du corps,(en 3D)
    - ct: données de la CT, (en 3D)
    - mask_eit: la coupe en z de la CT avec  masques, pour EIT, (en 2D)
    """
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
    body=np.logical_or(rest_of_body_clean,lungs_mask).astype(np.uint8)

    mask_eit = np.zeros_like(lungs_mask[:,:,z])
    mask_eit[rest_of_body_clean[:,:,z] > 0] = 1
    mask_eit[lungs_mask[:,:,z] > 0] = 2 

    return mask_eit, lungs_mask,rest_of_body_clean,outside_mask_clean,body,ct

  

def show_masks_eit(mask_eit):
    """""
    INPUT:
    - mask_eit: la coupe en z de la CT avec  masques, pour EIT,en 2D
    OUTPUT:
    - Affiche la coupe en z de la CT avec les masques pour EIT.
    """""
    plt.figure(figsize=(6,6))
    plt.imshow(mask_eit, interpolation='nearest')
    plt.title('Tranche EIT (mask_eit)')
    plt.axis('off')
    cbar = plt.colorbar(ticks=[0,1,2])
    cbar.ax.set_yticklabels(['extérieur','corps','poumons'])
    plt.show()


"this function only works with napari installed, which you probably struggle to do (see README.md file), so I commented it out"
# def show_masks_3D(organ_mask,body_mask,outside_mask,ct, case_id):
#     viewer = napari.Viewer()
#     viewer.add_image(ct, name="CT of "+case_id, colormap="gray", contrast_limits=[-200, 500])

#     viewer.add_labels(organ_mask, name='lungs mask', opacity=0.5)
#     viewer.add_labels(body_mask, name='body mask Mask', opacity=0.5)

#     viewer.add_labels(outside_mask, name='outside Mask', opacity=0.5)


#     napari.run()



def Electrodes_position(body_mask_2D,n_el):
    """
    Extract the contour of the body mask and return the electrode positions.

    INPUT:
    - body_mask_2D: 2D array of the body mask, where the contour is extracted,
    - n_el: number of electrodes, which must be a multiple of 4.
    OUTPUT:
    - p_electrodes: (n_el, 2) array of the coordinates of electrodes 
    """
    



    #this returns a list of (N,2) arrays of [row, col] points, wich determin the contour, need to see level parameter later
    contours = measure.find_contours(body_mask_2D, level=0.5)
    cnt = max(contours, key=lambda c: c.shape[0])

    #  convert pixel coordinates [row, col] to Cartesian coordinates in [-1,1]
    h, w = body_mask_2D.shape

    p_contour = np.vstack([(cnt[:,1] / (w-1))*2 - 1,1 - (cnt[:,0] / (h-1))*2]).T

    


    # compute the total distance along the contour, then normalize
    d = np.sqrt(((p_contour[1:] - p_contour[:-1])**2).sum(1))
    s  = np.concatenate([[0], np.cumsum(d)])
    s /= s[-1]
    #interpolate to get n_el points: first on the segment [0,1](fx ad fy are funcions defined in [0,1]), then return the image of these points
    fx = interp1d(s, p_contour[:,0], kind='linear')
    fy = interp1d(s, p_contour[:,1], kind='linear')
    s_el = np.linspace(0, 1, n_el, endpoint=False)


    p_electrodes= np.vstack([fx(s_el), fy(s_el)]).T  # array of the coordinates of electrodes, of shape (n_el,2)

    return p_electrodes
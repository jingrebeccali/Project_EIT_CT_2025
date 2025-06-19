from shapely.geometry import Polygon, Point
import numpy as np
import pyeit.eit.protocol as protocol


from pyeit.mesh.shape import *
from scipy.spatial import Delaunay, cKDTree
import nibabel as nib


##  -  Help functions for EIT  -  ##


def make_fd_body(body_poly: Polygon):
    """
    Returns a function fd_body(pts) that takes an (M,2) array of (x,y) points
    and returns a length‐M array of signed distances to the polygon boundary:
       fd_body[i] < 0 if point i is iniside body_poly,
       fd_body[i] = 0 if point i is exactly on the boundary,
       fd_body[i] > 0 if point i is outside body_poly.
       To be used directly in mesh.creaate, as the "fd" variable
    """
    
    def fd_body(pts):
        """return signed distance of polygon"""
        pts_ = [Point(p) for p in pts]
        # calculate signed distance
        dist = [body_poly.exterior.distance(p) for p in pts_]
        sign = np.sign([-int(body_poly.contains(p)) + 0.5 for p in pts_])

        return sign * dist
    

    return fd_body

def compute_element_labels_affine(mask,mesh, case_id, slice_index=None):
    """
    Pour chaque centre de triangle (en mm), retourne le label du mask 2D.

    - mask        : array 2D (H x W)
    - centers_mm  : array (Ne,2) ou (Ne,3) de coordonnées réelles (mm)
    - case_id     : pour charger Data_set/{case_id}/ct.nii.gz
    - slice_index : si centers_mm est (Ne,2), donne le z (int) de coupe

    Sortie :
    - labels : array (Ne,) de labels issus de mask[y_vox, x_vox]
    """
    centers_mm=mesh.elem_centers
    # 1) Charge l'affine et son inverse
    img    = nib.load(f"Data_set/{case_id}/ct.nii.gz")
    inv_aff = np.linalg.inv(img.affine)

    # 2) Construis la matrice homogène (N,4)
    N = centers_mm.shape[0]
    ones = np.ones((N,1))

    if centers_mm.shape[1] == 3:
        # on a déjà (x,y,z)
        homog = np.hstack([centers_mm, ones])      # (N,4)
    elif centers_mm.shape[1] == 2:
        if slice_index is None:
            raise ValueError("slice_index requis lorsque centers_mm est (N,2)")
        zs = np.full((N,1), slice_index)
        homog = np.hstack([centers_mm, zs, ones])  # (N,4)
    else:
        raise ValueError("centers_mm doit être (N,2) ou (N,3)")

    # 3) Applique l'inverse d'affine pour retomber en voxels
    vox = homog.dot(inv_aff.T)[:, :3]  # (i,j,k) flottants

    # 4) Round, clip, et extrait le mask
    # note : mask[y, x], donc voxel[:,1]→row, voxel[:,0]→col
    xi = np.clip(np.round(vox[:,0]).astype(int), 0, mask.shape[1]-1)
    yi = np.clip(np.round(vox[:,1]).astype(int), 0, mask.shape[0]-1)

    return mask[yi, xi]


def set_protocol(n_el,dist_exc,step_meas):
    """
    Set the protocol for EIT measurements.
    Returns a protocol object with the specified parameters.
    
    n_el =    # Number of electrodes
    dist_exc  # Distance between excitation electrodes
    step_meas # Step size for measurements

    """
    return protocol.create(n_el, dist_exc=dist_exc, step_meas=step_meas,parser_meas="std")

def set_condu_eit(mask2d,mesh,case_id,slice_index,len_organs):
    """
    Set the conductivity values for the EIT mesh.
    Returns an array of conductivity values for each element in the mesh.
    mesh: EIT mesh object, of type pyeit.mesh.Mesh
    condu_body: Conductivity value for the body
    condu_lung: Conductivity value for the lung
    labels_elems: Array of labels for each element in the mesh
    Ne = labels_elems.shape[0](number of triangles in the mesh)
    Output:
    perm: Array of conductivity values for each element in the mesh
    perm0: Array of initial conductivity values (all ones)(Reference conductivity)
    """
    tri_labels=compute_element_labels_affine(mask2d,mesh,case_id, slice_index=slice_index)
    Ne=mesh.element.shape[0]
    
    perm= np.ones(Ne)

    # applique les valeurs
    for j in range(1,69):

        perm[tri_labels==j] = j
    return perm




def compute_node_labels_affine(mask, nodes, case_id, slice_index):
    """
    Pour chaque nœud (en mm), retourne le label du mask 2D.

    - mask       : array 2D (H x W)
    - nodes_mm   : array (Nnodes,2) de coords réelles (mm)
    - case_id    : pour charger Data_set/{case_id}/ct.nii.gz
    - slice_index: indice de coupe Z

    Sortie :
    - labels_nœuds : array (Nnodes,) de labels issus de mask[row, col]
    """
    # 1) Charge l'affine et son inverse
    img     = nib.load(f"Data_set/{case_id}/ct.nii.gz")
    inv_aff = np.linalg.inv(img.affine)
    nodes_mm = nodes[:, :2]
    # 2) Construit les coords homogènes (Nnodes,4)
    N = nodes_mm.shape[0]
    ones = np.ones((N,1))
    zs   = np.full((N,1), slice_index)
    homog = np.hstack([nodes_mm, zs, ones])  # (N,4)

    # 3) Mappe en voxels flottants puis quantifie
    vox = homog.dot(inv_aff.T)[:, :3]        # (i,j,k) floats
    xi  = np.clip(np.round(vox[:,0]).astype(int), 0, mask.shape[1]-1)
    yi  = np.clip(np.round(vox[:,1]).astype(int), 0, mask.shape[0]-1)

    # 4) indexation mask[y, x]
    return mask[yi, xi]



def set_condu_nodes(mask, mesh,case_id, slice_index,len_organs):
    """
    Retourne un vecteur de conductivité par nœud (taille mesh.n_nodes),
    à partir du masque et des valeurs pour poumon et corps.
    """
    # récupère les labels 0/1/2 pour chaque nœud
    nodes_mm     = mesh.node[:, :2]  # (Nnodes,2)
    node_labels  = compute_node_labels_affine(mask, nodes_mm, case_id, slice_index)
    Nnodes       = mesh.node.shape[0]

    # initialise à 1 (milieu de référence)
    perm_nodes = np.ones(Nnodes)

    # applique les valeurs
    for j in range(1,69):

        perm_nodes[node_labels==j] = j

    return perm_nodes











def sample_electrodes(poly, n_el):
    perim = poly.exterior.length
    dists = np.linspace(0, perim, num=n_el, endpoint=False)
    return np.array([poly.exterior.interpolate(d).coords[0] for d in dists])

def find_closest_node_indices(nodes, pts):
    tree = cKDTree(nodes)
    _, idx = tree.query(pts)
    return idx

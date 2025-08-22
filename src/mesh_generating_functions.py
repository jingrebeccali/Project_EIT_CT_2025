import nibabel as nib
import numpy as np
from skimage import measure

import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes,label
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from shapely.geometry import Polygon, Point
from skimage.morphology import disk
from scipy.ndimage import binary_closing, binary_fill_holes,binary_opening
from pyeit.mesh import PyEITMesh




from Extract_skin_mask import *


if sys.platform.startswith("win"):
    os.add_dll_directory(r"C:\vcpkg\installed\x64-windows\bin")

import pygalmesh,math

### helper for meshing ##

def filter_pts(pts, tol):
    """
    Remove consecutive points in `pts` that are within `tol` Euclidean distance.
    Ensures the resulting list still represents a closed loop when you wrap around.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        raise RuntimeError("Not enough unique points for a polygon.")
    filtered = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - filtered[-1]) > tol:
            filtered.append(p)
    # Check closure: last → first
    if np.linalg.norm(filtered[0] - filtered[-1]) < tol:
        # drop the last duplicate
        filtered.pop()
    return np.array(filtered)

#######################################


def mesh_from_mask2d_normalized(mask2d,
    closing_radius:int=5, opening_radius:int=3,
    simplify_tol:float=0.01,n_el: int=16, h:int=0.025):
    """
    Main function to create a mesh from a 2D mask.
    INputs:
    - mask2d: 2D binary mask (H, W) where 1s represent the body.
    - closing_radius: radius for binary closing to fill small holes.
    - opening_radius: radius for binary opening to remove small objects.
    - simplify_tol: tolerance for simplifying the contour.
    - n_el: number of electrodes for pyeit.mesh.create.
    - h: target maximum (in normalized space) edge length for the mesh.
6    """
    body = binary_fill_holes(mask2d==1)
    body = binary_closing(body, structure=disk(closing_radius))
    body = binary_opening(body, structure=disk(opening_radius))

    contours = measure.find_contours(body.astype(float), level=0.5)
    if not contours:
        raise RuntimeError("Aucun contour trouvé pour mask2d==1 !")
    exterior = max(contours, key=lambda c: c.shape[0])


    

    pts = [(p[1], p[0]) for p in exterior]

    H, W = mask2d.shape
    pts_norm = [(
        (x/(W-1))*2 - 1,      # x_norm ∈ [-1,1]
        (y/(H-1))*2 - 1       # y_norm ∈ [-1,1]
    ) for x, y in pts]

    pts_norm = filter_pts(pts_norm, simplify_tol)

     
    N = 50
    max_edge_size    = 1.25 / N     
    num_lloyd_steps = 3  
    n = len(pts_norm)
    constraints = np.vstack([
        [i, (i+1) % n] for i in range(n)
    ])

    # Generate a 2D mesh
    mesh = pygalmesh.generate_2d(
        points=pts_norm,
        constraints=constraints,
        B=math.sqrt(2),         # default
        max_edge_size=h,
        num_lloyd_steps=num_lloyd_steps,
    )

    # Extract normalized nodes & triangles
    nodes = mesh.points            # shape (M,2)
    elements  = mesh.cells_dict["triangle"]

    xs = exterior[:,1] / (W-1) * 2 - 1
    ys = exterior[:,0] / (H-1) * 2 - 1
    contour_pts = np.vstack([xs, ys]).T  # (Npts, 2)

    # 3) arc‐longueur cumulée
    diffs = np.diff(contour_pts, axis=0)
    dists = np.hypot(diffs[:,0], diffs[:,1])
    cumd = np.concatenate([[0], np.cumsum(dists)])
    L = cumd[-1]

    # cibles équi‐espacées
    targets = np.linspace(0, L, n_el+1)[:-1]

    el_pos = []
    for t in targets:
        idx = np.searchsorted(cumd, t)
        if idx == 0:
            p = contour_pts[0]
        else:
            t0, t1 = cumd[idx-1], cumd[idx]
            p0, p1 = contour_pts[idx-1], contour_pts[idx]
            α = (t - t0) / (t1 - t0)
            p = p0 + α*(p1 - p0)
        el_pos.append(p)
    el_pos = np.array(el_pos)  # (n_el,2) en [-1,1]^2

    # 4) snapping : choisir pour chaque p le nœud de mesh le plus proche
    #    pour garantir que l'électrode est sur un vertex existant
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    dists, idxs = tree.query(el_pos, k=1)

    return nodes, elements, idxs


################## Helpers for mesh parameterization ##################


def scale_pyeit_mesh_to_mm(mesh, case_id, slice_index, mask2d):
    """
    Handles the scaling of a PyEIT mesh from normalized coordinates
    ([-1, 1]) to real-world coordinates (mm) based on the CT affine.
    Inputs:
    - mesh.node : (Nnodes,2) in normalized coordinates [-1,1]
    - case_id    : to load ct.nii.gz and its affine
    - slice_index: z index of the slice
    - mask2d     : the original 2D mask, shape (h, w)
    Outputs:
    - mesh.node : (Nnodes,2) in real-world coordinates (mm)
    """
    # Récupère l'affine du CT
    ct = nib.load(f"Data_set/{case_id}/ct.nii.gz")
    affine = ct.affine  # 4×4
    
    h, w = mask2d.shape

    #  On renverse la normalisation [-1,1] → pixel indices [0..w-1], [0..h-1]
    #    x_norm = mesh.node[:,0] ∈ [-1,1] correspond à col
    #    y_norm = mesh.node[:,1] ∈ [-1,1] correspond à row (inversion d'axe Y)
    xn = mesh.node[:, 0]
    yn = mesh.node[:, 1]
    # colonnes pixels
    px = (xn + 1) * 0.5 * (w - 1)
    # lignes pixels (on a fait y = 1 - row/(h-1)*2 dans Extract_contour)
    py = (1 - yn) * 0.5 * (h - 1)

    #  Passe en coordonnées homogènes (i, j, z, 1) puis affine → (x_mm, y_mm, z_mm)
    ones = np.ones_like(px)
    zs   = np.full_like(px, slice_index)
    vox4 = np.vstack([px, py, zs, ones]).T         # (Nnodes,4)
    xyz  = vox4.dot(affine.T)[:, :3]               # (Nnodes,3)

    #  Remplace node par (x_mm, y_mm)
    mesh.node[:, 0] = xyz[:, 0]
    mesh.node[:, 1] = xyz[:, 1]

    return mesh



def compute_element_labels_affine(mask, mesh, case_id, slice_index=None):
    """

    For each element center (in mm), returns the label from the 2D mask.
    Inputs:
    - mesh.elem_centers : array (Ne,2) in real-world coordinates (mm
    - case_id     : to load Data_set/{case_id}/ct.nii.gz
    - slice_index : if elem_centers is (Ne,2), gives the z (int) of the slice
    Outputs:
    - labels : array (Ne,) of labels from mask[row, col]
    """
    
    # 1) Récupère les centres en mm
    centers = mesh.elem_centers  # shape (Ne,2)

    # 2) Construis les homogènes (Ne,4)
    N = centers.shape[0]
    ones = np.ones((N,1))
    if centers.shape[1] == 2:
        if slice_index is None:
            raise ValueError("slice_index requis si elem_centers est (N,2)")
        zs = np.full((N,1), slice_index)
        homog = np.hstack([centers, zs, ones])
    elif centers.shape[1] == 3:
        homog = np.hstack([centers, ones])
    else:
        raise ValueError("elem_centers doit être (N,2) ou (N,3)")

    # 3) Applique l’inverse de l’affine du CT pour retomber en voxels
    img     = nib.load(f"Data_set/{case_id}/ct.nii.gz")
    inv_aff = np.linalg.inv(img.affine)
    vox     = homog.dot(inv_aff.T)  # (Ne,4) -> (i,j,k,1)
    xi      = np.round(vox[:, 0]).astype(int)
    yi      = np.round(vox[:, 1]).astype(int)

    # 4) Comme imshow(mask) met l’origine en haut à gauche,
    #    il faut inverser l’indice de ligne :
    rows = mask.shape[0] - 1 - yi
    cols = xi

    # 5) Clip et retourne
    rows = np.clip(rows, 0, mask.shape[0]-1)
    cols = np.clip(cols, 0, mask.shape[1]-1)
    return mask[rows, cols]



def set_condu_eit(mask2d,mesh,case_id,slice_index,len_organs):
    """
    Returns a conductivity vector per element (size mesh.n_elements),
    from the mask and values for lung and body.
    Inputs:
    - mask2d: 2D binary mask (H, W) where 1s represent the body.
    - mesh: EIT mesh object, of type pyeit.mesh.Mesh
    - case_id: to load Data_set/{case_id}/ct.nii.gz
    - slice_index: index of the slice
    - len_organs: number of organs in the segmentation
    Outputs:
    - perm: array of conductivity values for each element in the mesh
    """
    tri_labels=compute_element_labels_affine(mask2d,mesh,case_id, slice_index=slice_index)
    Ne=mesh.element.shape[0]
    
    perm= np.ones(Ne)

    # applique les valeurs
    for j in range(1,150):

        perm[tri_labels==j] = j
    return perm

def compute_node_labels_affine(mask, nodes, case_id, slice_index):
    """
    For each node (in mm), returns the label from the 2D mask.
    Inputs:
    - mask       : array 2D (H x W) where 1s represent the body
    - nodes      : array (Nnodes,2) in mm coordinates
    - case_id    : to load Data_set/{case_id}/ct.nii.gz
    - slice_index: if nodes is (Nnodes,2), gives the z (int) of the slice
    Outputs:
    - labels : array (Nnodes,) of labels from mask[row, col]

    """
    # 1) Récupère l’affine inverse du CT
    img     = nib.load(f"Data_set/{case_id}/ct.nii.gz")
    inv_aff = np.linalg.inv(img.affine)

   
    nodes_mm = nodes[:, :2]
    N = nodes_mm.shape[0]
    ones = np.ones((N,1))
    if nodes.shape[1] == 2:
        zs = np.full((N,1), slice_index)
        homog = np.hstack([nodes_mm, zs, ones])
    elif nodes.shape[1] == 3:
        homog = np.hstack([nodes_mm, nodes[:,2:3], ones])
    else:
        raise ValueError("nodes must be (N,2) or (N,3)")

    # 3) Passage en voxels floating
    vox = homog.dot(inv_aff.T)  # shape (N,4)
    xi  = np.round(vox[:, 0]).astype(int)
    yi  = np.round(vox[:, 1]).astype(int)

    # 4) Inversion de l’axe Y pour matcher imshow(origin='lower')
    rows = mask.shape[0] - 1 - yi
    cols = xi

    # 5) Clip pour rester dans l’image
    rows = np.clip(rows, 0, mask.shape[0]-1)
    cols = np.clip(cols, 0, mask.shape[1]-1)

    # 6) Extraction
    return mask[rows, cols]


def set_condu_nodes(mask, mesh,case_id, slice_index,len_organs):
    """
    Same as set_condu_eit but for nodes.
    Returns a conductivity vector per node (size mesh.n_nodes),
    """
    # récupère les labels 0/1/2 pour chaque nœud
    nodes_mm     = mesh.node[:, :2]  # (Nnodes,2)
    node_labels  = compute_node_labels_affine(mask, nodes_mm, case_id, slice_index)
    Nnodes       = mesh.node.shape[0]

    # initialise à 1 (milieu de référence)
    perm_nodes = np.ones(Nnodes)

    # applique les valeurs
    for j in range(1,150):

        perm_nodes[node_labels==j] = j

    return perm_nodes




from numba import njit

@njit
def edge_list_numba(tri):
    """
    Returns a list of unique edges from the triangle mesh.
    Input:
    - tri: array (n_tri, K) of triangle indices, where K is the number of vertices per triangle.
    Output:
    - bars: array (m, 2) of unique edges, where m is the number of unique edges.
    """
    n_tri, K = tri.shape
    m = n_tri * K
    # 1) Construire toutes les arêtes triées
    bars = np.empty((m, 2), np.int64)
    for t in range(n_tri):
        for j in range(K):
            idx = t * K + j
            a = tri[t, j]
            b = tri[t, (j + 1) % K]
            if a < b:
                bars[idx, 0] = a
                bars[idx, 1] = b
            else:
                bars[idx, 0] = b
                bars[idx, 1] = a

    # 2) Identifier et éliminer les duplicats (les arêtes partagées)
    ix = np.ones(m, np.bool_)
    for i in range(m - 1):
        if not ix[i]:
            continue
        for j in range(i + 1, m):
            if bars[i, 0] == bars[j, 0] and bars[i, 1] == bars[j, 1]:
                ix[i] = False
                ix[j] = False
                break

    # 3) Collecter les arêtes uniques restantes
    #    (celles qui n'ont pas été marquées comme duplicat)
    count = 0
    for i in range(m):
        if ix[i]:
            count += 1

    out = np.empty((count, 2), np.int64)
    pos = 0
    for i in range(m):
        if ix[i]:
            out[pos, 0] = bars[i, 0]
            out[pos, 1] = bars[i, 1]
            pos += 1

    return out


#######################################################################

def build_and_store_meshes(
    case_id,            # 
    base_dir,            # dossier racine où sont Data_set/{case_id}/ct.nii.gz
    organs,         # liste de noms de segmentation à charger
    slice_index,         # coupe à extraire
    z_min,
    z_max,
    mask3d,
    present_organs,
    n_el,                # nb électrodes pour pyeit.mesh.create
    h0,                  # résolution pour pyeit.mesh.create
    output_dir           # où écrire les .pkl
):
    """
    Main function to build and store the mesh for a given case_id and slice_index.
    Inputs:
    - case_id: identifier for the case, used to load ct.nii.gz
    - base_dir: root directory where Data_set/{case_id}/ct.nii.gz is located
    - organs: list of organ names to load from segmentation
    - slice_index: index of the slice to extract
    - z_min: minimum z index for the slice
    - z_max: maximum z index for the slice
    - mask3d: 3D mask array (H, W, D) where 1s represent the body
    - present_organs: list of organs present in the segmentation
    - n_el: number of electrodes for pyeit.mesh.create
    - h0: target maximum edge length for the mesh
    - output_dir: directory where to save the mesh as .pkl
    Outputs:
    - mask2d: 2D mask for the slice
    - mesh_obj: PyEIT mesh object for the slice
    - edges: array of unique edges from the mesh
    """

    ct_path=os.path.join(base_dir,case_id,"ct.nii.gz")

    print(f"==> Traitement de {case_id} slice {slice_index}")
    #Create the 2D mask for the slice
    mask2d = Create_mask_2D(mask3d, slice_index)
    
    print(f"mask de la slice {slice_index}crée !")
    
   
    # obtain nodes, elements and el_pos after meshing
    nodes,elements,el=mesh_from_mask2d_normalized(mask2d,closing_radius= 5,
    opening_radius= 3,
    simplify_tol=0.01,n_el=16, h=h0)
    
    
    # Create the PyEIT mesh object
    mesh_obj = PyEITMesh(
        node=nodes,
        element=elements,
        el_pos=el
    )

    # Set the conductivity values for the mesh
    mesh_obj = scale_pyeit_mesh_to_mm(mesh_obj, case_id, slice_index, mask2d)  
    perm0 = set_condu_eit(mask2d,mesh_obj,case_id,slice_index,len(organs))


    mesh_obj.perm = perm0



     

    # scale nodes from mm to meters
    mesh_obj.node[:,0] /= 1000.0
    mesh_obj.node[:,1] /= 1000.0



    print(f"mesh de la slice {slice_index} géneré !")



    # boundary edges
    edges=edge_list_numba(mesh_obj.element )
    
    return mask2d, mesh_obj,edges

    # print("boundary edges saved !!")
    # out_path = os.path.join(output_dir, f"{case_id}_mesh_slice_{slice_index-z_min}.pkl")
    # with open(out_path, "wb") as f:
    #     pickle.dump({
    #         "mask":      mask2d,
    #         "node":    mesh_obj.node,
    #         "element": mesh_obj.element,
    #         "perm":    mesh_obj.perm,
    #         "perm_OOEIT":    perm,
    #         "el_pos":  mesh_obj.el_pos,
    #         "boundary_edges": edges,
    #         "present_organs": present_organs,
    #         "slice_number": slice_index,
    #         "z_min": z_min,
    #         "z_max": z_max,
    #         "aff"   : inv_aff_node
    #     }, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print(f"Mesh de la slice {slice_index} enregistré dans {out_path}\n")



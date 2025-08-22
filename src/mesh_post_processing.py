import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import nibabel as nib

from pyeit.mesh import PyEITMesh

class ThinPlateSpline2D:
    """Thin Plate Spline (TPS) interpolation in 2D.
    This class implements TPS interpolation using a kernel-based approach.
    It can be used to fit a TPS model to a set of source points and their corresponding destination points.
    The model can then be used to transform new points.
    The transformation is a combination of a non-affine part (based on the TPS kernel)
    and an affine part (linear transformation).
    The TPS kernel is defined as K(r) = r^2 * log(r), where r is the distance between points.
    The affine part is defined by a linear transformation matrix.
    The source points are used to compute the kernel matrix, and the destination points are used to solve for the weights and affine parameters.
    The transformation can be applied to new points to obtain their transformed coordinates.
    Parameters
    ----------
    src : (M, 2) array
        Source points in 2D space.
    dst : (M, 2) array  
        Destination points in 2D space.
    Attributes  
    ----------
    w : (M,) array  
        Weights for the TPS kernel.
    a : (3, 2) array
        Affine transformation parameters.
    src : (M, 2) array  
        Source points used for the TPS model.
    Methods     
    -------
    fit(src, dst)
        Fit the TPS model to the source and destination points.
    transform(pts)  
        Transform new points using the fitted TPS model.
    The transformation is a combination of the non-affine TPS kernel and the affine transformation. 
    Parameters
    ----------  
    pts : (N, 2) array
        Points to be transformed using the TPS model.
    Returns 
    -------
    transformed_pts : (N, 2) array
    """


    def __init__(self):
        self.w = None
        self.a = None
        self.src = None

    def _kernel(self, r):
        with np.errstate(divide='ignore', invalid='ignore'):
            K = r**2 * np.log(r)
        K[np.isnan(K)] = 0
        return K

    def fit(self, src, dst):
        M = src.shape[0]
        # assemble
        D = cdist(src, src)
        K = self._kernel(D)
        P = np.hstack([np.ones((M,1)), src])
        A = np.zeros((M+3, M+3))
        A[:M,:M]   = K
        A[:M,M:]   = P
        A[M:,:M]   = P.T
        Y = np.vstack([dst, np.zeros((3,2))])
        sol = np.linalg.solve(A, Y)
        self.w = sol[:M]
        self.a = sol[M:]
        self.src = src

    def transform(self, pts):
        D = cdist(pts, self.src)
        U = self._kernel(D)
        non_affine = U.dot(self.w)
        P = np.hstack([np.ones((pts.shape[0],1)), pts])
        affine     = P.dot(self.a)
        return non_affine + affine


def extract_ordered_boundary(mesh,bars):
    """
    Extract the ordered boundary loop from the mesh based on the provided edges (bars).
    Parameters
    ----------
    mesh : PyEITMesh
        The mesh object containing the nodes and elements.
    bars : array-like   
        An array of shape (N, 2) representing the edges of the mesh.
        Each row contains two node indices that form an edge.
    Returns
    ------- 
    bnd_idx : array
        Indices of the nodes forming the ordered boundary loop.
    bnd_pts : array
        Coordinates of the nodes forming the ordered boundary loop.
    Notes
    -----   
    This function constructs the adjacency dictionary for the edges,
    then traverses the boundary loop starting from the first edge.
    """
 
    # 2) construis le dictionnaire d'adjacence sur le contour
    # chaque nœud y apparaît exactement dans 2 arêtes (boucle fermée)
    uniq = np.unique(bars)
    neigh = {i: [] for i in uniq}
    for i,j in bars:
        neigh[i].append(j)
        neigh[j].append(i)

    # 3) parcours la boucle
    start = bars[0,0]
    boundary = [start]
    prev = None
    curr = start
    while True:
        nbrs = neigh[curr]
        # choisis le suivant : celui qui n'est pas le précédent
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        boundary.append(nxt)
        prev, curr = curr, nxt

    bnd_idx = np.array(boundary, dtype=int)
    bnd_pts = mesh.node[bnd_idx, :2]
    return bnd_idx, bnd_pts


def resample_contour_by_arclength(pts, N):
    """
    Resample a contour defined by points `pts` to have exactly `N` points
    evenly spaced along the contour's length.
    Parameters
    ----------
    pts : (K, 2) array
        Original contour points.
    N : int
        Number of points to sample along the contour.
    Returns
    -------
    new_pts : (N, 2) array
        Resampled contour points, evenly spaced along the original contour.
    
    
    Parameters
    ----------  
    pts : (K, 2) array
        Original contour points.
    N : int
        Number of points to sample along the contour.   
    Returns
    -------
    new_pts : (N, 2) array
        Resampled contour points, evenly spaced along the original contour.
        
    Notes
    -----
    This function ensures that the contour is closed by appending the first point at the end.
    It computes the cumulative distance along the contour and interpolates to find
    evenly spaced points.
    If `N` is less than the number of original points, it will downsample the contour.
    If `N` is greater, it will interpolate additional points.
    If `N` is equal to the number of original points, it will return the original points.

    """
    # 1) Assurer que le contour est bien fermé en re-ajoutant le 1er point à la fin
    pts_closed = np.vstack([pts, pts[0]])
    
    # 2) Calculer les longueurs de chaque segment
    seg_vecs    = np.diff(pts_closed, axis=0)              # (K,2)
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)         # (K,)
    
    # 3) Distance cumulée le long du contour
    cumdist = np.concatenate([[0], np.cumsum(seg_lengths)])  # (K+1,)
    total_length = cumdist[-1]
    
    # 4) Positions cibles régulièrement réparties sur [0, total_length)
    sample_d = np.linspace(0, total_length, N, endpoint=False)
    
    # 5) Pour chaque position d, trouver le segment et interpoler
    new_pts = np.zeros((N, 2))
    for i, d in enumerate(sample_d):
        # indice du segment précédent le point d
        idx = np.searchsorted(cumdist, d) - 1
        idx = np.clip(idx, 0, len(seg_lengths)-1)
        # paramètre t dans [0,1] sur ce segment
        t = (d - cumdist[idx]) / seg_lengths[idx]
        # interpolation linéaire
        new_pts[i] = (1 - t) * pts_closed[idx] + t * pts_closed[idx+1]
    
    return new_pts

def align_contours(src, dst):
    """
    Aligns the source contour `src` to the destination contour `dst` by finding the best circular shift
    and whether to reverse the source contour.
    Parameters
    ----------
    src : (N, 2) array
        Source contour points to be aligned.
    dst : (N, 2) array
        Destination contour points to align to.
    Returns
    -------
    src_aligned : (N, 2) array
        Aligned source contour points.
    best_shift : int
        The best circular shift applied to the source contour.
    reversed_flag : bool
        Whether the source contour was reversed to achieve the best alignment.
    Notes
    -----   
    This function tests all possible circular shifts of the source contour
    and both orientations (normal and reversed) to find the one that minimizes
    the squared error with respect to the destination contour.
    
    """
    N = src.shape[0]
    best_err = np.inf
    best_shift = 0
    reversed_flag = False

    # tester les deux orientations : normale et renversée
    for rev in (False, True):
        candidate = src if not rev else src[::-1]
        # tester toutes les rotations circulaires
        for k in range(N):
            rolled = np.roll(candidate, -k, axis=0)
            err = np.sum((rolled - dst)**2)
            if err < best_err:
                best_err = err
                best_shift = k
                reversed_flag = rev

    # construire la version alignée finale
    aligned = src[::-1] if reversed_flag else src
    src_aligned = np.roll(aligned, -best_shift, axis=0)
    return src_aligned, best_shift, reversed_flag



def boundary_loop_indices_from_edges(bars):
    """
    Extracts the ordered boundary loop indices from the provided edges (bars).
    Parameters
    ----------
    bars : array-like   
        An array of shape (N, 2) representing the edges of the mesh.
        Each row contains two node indices that form an edge.
    Returns
    ------- 
    loop : array
        Indices of the nodes forming the ordered boundary loop.
    Notes
    -----
    This function constructs an adjacency dictionary for the edges,
    then traverses the boundary loop starting from the first edge.
    It ensures that the loop is closed by returning to the starting node.
    """
    adj = {}
    for u, v in bars:
        u, v = int(u), int(v)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    # parcourir le cycle
    start = bars[0, 0]
    prev, cur = None, int(start)
    loop = [cur]
    while True:
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    return np.array(loop, dtype=int)

def ensure_ccw(nodes_xy, loop_idx):
    """
    Ensure the boundary loop is counter-clockwise (CCW) by checking the area.
    Parameters
    ----------
    nodes_xy : (N, 2) array
        Coordinates of the nodes in 2D space.
    loop_idx : array-like   
        Indices of the nodes forming the boundary loop.
    Returns 
    -------
    loop_idx : array    
        Indices of the nodes forming the boundary loop, ordered CCW.
    Notes
    -----
    This function calculates the area of the polygon formed by the loop indices.
    If the area is negative (indicating a clockwise orientation), it reverses the loop indices
    to ensure a counter-clockwise orientation.
    """
    # calcul de l’aire par la formule du polygone (shoelace)
    pts = nodes_xy[loop_idx]
    x, y = pts[:,0], pts[:,1]
    area2 = np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))
    # si orientation horaire on inverse
    return loop_idx[::-1] if area2 < 0 else loop_idx

def smooth_mesh_boundary(mesh: PyEITMesh, iterations:int=20, alpha:float=0.5,bars:list=None):
    """
    Smooth the boundary of the mesh by averaging the positions of nodes along the boundary.
    Parameters
    ----------
    mesh : PyEITMesh    
        The mesh object containing the nodes and elements.
    iterations : int, optional
        Number of smoothing iterations to perform (default is 20).
    alpha : float, optional 
        Smoothing factor (default is 0.5).
    bars : list, optional   
        List of edges (bars) defining the boundary of the mesh.
        If not provided, the boundary will be extracted from the mesh.
    Returns
    -------
    mesh_smooth : PyEITMesh
        A new mesh object with smoothed boundary nodes.
        
    """
    nodes = mesh.node.copy()
    # 1) extraire et ordonner la boucle frontière
    loop = boundary_loop_indices_from_edges(bars)
    loop = ensure_ccw(nodes[:,:2], loop)

    for _ in range(iterations):
        new_nodes = nodes.copy()
        coords = nodes[loop, :2]
        N = len(loop)
        for k, idx in enumerate(loop):
            prev = coords[k-1] if k>0    else coords[-1]
            curr = coords[k]
            nxt  = coords[k+1] if k<N-1 else coords[0]
            avg  = (prev + curr + nxt)/3
            # on fait une interpolation curr → avg
            new_nodes[idx, 0:2] = (1-alpha)*curr + alpha*avg
        nodes = new_nodes

    # reconstruire le mesh
    mesh_smooth = PyEITMesh(
        node    = nodes,
        element = mesh.element,
        el_pos  = mesh.el_pos
    )
    mesh_smooth.perm = mesh.perm
    return mesh_smooth

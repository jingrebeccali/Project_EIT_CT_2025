from shapely.geometry import Polygon, Point
import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.eit.greit    as greit
import pyeit.eit.bp       as bp
from pyeit.visual.plot    import create_mesh_plot, create_plot
from pyeit.mesh.shape import *


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
    def fd_body(pts: np.ndarray):
        """
        pts: shape (M,2) each row = (x,y) in the same coordinate system as body_poly
        returns:  array of length‐M of signed distances
        """
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("fd_body(pts): pts must be shape (M,2).")
        
        signed_d = np.zeros(pts.shape[0], dtype=float)
        for i, (x, y) in enumerate(pts):
            p = Point(x, y)
            d = p.distance(body_poly.exterior)  # always ≥ 0
            if body_poly.contains(p):
                signed_d[i] = -d
            else:
                signed_d[i] = +d
        return signed_d
    
    return fd_body


def compute_element_labels(mask, nodes, elements):
    """
    Calculate for each triangle the label of the mask by taking the barycenter of the triangle
    and converting it to PIXEL coordinates.
    Returns an array of length the number of triangles, where each element is the label of the mask at the barycenter of the triangle.
    """
    
    H, W = mask.shape
    Ne = elements.shape[0]

    # get the coordinates of the nodes of each triangle
    pts_tri = nodes[elements, :2]  # shape (Ne, 3, 2)

    # obtain the barycenter of each triangle
    centers = pts_tri.mean(axis=1)  # shape (Ne, 2)

    # convert the barycenter coordinates to pixel coordinates
    x_world = centers[:, 0]
    y_world = centers[:, 1]

    u = ((x_world + 1.0) * 0.5) * (W - 1)
    v = (   1.0 - (y_world + 1.0) * 0.5) * (H - 1)

    xi = np.round(u).astype(int)
    yi = np.round(v).astype(int)

    # clip the pixel coordinates to be within the mask bounds
    xi = np.clip(xi, 0, W - 1)
    yi = np.clip(yi, 0, H - 1)

    # retrieve the mask value at the pixel coordinates
    return mask[yi, xi]  # vecteur (Ne,)



def set_protocol(n_el,dist_exc,step_meas):
    """
    Set the protocol for EIT measurements.
    Returns a protocol object with the specified parameters.
    
    n_el =    # Number of electrodes
    dist_exc  # Distance between excitation electrodes
    step_meas # Step size for measurements

    """
    return protocol.create(n_el, dist_exc=dist_exc, step_meas=step_meas)

def set_condu_eit(mesh,condu_body, condu_lung, labels_elems):
    """
    Set the conductivity values for the EIT mesh.
    Returns an array of conductivity values for each element in the mesh.
    mesh: EIT mesh object, of type pyeit.mesh.Mesh
    condu_body: Conductivity value for the body
    condu_lung: Conductivity value for the lung
    labels_elems: Array of labels for each element in the mesh
    Ne = labels_elems.shape[0]
    Output:
    perm: Array of conductivity values for each element in the mesh
    perm0: Array of initial conductivity values (all ones)(Reference conductivity)
    """
    Ne=mesh.element.shape[0]
    perm0 = np.ones(Ne)
    perm = perm0.copy()
    perm[labels_elems == 1] = condu_lung 
    perm[labels_elems == 2] = condu_body  # to adapt later for simulations with other organs
    mesh.perm = perm
    return perm,perm0
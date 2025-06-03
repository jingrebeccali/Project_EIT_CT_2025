from shapely.geometry import Polygon, Point
import numpy as np




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


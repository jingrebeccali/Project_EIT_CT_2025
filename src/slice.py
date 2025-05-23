import numpy as np
import trimesh

from typing import Optional, Tuple

# help funtion
def apply_bound(mask, coords, bounds):
        #Input: mask : high for coordinates wich enter in the bounds, low for the rest(all true at first)
        #       coords : coords of the points of the mesh
        #       bounds : the segment we want to slice(in one direction)
        mn, mx = sorted(bounds)
        return mask & (coords >= mn) & (coords <= mx)



def crop_mesh_by_bounds(verts,faces,x_bounds:Optional[Tuple[float,float]] = None,y_bounds:Optional[Tuple[float,float]] = None,z_bounds:Optional[Tuple[float,float]] = None):
    # Input:   verts,faces: vertices and faces of the mesh
    #          the rest: the three segmants in direction of x,y,z that are wanted.(in mm)
    #Output :  Same type of output as make_mesh_MC, the mesh of the sliced part, its vertices, faces,volume and surface
    inside=np.ones(len(verts),dtype=bool)
 
    if x_bounds is not None:
        inside=apply_bound(inside,verts[:,0],x_bounds)
    if y_bounds is not None:
        inside=apply_bound(inside,verts[:,1],y_bounds)
    if z_bounds is not None:
        inside=apply_bound(inside,verts[:,2],z_bounds)

    # after keeping the wanted points, keep only faces whose all three vertices are kept
    face_mask=inside[faces].all(axis=1)
    kept=face_mask.sum()
    if kept==0:    #throw an exception in case no faces are kept
        raise ValueError("No faces remain after slicing")

    # re-construct the mesh
    idx,inv=np.unique(faces[face_mask], return_inverse=True)
    new_verts=verts[idx]
    new_faces=inv.reshape(-1,3)

    submesh = trimesh.Trimesh(vertices=new_verts,faces=new_faces,process=False)

    vol_cm3=abs(submesh.volume)/1000.0
    surf_cm2=submesh.area/100.0

    return submesh,new_verts,new_faces,vol_cm3,surf_cm2


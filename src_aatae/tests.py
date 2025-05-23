import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from visualise_ALPHA import *
from visualise_MC import *
from slice import *



case_id="s0011"
organ_name="rib_left_1.nii.gz"

ms,p,faces,vol_cm3,surf_cm2=make_mesh_MC(case_id,organ_name,level=0.5)#to adjust

ms_alpha,p_alpha,faces_alpha,vol_alpha_cm3,surf_alpha_cm2=make_mesh_alpha(case_id,organ_name)

ms_sliced,a,b,c,d=crop_mesh_by_bounds(p,faces,x_bounds=None,y_bounds=[168,210],z_bounds=None) #to adjust

print(f"Volume α-wrap:{vol_alpha_cm3:.2f} cm³")
print(f"Surface α-wrap:{surf_alpha_cm2:.2f}cm²")

print(f"Volume MC:{vol_cm3:.2f}cm³")
print(f"Surface MC:{surf_cm2:.2f}cm²")
print(f"Volume MC sliced: {c:.2f}cm³")
print(f"Surface MC sliced:{d:.2f}cm²")

view_mesh_3D(p,faces,f"{organ_name}_MC")
view_mesh_3D(p_alpha,faces_alpha,f"{organ_name}_alpha")
view_mesh_3D(a,b,f"{organ_name}_MC_sliced")




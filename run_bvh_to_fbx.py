import bpy
import os


dir_name = 'main2_09_25_15_56_WalkingDog-1_3d_1+2d_1e-05+lim_0.1+temp_0.1+filter_51.99mm_0_1.30s'
file_name = '09_25_15_56_WalkingDog-1_dofs'
data_path = 'E:/Projects/Kinematic_Skeleton_Fitting/out/' + dir_name + '/' + file_name + '.bvh'
save_path = 'E:/Projects/Kinematic_Skeleton_Fitting/out/' + dir_name + '/' + file_name + '.fbx'

if "Cube" in bpy.data.meshes:
    mesh = bpy.data.meshes["Cube"]
    bpy.data.meshes.remove(mesh)
while len(bpy.data.objects) > 0:
    bpy.data.objects.remove(bpy.data.objects[0])
# cmu: Y;   h36m/S1: Y;     mixamo: Y;      hmmr: -Y
bpy.ops.import_anim.bvh(filepath=data_path, axis_up='-Y')
bpy.ops.export_scene.fbx(filepath=save_path)


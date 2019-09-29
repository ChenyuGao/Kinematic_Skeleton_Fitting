import pickle as pkl
import numpy as np
import cv2
import os
import BVH
import Animation
from Quaternions import Quaternions
from config import *


dir_name = 'main2_09_25_15_56_WalkingDog-1_3d_1+2d_1e-05+lim_0.1+temp_0.1+filter_51.99mm_0_1.30s'
file_name = '09_25_15_56_WalkingDog-1_dofs'
data_path = 'E:/Projects/Kinematic_Skeleton_Fitting/out/' + dir_name + '/' + file_name + '.txt'
save_path = 'E:/Projects/Kinematic_Skeleton_Fitting/out/' + dir_name + '/' + file_name + '.bvh'
dofs = np.loadtxt(data_path)
frames = dofs.shape[0]
offsets = input_tpose_j3d - input_tpose_j3d[j17_parents]
offsets[0] = 0
positions = offsets[np.newaxis].repeat(frames, axis=0)  # (frames, 17, 3)
positions[:, 0, :3] = dofs[:, :3]
parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
frametime = 0.1
joints_list = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Spine', 'Thorax',
               'Neck', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Hand', 'R_Shoulder', 'R_Elbow', 'R_Hand']

axis_angle = np.zeros((frames, 17, 3))
axis_angle[:, 0] = dofs[:, 3: 6]
axis_angle[:, 1] = dofs[:, 6: 9]
axis_angle[:, 2, 0] = dofs[:, 9]
axis_angle[:, 4] = dofs[:, 10: 13]
axis_angle[:, 5, 0] = dofs[:, 13]
axis_angle[:, 7] = dofs[:, 14: 17]
axis_angle[:, 9] = dofs[:, 17: 20]
axis_angle[:, 11] = dofs[:, 20: 23]
axis_angle[:, 12, 1] = dofs[:, 23]
axis_angle[:, 14] = dofs[:, 24: 27]
axis_angle[:, 15, 1] = dofs[:, 27]
# axis_angle[:, :, 1] = -axis_angle[:, :, 1]
angles = np.linalg.norm(axis_angle, axis=2)[..., np.newaxis]
vector = np.divide(axis_angle, angles, where=angles != 0)
quaternions = np.concatenate((np.cos(angles / 2), vector * np.sin(angles / 2)), axis=2)
rotations = Quaternions(quaternions)
orients = Quaternions.id(0)
for i in range(offsets.shape[0]):
    orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
anim = Animation.Animation(rotations, positions, orients, offsets, parents)
# Put on Floor
joints = Animation.positions_global(anim)       # (Frames, 24, 3)
fid_l, fid_r = np.array([4, 5]), np.array([8, 9])
foot_heights = np.maximum(joints[:, fid_l, 1],
                          joints[:, fid_r, 1]).max(axis=1)      # (Frames, )
# floor_height = softmax(foot_heights, softness=0.5, axis=0)
positions[:, :, 1] -= foot_heights.reshape((foot_heights.shape[0], 1))
anim = Animation.Animation(rotations, positions, orients, offsets, parents)

BVH.save(save_path, anim, joints_list, frametime)


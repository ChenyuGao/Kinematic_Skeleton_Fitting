import numpy as np
from scipy.optimize import root
import h5py
import os
from Quaternions import Quaternions
import Animation
from visualization import plot_2skeleton

'''
dofs:
0: pelvis_tx -> 0
1: pelvis_ty -> 0
2: pelvis_tz -> 0
3: pelvis_rx -> 0
4: pelvis_ry -> 0
5: pelvis_rz -> 0
6: right_hip_rx -> 1
7: right_hip_ry -> 1
8: right_hip_rz -> 1
9: right_knee_rx -> 2
10: left_hip_rx -> 4
11: left_hip_ry -> 4
12: left_hip_rz -> 4
13: left_knee_rx -> 5
14: spine_rx -> 7
15: spine_ry -> 7
16: spine_rz -> 7
17: neck_x -> 9
18: neck_y -> 9
19: neck_z -> 9
20: left_shoulder_rx -> 11
21: left_shoulder_ry -> 11
22: left_shoulder_rz -> 11
23: left_elbow_ry -> 12
24: right_shoulder_rx -> 14
25: right_shoulder_ry -> 14
26: right_shoulder_rz -> 14
27: right_elbow_ry -> 15
'''
dofs_limit = np.array([
    [-2.5, 1.4],    # 6
    [-0.8, 1.2],    # 7
    [-0.6, 0.6],    # 8
    [-0. , 2.5],    # 9
    [-2.5, 1.4],    # 10
    [-1.2, 0.8],    # 11
    [-0.6, 0.6],    # 12
    [-0. , 2.5],    # 13
    [ 0. , 0.3],    # 14
    [-1.0, 1.0],    # 15
    [-0.1, 0.1],    # 16
    [-0.5, 0.5],    # 17
    [-0.667, 0.667],    # 18
    [-0.3, 0.3],    # 19
    [-1.0, 1.0],    # 20
    [-0.7, 2.0],    # 21
    [-1.4, 1.8],    # 22
    [0.065, 2.7],     # 23
    [-1.0, 1.0],    # 24
    [-2.0, 0.7],    # 25
    [-1.8, 1.4],    # 26
    [-2.7, -0.065],    # 27
])
j3d, j2d, cam = 0, 0, 0
j17_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
# passive_marker_man-tpose is good, xbot is bad
# human forward to  -z
mixamo_tpose_j3d = np.array([[  0.        ,   0.        ,   0.        ],
                             [ -9.63380134,   3.27329199,   0.        ],
                             [ -9.64029234,  45.03542952,  -0.55685251],
                             [ -9.63379952,  84.18255651,  -2.57603751],
                             [  9.63379866,   3.27330101,   0.        ],
                             [  9.62730693,  45.03545572,  -0.55539312],
                             [  9.63380007,  84.18250635,  -2.57608716],
                             [  0.00000397, -25.480301  ,   1.4115    ],
                             [  0.00000397, -39.119904  ,   0.1192    ],
                             [  0.00000397, -52.9973    ,  -4.234999  ],
                             [  0.00000397, -57.023911  ,  -3.376499  ],
                             [ 18.46330306, -48.19389392,  -5.4674999 ],
                             [ 46.15030306, -48.19389392,  -5.4674999 ],
                             [ 74.02410106, -48.19389392,  -5.4674999 ],
                             [-18.46329494, -48.19389186,  -5.46749996],
                             [-46.15029694, -48.19389186,  -5.46749996],
                             [-74.02409894, -48.19389186,  -5.46749996]])
# mixamo_tpose_j3d[:, 0] = -mixamo_tpose_j3d[:, 0]


def compute_joints_from_dofs(dofs, cam):
    frame_num = 1
    axis_angle = np.zeros((frame_num, 17, 3))
    # 28 - (tx, ty, tz) = 25
    axis_angle[0, 0] = dofs[0:3]
    axis_angle[0, 1] = dofs[3:6]
    axis_angle[0, 2, 0] = dofs[6]
    axis_angle[0, 4] = dofs[7: 10]
    axis_angle[0, 5, 0] = dofs[10]
    axis_angle[0, 7] = dofs[11: 14]
    axis_angle[0, 9] = dofs[14: 17]
    axis_angle[0, 11] = dofs[17: 20]
    axis_angle[0, 12, 1] = dofs[20]
    axis_angle[0, 14] = dofs[21: 24]
    axis_angle[0, 15, 1] = dofs[24]
    quaternions = Quaternions.from_angle_axis(np.sqrt(np.sum(axis_angle ** 2, axis=-1)), axis_angle)
    # y-up, z-forward, x-right
    global mixamo_tpose_j3d, j3d
    offsets = mixamo_tpose_j3d - mixamo_tpose_j3d[j17_parents]
    offsets[0] = j3d[0] * 100   # m -> cm
    positions = offsets[np.newaxis].repeat(frame_num, axis=0)  # (frames, jointsNum, 3)
    orients = Quaternions.id(0)
    for i in range(offsets.shape[0]):
        orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
    anim = Animation.Animation(quaternions, positions, orients, offsets, j17_parents)
    j3d_pre = Animation.positions_global(anim) / 100    # cm -> m
    j2d_pre = np.dot(np.divide(j3d_pre, j3d_pre[:, :, 2:], where=j3d_pre[:, :, 2:] != 0), cam.T)[:, :, :2]
    return j3d_pre[0], j2d_pre[0]


def infer_camera_intrinsics(points2d, points3d):
    """Infer camera instrinsics from 2D<->3D point correspondences."""
    pose2d = points2d.reshape(-1, 2)
    pose3d = points3d.reshape(-1, 3)
    x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
    x2d = (pose2d[:, 0] * pose3d[:, 2])
    alpha_x, x_0 = list(np.linalg.lstsq(x3d, x2d, rcond=-1)[0].flatten())
    y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
    y2d = (pose2d[:, 1] * pose3d[:, 2])
    alpha_y, y_0 = list(np.linalg.lstsq(y3d, y2d, rcond=-1)[0].flatten())
    return np.array([alpha_x, 0, x_0, 0, alpha_y, y_0, 0, 0, 1]).reshape((3, 3))


def mixamo_skeleton_fit(j3d_h36m):
    global mixamo_tpose_j3d
    j17_ori = mixamo_tpose_j3d / 100
    bone_ori = np.linalg.norm(j17_ori - j17_ori[j17_parents], axis=-1)[1:].reshape((-1, 1))
    bone_h36m = np.linalg.norm(j3d_h36m - j3d_h36m[j17_parents], axis=-1)[1:].reshape((-1, 1))
    bone_ori_norm = (j17_ori - j17_ori[j17_parents])[1:] / bone_ori
    bone_fit = bone_ori_norm * bone_h36m    # 4150
    j17_fit = j17_ori.copy()
    j17_fit[[1, 4, 7]] = j17_fit[[0, 0, 0]] + bone_fit[[0, 3, 6]]
    j17_fit[[2, 5, 8]] = j17_fit[[1, 4, 7]] + bone_fit[[1, 4, 7]]
    j17_fit[[3, 6, 9, 11, 14]] = j17_fit[[2, 5, 8, 8, 8]] + bone_fit[[2, 5, 8, 10, 13]]
    j17_fit[[10, 12, 15]] = j17_fit[[9, 11, 14]] + bone_fit[[9, 11, 14]]
    j17_fit[[13, 16]] = j17_fit[[12, 15]] + bone_fit[[12, 15]]
    return j17_fit


def h36m_skeleton_fit(j3d_h36m):
    head = j3d_h36m[:, 10] - j3d_h36m[:, 9]
    neck = j3d_h36m[:, 9] - j3d_h36m[:, 8]
    j3d_h36m[:, 8] = j3d_h36m[:, 8] - (j3d_h36m[:, 8] - j3d_h36m[:, 7]) / 3
    j3d_h36m[:, 9] = j3d_h36m[:, 8] + neck
    j3d_h36m[:, 10] = j3d_h36m[:, 9] + head
    j3d_h36m[:, 0] = ((j3d_h36m[:, 1] + j3d_h36m[:, 4] + j3d_h36m[:, 7]) / 3 - j3d_h36m[:, 0]) / 2 + j3d_h36m[:, 0]
    global mixamo_tpose_j3d
    j17_ori = mixamo_tpose_j3d / 100     # cm -> m
    bone_ori = np.linalg.norm(j17_ori - j17_ori[j17_parents], axis=-1)[1:].reshape((-1, 1))
    bone_h36m = np.linalg.norm(j3d_h36m - j3d_h36m[:, j17_parents], axis=-1)[:, 1:].reshape((j3d_h36m.shape[0], -1, 1))
    bone_h36m_norm = (j3d_h36m - j3d_h36m[:, j17_parents])[:, 1:] / bone_h36m
    bone_fit = bone_h36m_norm * bone_ori
    j17_fit = j3d_h36m.copy()
    j17_fit[:, [1, 4, 7]] = j17_fit[:, [0, 0, 0]] + bone_fit[:, [0, 3, 6]]
    j17_fit[:, [2, 5, 8]] = j17_fit[:, [1, 4, 7]] + bone_fit[:, [1, 4, 7]]
    j17_fit[:, [3, 6, 9, 11, 14]] = j17_fit[:, [2, 5, 8, 8, 8]] + bone_fit[:, [2, 5, 8, 10, 13]]
    j17_fit[:, [10, 12, 15]] = j17_fit[:, [9, 11, 14]] + bone_fit[:, [9, 11, 14]]
    j17_fit[:, [13, 16]] = j17_fit[:, [12, 15]] + bone_fit[:, [12, 15]]
    return j17_fit


def read_joints_from_h36m(annot_dir='E:/Datasets/Human3.6m/processed/S11/WalkingDog-2/'):
    cam_id = ['54138969', '55011271', '58860488', '60457274']
    annot_file_path = annot_dir + 'annot-' + cam_id[0] + '.h5'
    with h5py.File(annot_file_path, 'r') as annot:
        j2d = np.array(annot['joints2D'])
        j3d = np.array(annot['joints3D-univ'])      # (f, 32, 3)
    h36m_to_17_index = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    j3d, j2d = j3d[:, h36m_to_17_index], j2d[:, h36m_to_17_index]
    j2d = j2d / 1000 * 2 - 1.
    j3d = j3d / 1000    # mm -> m
    global mixamo_tpose_j3d
    # mixamo_tpose_j3d = mixamo_skeleton_fit(j3d[0]) * 100
    j3d = h36m_skeleton_fit(j3d)
    j_cal_cam = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16])
    cam = infer_camera_intrinsics(j2d[:, j_cal_cam], j3d[:, j_cal_cam])
    j2d = np.dot(j3d / j3d[:, :, 2:], cam.T)[:, :, :2]
    confidence = np.ones((j2d.shape[0], j2d.shape[1], 1))
    j2d = np.concatenate([j2d, confidence], axis=2)
    return j3d, j2d, cam


def optimize(dofs):
    global j3d, j2d, cam
    j3d_pre, j2d_pre = compute_joints_from_dofs(dofs, cam)   # (17, 3/2)
    e3d = np.mean((j3d - j3d_pre) ** 2, axis=-1)
    j_cal_cam = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16])
    e2d = np.mean(j2d[:, 2:] * (j2d[:, :2] - j2d_pre[:, :2]) ** 2, axis=-1)
    error = []
    for j in range(e3d.shape[0]):
        error.append(e3d[j])
    for j in range(e2d.shape[0]):
        error.append(e2d[j])
    for d in range(dofs_limit.shape[0]):
        if dofs[d + 3] < dofs_limit[d, 0]:
            error.append((dofs[d + 3] - dofs_limit[d, 0]) ** 2)
        elif dofs[d + 3] > dofs_limit[d, 1]:
            error.append((dofs[d + 3] - dofs_limit[d, 1]) ** 2)
        else:
            error.append(0.)
    return error


def main():
    np.set_printoptions(suppress=True)
    global j3d, j2d, cam
    j3ds, j2ds, cam = read_joints_from_h36m()      # (F, 17, 3/3)
    frame_num = j3ds.shape[0]
    dofs = np.zeros((frame_num, 28))
    dofs[:, :3] = j3ds[:, 0]
    # for f in range(frame):
    f = 200
    print('-------------------------------------')
    j3d, j2d = j3ds[f], j2ds[f]
    dof = dofs[f]
    sol = root(optimize, dof[3:], method='lm')
    print(dof[:3])
    print(sol.x)
    print(sol.success)
    print(sol.nfev)
    print(sol.message)
    print(np.sum(sol.fun))
    dof[3:] = sol.x
    j3d_pre, j2d_pre = compute_joints_from_dofs(dof[3:], cam)
    mpjpe = np.mean(np.linalg.norm(j3d * 1000 - j3d_pre * 1000, axis=-1))
    print('MPJPE: ' + str(mpjpe) + ' mm')
    plot_2skeleton(j3d * 100, j3d_pre * 100, mpjpe)

    # all bone length
    # print(np.sum(np.linalg.norm(j3ds - j3ds[:, j17_parents], axis=-1)[:, 1:], axis=-1) * 1000)    # 4150 mm

    # test cam (ok)
    # j2ds_pre = np.dot(j3ds / j3ds[:, :, 2:], cam.T)[:, :, :2]
    # print(j2ds[0])
    # print(j2ds_pre[0])
    # print(np.mean(np.linalg.norm(j2ds[:, :, :2] - j2ds_pre, axis=-1)))      # 2.15 px

    # test FK (ok)
    # dof[9] = -0.
    # j3d_pre1, j2d_pre1 = compute_joints_from_dofs(dof[3:], cam)
    # dof[9] = 2.5
    # j3d_pre2, j2d_pre2 = compute_joints_from_dofs(dof[3:], cam)
    # plot_2skeleton(j3d_pre1 * 100, j3d_pre2 * 100)


if __name__ == '__main__':
    main()

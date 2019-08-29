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
    [-0.1, 2.5],    # 9
    [-2.5, 1.4],    # 10
    [-1.2, 0.8],    # 11
    [-0.6, 0.6],    # 12
    [-0.1, 2.5],    # 13
    [ 0. , 0.5],    # 14
    [-1.0, 1.0],    # 15
    [-0.1, 0.1],    # 16
    [-0.5, 0.5],    # 17
    [-0.667, 0.667],    # 18
    [-0.3, 0.3],    # 19
    [-1.0, 1.0],    # 20
    [-2.0, 0.7],    # 21
    [-1.4, 1.8],    # 22
    [-2.7, -0.065],     # 23
    [-1.0, 1.0],    # 24
    [-0.7, 2.0],    # 25
    [-1.8, 1.4],    # 26
    [0.065, 2.7],    # 27
])
j3d, j2d, cam = 0, 0, 0


def compute_joints_from_dofs(dofs, cam):
    frame_num = 1
    axis_angle = np.zeros((frame_num, 17, 3))
    # 28
    # axis_angle[0, 0] = dofs[3:6]
    # axis_angle[0, 1] = dofs[6:9]
    # axis_angle[0, 2, 0] = dofs[9]
    # axis_angle[0, 4] = dofs[10: 13]
    # axis_angle[0, 5, 0] = dofs[13]
    # axis_angle[0, 7] = dofs[14: 17]
    # axis_angle[0, 9] = dofs[17: 20]
    # axis_angle[0, 11] = dofs[20: 23]
    # axis_angle[0, 12, 1] = dofs[23]
    # axis_angle[0, 14] = dofs[24: 27]
    # axis_angle[0, 15, 1] = dofs[27]
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
    # default_j3d = np.array([[  0.        ,   0.        ,   0.        ],
    #                         [ 10.16021151,   3.45215121,   0.        ],
    #                         [ 10.16705719,  47.49625544,  -0.58728004],
    #                         [ 10.16020959,  88.78245973,  -2.71679735],
    #                         [-10.16020868,   3.45216072,   0.        ],
    #                         [-10.15336223,  47.49628308,  -0.58574091],
    #                         [-10.16021017,  88.78240684,  -2.71684972],
    #                         [ -0.00000418, -26.8725956 ,   1.48862718],
    #                         [ -0.00000418, -41.25749378,   0.12571333],
    #                         [ -0.00000418, -55.89317845,  -4.46640781],
    #                         [ -0.00000418, -60.13981153,  -3.56099766],
    #                         [-19.47217486, -50.82730465,  -5.76625503],
    #                         [-48.67204791, -50.82730465,  -5.76625503],
    #                         [-78.06892597, -50.82730465,  -5.76625503],
    #                         [ 19.47216629, -50.82730249,  -5.76625509],
    #                         [ 48.67204145, -50.82730249,  -5.76625509],
    #                         [ 78.06892373, -50.82730249,  -5.76625509]])
    # # from default skeleton to h36m skeleton
    # # default_j3d[0] = (default_j3d[1] + default_j3d[4]) / 2
    # # default_j3d[8] = (default_j3d[11] + default_j3d[14]) / 2
    # # bone_head = default_j3d[10] - default_j3d[9]
    # # default_j3d[9] = (default_j3d[9] - default_j3d[8]) * 1.5 + default_j3d[8]
    # # default_j3d[10] = default_j3d[9] + bone_head * 1.5
    # j17_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    # sum_bone_length = np.sum(np.linalg.norm(default_j3d - default_j3d[j17_parents], axis=-1)[1:])     # 394cm
    # mean_bone_length = 415
    # default_j3d = default_j3d * mean_bone_length / sum_bone_length
    # offsets = default_j3d - default_j3d[j17_parents, :]
    offsets = np.array([[  0.        ,   0.        ,   0.        ],
                        [ 10.14939171,   3.44847495,   0.        ],
                        [  0.00683839,  43.99720083,  -0.58665463],
                        [ -0.00684031,  41.24223783,  -2.12724955],
                        [-10.14938888,   3.44848445,   0.        ],
                        [  0.00683916,  43.99721894,  -0.58511715],
                        [ -0.00684065,  41.24215739,  -2.12883934],
                        [ -0.00000418, -26.84397847,   1.48704192],
                        [  0.        , -14.36957943,  -1.36146246],
                        [  0.        , -14.62009885,  -4.5872309 ],
                        [  0.        ,  -4.24211076,   0.90444597],
                        [-19.4514344 ,  -9.5596198 ,  -5.88569389],
                        [-29.16877755,   0.        ,  -0.        ],
                        [-29.36557277,   0.        ,  -0.        ],
                        [ 19.45143419,  -9.55961763,  -5.88569395],
                        [ 29.16877966,   0.        ,   0.        ],
                        [ 29.36557699,   0.        ,   0.        ]])
    offsets_fit = np.array([[  0.        ,   0.        ,  0.        ],
                            [ 12.58808624,   4.27707407,  0.        ],
                            [  0.0068832 ,  44.28552405, -0.59049911],
                            [ -0.00752333,  45.36034508, -2.33965902],
                            [-12.58810453,   4.27709326,  0.        ],
                            [  0.00688397,  44.28552475, -0.58895132],
                            [ -0.0075237 ,  45.36026917, -2.3414082 ],
                            [ -0.00000363, -23.31235405,  1.29140499],
                            [  0.        , -25.593152  , -2.42485286],
                            [  0.        , -11.55792713, -3.62643789],
                            [  0.        , -11.24742622,  2.39802539],
                            [-13.08113671,  -6.42886745, -3.95814338],
                            [-27.88827728,   0.        , -0.        ],
                            [-25.1733451 ,   0.        , -0.        ],
                            [ 13.08089567,  -6.4287476 , -3.95807053],
                            [ 27.88929236,   0.        ,  0.        ],
                            [ 25.17286803,   0.        ,  0.        ]])
    offsets = offsets_fit
    global j3d
    offsets[0] = j3d[0] * 100   # m -> cm
    positions = offsets[np.newaxis].repeat(frame_num, axis=0)  # (frames, jointsNum, 3)
    orients = Quaternions.id(0)
    for i in range(offsets.shape[0]):
        orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    anim = Animation.Animation(quaternions, positions, orients, offsets, parents)
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


def read_joints_from_h36m(annot_dir='E:/Datasets/Human3.6m/processed/S11/WalkingDog-2/'):
    cam_id = ['54138969', '55011271', '58860488', '60457274']
    annot_file_path = annot_dir + 'annot-' + cam_id[0] + '.h5'
    with h5py.File(annot_file_path, 'r') as annot:
        j2d = np.array(annot['joints2D'])
        j3d = np.array(annot['joints3D-univ'])
    j2d = j2d / 1000 * 2 - 1.
    j3d = j3d / 1000    # mm -> m
    cam = infer_camera_intrinsics(j2d, j3d)
    confidence = np.ones((j2d.shape[0], j2d.shape[1], 1))
    j2d = np.concatenate([j2d, confidence], axis=2)
    return j3d, j2d, cam


def optimize(dofs):
    global j3d, j2d, cam
    j3d_pre, j2d_pre = compute_joints_from_dofs(dofs, cam)   # (17, 3/2)
    e3d = np.mean((j3d - j3d_pre) ** 2, axis=-1)
    e2d = np.mean(j2d[:, 2:] * (j2d[:, :2] - j2d_pre[:, :2]) ** 2, axis=-1)
    error = []
    for j in range(j3d_pre.shape[0]):
        error.append(e3d[j])
        error.append(e2d[j])
    # for d in range(dofs_limit.shape[0]):
    #     if dofs[d + 3] < dofs_limit[d, 0]:
    #         error.append((dofs[d + 3] - dofs_limit[d, 0]) ** 2)
    #     else:
    #         error.append(0.)
    #     if dofs[d + 3] > dofs_limit[d, 1]:
    #         error.append((dofs[d + 3] - dofs_limit[d, 1]) ** 2)
    #     else:
    #         error.append(0.)
    return error


def main():
    np.set_printoptions(suppress=True)
    global j3d, j2d, cam
    j3ds, j2ds, cam = read_joints_from_h36m()      # (F, 32, 3/3)
    h36m_to_17_index = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    j3ds, j2ds = j3ds[:, h36m_to_17_index], j2ds[:, h36m_to_17_index]
    frame_num = j3ds.shape[0]
    dofs = np.zeros((frame_num, 28))
    dofs[:, :3] = j3ds[:, 0]
    # for f in range(frame):
    f = 200
    print('-------------------------------------')
    j3d, j2d = j3ds[f], j2ds[f]
    dof = dofs[f]
    sol = root(optimize, dof[3:], method='lm', tol=0.000001)
    print(dof[:3])
    print(sol.x)
    print(sol.success)
    print(sol.nfev)
    print(sol.message)
    print(np.sum(sol.fun))
    dof[3:] = sol.x
    j3d_pre, j2d_pre = compute_joints_from_dofs(dof[3:], cam)
    mpjpe = np.mean(np.linalg.norm(j3d * 1000 - j3d_pre * 1000, axis=-1))
    print('MPJPE: ' + str(mpjpe) + 'mm')
    plot_2skeleton(j3d * 100, j3d_pre * 100, mpjpe)

    # all bone length
    # joints_17_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    # print(np.sum(np.linalg.norm(j3ds - j3ds[:, joints_17_parents], axis=-1)[:, 1:], axis=-1) * 1000)    # 4150 mm

    # test cam (ok)
    # j2ds_pre = np.dot(j3ds / j3ds[:, :, 2:], cam.T)[:, :, :2]
    # print(j2ds[0])
    # print(j2ds_pre[0])
    # print(np.mean(np.linalg.norm(j2ds[:, :, :2] - j2ds_pre, axis=-1)))      # 2.15 px

    # test FK (ok)
    # dof[3:] = np.array([-0.00001906, -0.03794922, -2.12959883, -0.0024452, -2.24640591, 0.50617998, 0.00036421,
    # -0.00220767, 2.30437399, -0.51361581, 0.00022483, -0.00007493, -0.1492, 3.07439269, -0.0000725, -1.12847026,
    # 0.27410293, 2.23685529, 0.02577756, -0.60063622, 0.00352356, -2.03622539, 0.00842168, 0.90066839, 0.07240781])
    # j3d_pre, j2d_pre = compute_joints_from_dofs(dof[3:], cam)
    # print((j3d_pre - j3d_pre[0]) * 1000)
    # print((j3d - j3d[0]) * 1000)


if __name__ == '__main__':
    main()

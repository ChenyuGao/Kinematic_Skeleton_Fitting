from Quaternions import Quaternions
import Animation
from visualization import plot_2skeleton, frame_to_video
from dataloader import read_joints_from_h36m
from OneEuroFilter import OneEuroFilter
from config import *

from scipy.optimize import root
import os
from datetime import datetime
import time
import cv2
import numpy as np
np.set_printoptions(suppress=True)


def compute_joints_from_dofs(dofs, cam):
    global j3d, root_rot
    frame_num = 1
    axis_angle = np.zeros((frame_num, 17, 3))
    # 28 - (tx, ty, tz, rx, ry, rz) = 22
    axis_angle[0, 0] = root_rot
    axis_angle[0, 1] = dofs[0: 3]
    axis_angle[0, 2, 0] = dofs[3]
    axis_angle[0, 4] = dofs[4: 7]
    axis_angle[0, 5, 0] = dofs[7]
    axis_angle[0, 7] = dofs[8: 11]
    axis_angle[0, 9] = dofs[11: 14]
    axis_angle[0, 11] = dofs[14: 17]
    axis_angle[0, 12, 1] = dofs[17]
    axis_angle[0, 14] = dofs[18: 21]
    axis_angle[0, 15, 1] = dofs[21]
    quaternions = Quaternions.from_angle_axis(np.sqrt(np.sum(axis_angle ** 2, axis=-1)), axis_angle)
    # y-up, z-forward, x-right
    offsets = input_tpose_j3d - input_tpose_j3d[j17_parents]
    offsets[0] = j3d[0] * 100   # m -> cm
    positions = offsets[np.newaxis].repeat(frame_num, axis=0)  # (frames, jointsNum, 3)
    orients = Quaternions.id(0)
    for i in range(offsets.shape[0]):
        orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
    anim = Animation.Animation(quaternions, positions, orients, offsets, j17_parents)
    j3d_pre = Animation.positions_global(anim) / 100    # cm -> m
    j2d_pre = np.dot(np.divide(j3d_pre, j3d_pre[:, :, 2:], where=j3d_pre[:, :, 2:] != 0), cam.T)[:, :, :2]
    return j3d_pre[0], j2d_pre[0]


def optimize_root_rot(root_rot):
    global j3d, ppre_root_rot, pre_root_rot
    gt_tri = (j3d[[1, 4, 7]] - j3d[0]) * 100    # m -> cm
    tpose_tri = input_tpose_j3d[[1, 4, 7]]
    # root_rot_m = Quaternions.from_angle_axis(np.linalg.norm(root_rot), root_rot).transforms()[0]
    root_rot_m = cv2.Rodrigues(root_rot)[0]
    pre_tri = np.dot(tpose_tri, root_rot_m.T)   # root_m · tpose_tri.T = gt_tri.T   根据FK的过程应估计列向量右乘时的旋转矩阵
    error = []
    for i in range(pre_tri.shape[0]):
        error.append(np.mean((gt_tri[i] - pre_tri[i]) ** 2) * w_3d)
    # 对根节点使用 use_temp 误差会提高不到1mm 所以暂且不用
    # if use_temp:
    #     if ppre_root_rot.shape[0] != 1 and pre_root_rot.shape[0] != 1:
    #         e_temp = np.sqrt(((pre_root_rot - ppre_root_rot - (root_rot - pre_root_rot)) ** 2).mean()) * w_temp
    #         error.append(e_temp)
    # 根节点的旋转自由度不需要范围限制
    # for i in range(root_rot.shape[0]):
    #     if root_rot[i] > np.pi:
    #         error.append((root_rot[i] - np.pi) ** 2 * w_lim)
    #     elif root_rot[i] < -np.pi:
    #         error.append((root_rot[i] + np.pi) ** 2 * w_lim)
    #     else:
    #         error.append(0.)
    return error

# 最小二乘法估计全局旋转虽然很准确，但是得到的旋转矩阵不是正交矩阵，因此变换为轴角再变换回矩阵造成的误差非常大
# def optimize_root_rot(j3d):
#     gt_tri = (j3d[[1, 4, 7]] - j3d[0]) * 100    # m -> cm
#     global input_tpose_j3d
#     tpose_tri = input_tpose_j3d[[1, 4, 7]]
#     root_rotm = np.linalg.lstsq(tpose_tri, gt_tri, rcond=-1)[0].T
#     return cv2.Rodrigues(root_rotm)[0].reshape((-1))


def optimize(dofs):
    global j3d, j2d, cam, ppre_dof, pre_dof
    j3d_pre, j2d_pre = compute_joints_from_dofs(dofs, cam)   # (17, 3/2)
    e3d = w_3d * np.mean((j3d - j3d_pre) ** 2, axis=-1)
    # j_cal_cam = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16])
    e2d = w_2d * np.mean(j2d[:, 2:] * (j2d[:, :2] - j2d_pre[:, :2]) ** 2, axis=-1)
    error = []
    for j in range(e3d.shape[0]):
        error.append(e3d[j])
    for j in range(e2d.shape[0]):
        error.append(e2d[j])
    # e_limit
    if use_lim:
        for d in range(dofs_limit.shape[0]):
            if dofs[d] < dofs_limit[d, 0]:
                error.append((dofs[d] - dofs_limit[d, 0]) ** 2 * w_lim)
            elif dofs[d] > dofs_limit[d, 1]:
                error.append((dofs[d] - dofs_limit[d, 1]) ** 2 * w_lim)
            else:
                error.append(0.)
    # e_temp
    if use_temp:
        if ppre_dof.shape[0] != 1 and pre_dof.shape[0] != 1:
            e_temp = np.sqrt(((pre_dof - ppre_dof - (dofs - pre_dof)) ** 2).mean()) * w_temp
            error.append(e_temp)
    return error


def begin():
    global j3d, j2d, cam, ppre_dof, pre_dof, root_rot, ppre_root_rot, pre_root_rot
    ppre_dof, pre_dof = np.zeros((1, 1)), np.zeros((1, 1))
    ppre_root_rot, pre_root_rot = np.zeros((1, 1)), np.zeros((1, 1))
    j3ds, j2ds, cam = read_joints_from_h36m()      # (F, 17, 3/3)
    frame_num = j3ds.shape[0]
    dofs = np.zeros((frame_num, 28), dtype=float)
    dofs[:, :3] = j3ds[:, 0]

    time_str = datetime.now().strftime("%m_%d_%H_%M")
    subject_name = data_path.split('/')[-2]
    if 'all' in data_path:
        subject_name = 'all_' + subject_name
    save_dir = './out/' + time_str + '_' + subject_name + '_3d_' + str(w_3d) + '+2d_' + str(w_2d)
    if use_lim:
        save_dir += '+lim_' + str(w_lim)
    if use_temp:
        save_dir += '+temp_' + str(w_temp)
    if use_filter:
        save_dir += '+filter'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('save path: ' + save_dir)
    config_filter = {
        'freq': 10,
        'mincutoff': 20.0,
        'beta': 0.4,
        'dcutoff': 1.0
    }
    if 'all' in data_path:
        config_filter['freq'] = 50
    # 有3个自由度的节点有6个，有1个自由度的节点有4个
    filter_dof3 = [(OneEuroFilter(**config_filter), OneEuroFilter(**config_filter),
                    OneEuroFilter(**config_filter), OneEuroFilter(**config_filter)) for _ in range(6)]
    filter_dof1 = [OneEuroFilter(**config_filter) for _ in range(4)]

    start_t = time.time()
    mpjpe_all = []
    threshold = 100.
    num = 0
    # dofs = np.loadtxt('./out/09_13_11_54_WalkingDog-2_3d_1+2d_1e-05+lim_0.1+te mp_0.1_49.63mm_1/09_13_11_54_dofs.txt')
    # j3d, j2d = j3ds[0], j2ds[0]
    # root_rot = dofs[0, 3:6]
    # j3d_pre, j2d_pre = compute_joints_from_dofs(dofs[0, 6:], cam)
    # print(j3d * 100)
    # print(root_rot)
    for f in range(frame_num):
        print('-------------------------------------')
        j3d, j2d = j3ds[f], j2ds[f]
        # 用上一帧的结果做初始化可以极大缩短迭代次数
        if f > 0:
            dofs[f, 3:] = dofs[f - 1, 3:]
        # predict root rot
        sol = root(optimize_root_rot, dofs[f, 3:6], method='lm')
        root_rot = sol.x
        dofs[f, 3:6] = root_rot

        sol = root(optimize, dofs[f, 6:], method='lm')
        dofs[f, 6:] = sol.x
        if f == 0:
            ppre_dof = dofs[f, 6:]
            ppre_root_rot = root_rot
        elif f == 1:
            pre_dof = dofs[f, 6:]
            pre_root_rot = root_rot
        else:
            ppre_dof = pre_dof
            pre_dof = dofs[f, 6:]
            ppre_root_rot = pre_root_rot
            pre_root_rot = root_rot

        # oneEuroFilter
        if use_filter:
            # 对根节点的旋转不做滤波，防止unity读取计算问题导致个别帧朝向错误
            for i, j in enumerate([6, 10, 14, 17, 20, 14]):
                dof_j = dofs[f, j: j + 3]
                theta = np.linalg.norm(dof_j)
                axis = np.divide(dof_j, theta, where=theta != 0)
                axis[0] = filter_dof3[i][0](axis[0])
                axis[1] = filter_dof3[i][1](axis[1])
                axis[2] = filter_dof3[i][2](axis[2])
                theta = filter_dof3[i][3](theta)
                dofs[f, j: j + 3] = axis * theta
            for i, j in enumerate([9, 13, 12, 15]):
                dofs[f, j] = filter_dof1[i](dofs[f, j])

        # print(sol.x)
        # print(sol.success)
        # print(sol.nfev)
        # print(sol.message)
        # print(sol.fun)
        j3d_pre, j2d_pre = compute_joints_from_dofs(dofs[f, 6:], cam)
        mpjpe = np.mean(np.linalg.norm(j3d * 1000 - j3d_pre * 1000, axis=-1))
        if mpjpe > threshold:
            num += 1
        mpjpe_all.append(mpjpe)
        print(('%04d' % f) + '-MPJPE: ' + ('%.2f' % mpjpe) + ' mm')
        fd = open(save_dir + '/logs.txt', 'a+')
        fd.write(('%04d' % f) + '-MPJPE: ' + ('%.2f' % mpjpe) + ' mm\n')
        fd.close()
        plot_2skeleton(j3d * 100, j3d_pre * 100, f, mpjpe, save_dir)

    end_t = time.time()
    fd = open(save_dir + '/logs.txt', 'a+')
    fd.write('MPJPE-mean: ' + ('%.2f' % np.mean(mpjpe_all)) + ' mm\n')
    fd.close()
    np.savetxt(save_dir + '/' + time_str + '_' + subject_name + '_dofs.txt', dofs, fmt='%1.6f')
    time_per = (end_t - start_t) / frame_num
    print('time every frame: ' + str(time_per))
    print(np.mean(mpjpe_all))

    frame_to_video(save_dir + '/3d_skeleton')
    os.rename(save_dir, save_dir + ('_%.2fmm_' % np.mean(mpjpe_all)) + str(num) + ('_%.2fs' % time_per))


def main():
    begin()
    # for w_3d in [10, 1, 0.1]:
    #     for w_2d in [1, 1e-1, 1e-3, 1e-5]:
    #         for w_lim in [10, 1, 0.1]:
    #             for w_temp in [1e-1, 1e-3, 1e-5, 1e-7]:
    #                 begin()


if __name__ == '__main__':
    main()

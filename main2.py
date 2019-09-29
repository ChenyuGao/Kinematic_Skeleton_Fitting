from visualization import plot_2skeleton, frame_to_video
from dataloader import read_joints_from_h36m, read_joints_from_eval
from OneEuroFilter import OneEuroFilter
from config import *
from utils import *
from Quaternions import Quaternions

from scipy.optimize import root
import os
from datetime import datetime
import time
import cv2
from numba import jit, cuda
import numpy as np
np.set_printoptions(suppress=True)


# @jit(nopython=True)
def compute_joints_from_dofs(dofs, cam):
    global j3d, root_rot
    axis_angle = np.zeros((17, 3))
    axis_angle[0] = root_rot
    axis_angle[1] = dofs[0: 3]
    axis_angle[2, 0] = dofs[3]
    axis_angle[4] = dofs[4: 7]
    axis_angle[5, 0] = dofs[7]
    axis_angle[7] = dofs[8: 11]
    axis_angle[9] = dofs[11: 14]
    axis_angle[11] = dofs[14: 17]
    axis_angle[12, 1] = dofs[17]
    axis_angle[14] = dofs[18: 21]
    axis_angle[15, 1] = dofs[21]
    offsets = input_tpose_j3d - input_tpose_j3d[j17_parents]
    offsets[0] = j3d[0] * 100  # m -> cm
    local_rot_mat = np.zeros((17, 4, 4))
    local_rot_mat[:, 0:3, 3] = offsets
    local_rot_mat[:, 3:4, 3] = 1.0
    global_rot_mat = np.zeros((17, 4, 4))
    for i in range(local_rot_mat.shape[0]):
        local_rot_mat[i, 0:3, 0:3] = cv2.Rodrigues(axis_angle[i])[0]
    # local_rot_mat[:, 0:3, 0:3] = axis_angles_to_matrixs(axis_angle)
    global_rot_mat[0] = local_rot_mat[0]
    for i in range(1, axis_angle.shape[0]):
        global_rot_mat[i, :] = np.dot(global_rot_mat[j17_parents[i]], local_rot_mat[i])
    j3d_pre = global_rot_mat[:, 0:3, 3] / 100    # cm -> m
    j2d_pre = np.dot(np.divide(j3d_pre, j3d_pre[:, 2:], where=j3d_pre[:, 2:] != 0), cam.T)[:, :2]
    return j3d_pre, j2d_pre


def optimize_root_rot(root_rot):
    global j3d
    gt_tri = (j3d[[1, 4, 7]] - j3d[0]) * 100    # m -> cm
    tpose_tri = input_tpose_j3d[[1, 4, 7]]
    root_rot_m = cv2.Rodrigues(root_rot)[0]
    pre_tri = np.dot(tpose_tri, root_rot_m.T)   # root_m · tpose_tri.T = gt_tri.T
    error = []
    for i in range(pre_tri.shape[0]):
        error.append(np.mean((gt_tri[i] - pre_tri[i]) ** 2) * w_3d)
    return error


def optimize(dofs):
    global j3d, j2d, cam, ppre_dof, pre_dof
    j3d_pre, j2d_pre = compute_joints_from_dofs(dofs, cam)   # (17, 3/3)
    e3d = w_3d * np.mean((j3d - j3d_pre) ** 2, axis=-1)
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


def main():
    global j3d, j2d, cam, ppre_dof, pre_dof, root_rot, ppre_root_rot, pre_root_rot
    ppre_dof, pre_dof = np.zeros((1, 1)), np.zeros((1, 1))
    ppre_root_rot, pre_root_rot = np.zeros((1, 1)), np.zeros((1, 1))
    print("Begin load data...")
    if use_gt:
        j3ds, j2ds, cam = read_joints_from_h36m()      # (F, 17, 3/3)
    else:
        j3ds, j2ds, cam, gt_j3ds = read_joints_from_eval()      # (F, 17, 3/3)
    frame_num = j3ds.shape[0]
    dofs = np.zeros((frame_num, 28), dtype=float)
    dofs[:, :3] = j3ds[:, 0]

    time_str = datetime.now().strftime("%m_%d_%H_%M")
    subject_name = data_path.split('/')[-2]
    if 'all' in data_path:
        subject_name = 'all_' + subject_name
    if not use_gt:
        subject_name = 'all_pre_' + subject_name
    prefix = time_str + '_'
    if isXiaoice:
        prefix += 'Xiaoice_'
    elif isAijiang:
        prefix += 'Aijiang_'
    save_dir = './out/main2_' + prefix + subject_name + '_3d_' + str(w_3d) + '+2d_' + str(w_2d)
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
        # predict all rot
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

        j3d_pre, j2d_pre = compute_joints_from_dofs(dofs[f, 6:], cam)
        if use_gt:
            mpjpe = np.mean(np.linalg.norm(j3d * 1000 - j3d_pre * 1000, axis=-1))
        else:
            mpjpe = np.mean(np.linalg.norm(gt_j3ds[f] * 1000 - j3d_pre * 1000, axis=-1))
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
    np.savetxt(save_dir + '/' + prefix + subject_name + '_dofs.txt', dofs, fmt='%1.6f')
    dofs_new = np.zeros((dofs.shape[0], 18, 3))
    dofs_new[:, 0] = dofs[:, 0: 3]
    dofs_new[:, 1] = dofs[:, 3: 6]
    dofs_new[:, 2] = dofs[:, 6: 9]
    dofs_new[:, 3, 0] = dofs[:, 9]
    dofs_new[:, 5] = dofs[:, 10: 13]
    dofs_new[:, 6, 0] = dofs[:, 13]
    dofs_new[:, 8] = dofs[:, 14: 17]
    dofs_new[:, 10] = dofs[:, 17: 20]
    dofs_new[:, 12] = dofs[:, 20: 23]
    dofs_new[:, 13, 1] = dofs[:, 23]
    dofs_new[:, 15] = dofs[:, 24: 27]
    dofs_new[:, 16, 1] = dofs[:, 27]
    np.savetxt(save_dir + '/' + prefix + subject_name + '_dofs_new.txt', dofs_new.reshape((frame_num, -1)), fmt='%1.6f')
    dofs_world = dofs_local2world(dofs)  # (f, 18, 3)
    np.savetxt(save_dir + '/' + prefix + subject_name + '_dofs_world.txt', dofs_world.reshape((frame_num, -1)),
               fmt='%1.6f')
    time_per = (end_t - start_t) / frame_num
    print('time every frame: ' + str(time_per))
    print(np.mean(mpjpe_all))
    frame_to_video(save_dir + '/3d_skeleton')
    os.rename(save_dir, save_dir + ('_%.2fmm_' % np.mean(mpjpe_all)) + str(num) + ('_%.2fs' % time_per))

    # dofs = np.loadtxt('./out/main2_09_29_17_19_WalkingTogether-1_3d_1+2d_1e-05+lim_0.1+temp_0.1+filter_49.60mm_0_1.26s/09_29_17_19_WalkingTogether-1_dofs.txt')
    # np.savetxt('./out/main2_09_29_17_19_WalkingTogether-1_3d_1+2d_1e-05+lim_0.1+temp_0.1+filter_49.60mm_0_1.26s/09_29_17_19_WalkingTogether-1_dofs_world.txt',
    #            dofs_world.reshape((dofs.shape[0], -1)), fmt='%1.6f')


if __name__ == '__main__':
    main()


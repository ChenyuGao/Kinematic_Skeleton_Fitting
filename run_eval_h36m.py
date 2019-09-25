from config import *
from utils import *
from visualization import plot_2skeleton, frame_to_video
from OneEuroFilter import OneEuroFilter

import itertools
from scipy.optimize import root
import os
from datetime import datetime
import time
import cv2
import deepdish as dd
from numba import jit
import numpy as np
from tqdm import tqdm
np.set_printoptions(suppress=True)


def get_h36m_seqs(protocol=2):
    action_names = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing',
        'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
        'Waiting', 'Walking', 'WalkingDog', 'WalkingTogether'
    ]
    print('Protocol %d!!' % protocol)
    if protocol == 2:
        trial_ids = [1]
        cam_ids = [3]
    else:
        trial_ids = [1, 2]
        cam_ids = range(0, 4)

    sub_ids = [9, 11]
    all_pairs = [p for p in list(itertools.product(*[sub_ids, action_names, trial_ids, cam_ids]))]

    return all_pairs, action_names


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
    j17_parents_arr = np.array(j17_parents)
    offsets = input_tpose_j3d - input_tpose_j3d[j17_parents_arr]
    offsets[0] = j3d[0] * 100  # m -> cm
    local_rot_mat = np.zeros((17, 4, 4))
    local_rot_mat[:, 0:3, 3] = offsets
    local_rot_mat[:, 3:4, 3] = 1.0
    global_rot_mat = np.zeros((17, 4, 4))
    for i in range(local_rot_mat.shape[0]):
        local_rot_mat[i, 0:3, 0:3] = cv2.Rodrigues(axis_angle[i])[0]
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


def evaluate_sequence(all_data, seq_info, pred_dir):
    global j3d, j2d, cam, ppre_dof, pre_dof, root_rot, ppre_root_rot, pre_root_rot
    ppre_dof, pre_dof = np.zeros((1, 1)), np.zeros((1, 1))
    ppre_root_rot, pre_root_rot = np.zeros((1, 1)), np.zeros((1, 1))

    sub_id, action, trial_id, cam_id = seq_info
    file_seq_name = 'S%d_%s-%d_cam%01d' % (sub_id, action, trial_id, cam_id)
    print('%s' % (file_seq_name))
    save_path = pred_dir + file_seq_name + '_pred.h5'
    if os.path.exists(save_path):
        results = dd.io.load(save_path)
        errors = results['errors']
        errors_pa = results['errors_pa']
        errors_n = results['errors_n']
        errors_vel = results['errors_vel']
        return errors, errors_pa, errors_n, errors_vel

    sub_id = 'S' + str(sub_id)
    action = action + '-' + str(trial_id)
    pose_3d = all_data['positions_3d_pred'].item()
    gt_3d = all_data['positions_3d_gt'].item()
    pose_2d = all_data['positions_2d'].item()
    j3ds = pose_3d[sub_id][action][cam_id]
    gt3ds = gt_3d[sub_id][action][cam_id]
    j2ds = pose_2d[sub_id][action][cam_id]
    index_f19_t17 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # j3d: m   j2d: [-1, 1]
    j3ds, j2ds, gt3ds = j3ds[:, index_f19_t17], j2ds[:, index_f19_t17], gt3ds[:, index_f19_t17]
    j3ds = h36m_skeleton_fit(j3ds)
    j3ds[:, 0] = gt3ds[:, 0]
    j_cal_cam = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16])
    cam = infer_camera_intrinsics(j2ds[:, j_cal_cam, :2], j3ds[:, j_cal_cam])
    j2ds[:, :, :2] = np.dot(j3ds / j3ds[:, :, 2:], cam.T)[:, :, :2]
    frame_num = j3ds.shape[0]
    dofs = np.zeros((frame_num, 28), dtype=float)
    dofs[:, :3] = j3ds[:, 0]
    config_filter = {
        'freq': 10,
        'mincutoff': 20.0,
        'beta': 0.4,
        'dcutoff': 1.0
    }
    # 有3个自由度的节点有6个，有1个自由度的节点有4个
    filter_dof3 = [(OneEuroFilter(**config_filter), OneEuroFilter(**config_filter),
                    OneEuroFilter(**config_filter), OneEuroFilter(**config_filter)) for _ in range(6)]
    filter_dof1 = [OneEuroFilter(**config_filter) for _ in range(4)]

    j3d_pre = np.zeros_like(j3ds)
    results = {}
    for f in tqdm(range(frame_num)):
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

        j3d_pre[f], _ = compute_joints_from_dofs(dofs[f, 6:], cam)

    errors, errors_pa = compute_errors(gt3ds * 1000., j3d_pre * 1000.)
    errors_n = n_mpjpe(gt3ds * 1000., j3d_pre * 1000.)
    errors_vel = mean_velocity_error(gt3ds * 1000., j3d_pre * 1000.)
    results['errors'] = errors
    results['errors_pa'] = errors_pa
    results['errors_n'] = errors_n
    results['errors_vel'] = errors_vel
    # Save results
    dd.io.save(save_path, results)
    np.savetxt(pred_dir + 'dofs/' + file_seq_name + '.txt', dofs, fmt='%1.6f')
    return errors, errors_pa, errors_n, errors_vel


def eval(pred_dir, protocol=2):
    all_pairs, actions = get_h36m_seqs(protocol=protocol)
    all_errors = {}
    all_errors_pa = {}
    all_errors_n = {}
    all_errors_vel = {}
    raw_errors, raw_errors_pa, raw_errors_n, raw_errors_vel = [], [], [], []
    all_time, all_image_num = 0, 0
    all_data = np.load(all_eval_path, allow_pickle=True)
    for itr, seq_info in enumerate(all_pairs):
        print('%d/%d' % (itr, len(all_pairs)))
        t0 = time.time()
        sub_id, action, trial_id, cam_id = seq_info
        errors, errors_pa, errors_n, errors_vel = evaluate_sequence(all_data, seq_info, pred_dir)
        mean_error = np.mean(errors)
        mean_error_pa = np.mean(errors_pa)
        mean_error_n = np.mean(errors_n)
        mean_errors_vel = np.mean(errors_vel)
        # med_error = np.median(errors)
        raw_errors.append(errors)
        raw_errors_pa.append(errors_pa)
        raw_errors_n.append(errors_n)
        raw_errors_vel.append(errors_vel)
        print('====================')
        print('mean error: %g, PA mean: %g' % (mean_error, mean_error_pa))
        raws = np.hstack(raw_errors)
        raws_pa = np.hstack(raw_errors_pa)
        raws_n = np.hstack(raw_errors_n)
        raws_vel = np.hstack(raw_errors_vel)
        print('Running average - mean: %g, median: %g' % (np.mean(raws), np.median(raws)))
        print('Running average - PA mean: %g, median: %g' % (np.mean(raws_pa), np.median(raws_pa)))
        print('Running average - N mean: %g, median: %g' % (np.mean(raws_n), np.median(raws_n)))
        print('Running average - Vel mean: %g, median: %g' % (np.mean(raws_vel), np.median(raws_vel)))
        print('====================')
        if action in all_errors.keys():
            all_errors[action].append(mean_error)
            all_errors_pa[action].append(mean_error_pa)
            all_errors_n[action].append(mean_error_n)
            all_errors_vel[action].append(mean_errors_vel)
        else:
            all_errors[action] = [mean_error]
            all_errors_pa[action] = [mean_error_pa]
            all_errors_n[action] = [mean_error_n]
            all_errors_vel[action] = [mean_errors_vel]
        t1 = time.time()
        print('Took %g sec for %d imgs' % (t1 - t0, len(errors)))
        all_time += (t1 - t0)
        all_image_num += len(errors)

    all_act_errors = []
    all_act_errors_pa = []
    all_act_errors_n = []
    all_act_errors_vel = []
    for act in actions:
        print('%s mean error %g, PA error %g, N error %g,Velocity error %g' % (
            act, np.mean(all_errors[act]), np.mean(all_errors_pa[act]), np.mean(all_errors_n[act]),
            np.mean(all_errors_vel[act])))
        all_act_errors.append(np.mean(all_errors[act]))
        all_act_errors_pa.append(np.mean(all_errors_pa[act]))
        all_act_errors_n.append(np.mean(all_errors_n[act]))
        all_act_errors_vel.append(np.mean(all_errors_vel[act]))

    print('Average error over all seq (over action) 3d: %g, PA: %g, N: %g,Velocity: %g' %
          (np.mean(all_act_errors), np.mean(all_act_errors_pa), np.mean(all_act_errors_n), np.mean(all_act_errors_vel)))

    act_names_in_order = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
        'TakingPhoto', 'Posing', 'Purchases', 'Sitting', 'SittingDown',
        'Smoking', 'Waiting', 'WalkingDog', 'Walking', 'WalkingTogether'
    ]
    act_error = [
        '%.2f' % np.mean(all_errors[act]) for act in act_names_in_order
    ]
    act_PA_error = [
        '%.2f' % np.mean(all_errors_pa[act]) for act in act_names_in_order
    ]
    act_N_error = [
        '%.2f' % np.mean(all_errors_n[act]) for act in act_names_in_order
    ]
    act_Vel_error = [
        '%.2f' % np.mean(all_errors_vel[act]) for act in act_names_in_order
    ]

    act_names_in_order.append('Average')
    act_error.append('%.2f' % np.mean(all_act_errors))
    act_PA_error.append('%.2f' % np.mean(all_act_errors_pa))
    act_N_error.append('%.2f' % np.mean(all_act_errors_n))
    act_Vel_error.append('%.2f' % np.mean(all_act_errors_vel))
    print('---for excel---')
    print(', '.join(act_names_in_order))
    print(', '.join(act_error))
    print('PA-MPJPE:')
    print(', '.join(act_PA_error))
    print('N-MPJPE:')
    print(', '.join(act_N_error))
    print('Velocity Error (MPJVE):')
    print(', '.join(act_Vel_error))

    MPJPE = np.mean(np.hstack(raw_errors))
    PA_MPJPE = np.mean(np.hstack(raw_errors_pa))
    N_MPJPE = np.mean(np.hstack(raw_errors_n))
    MPJVE = np.mean(np.hstack(raw_errors_vel))
    print('Average error over all joints 3d: %g, PA: %g, N: %g, Vel: %g' % (MPJPE, PA_MPJPE, N_MPJPE, MPJVE))

    err = np.hstack(raw_errors)
    median = np.median(np.hstack(raw_errors))
    pa_median = np.median(np.hstack(raw_errors_pa))
    n_median = np.median(np.hstack(raw_errors_n))
    vel_median = np.median(np.hstack(raw_errors_vel))
    print(
        'Percentiles 90th: %.1f 70th: %.1f 50th: %.1f 30th: %.1f 10th: %.1f' %
        (np.percentile(err, 90), np.percentile(err, 70),
         np.percentile(err, 50), np.percentile(err, 30),
         np.percentile(err, 10)))

    print(
        'MPJPE: %.2f, PA-MPJPE: %.2f, N-MPJPE: %.2f, MPJVE: %.2f, Median: %.2f, PA-Median: %.2f, N-Median: %.2f, Vel-Median: %.2f' %
        (MPJPE, PA_MPJPE, N_MPJPE, MPJVE, median, pa_median, n_median, vel_median))
    print('%.2f frames every second' % (all_image_num / all_time))


if __name__ == '__main__':
    protocol = 2
    pred_dir = './out/eval/main2_P' + str(protocol) + '/'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
        os.makedirs(pred_dir + 'dofs/')
    eval(pred_dir, protocol=protocol)

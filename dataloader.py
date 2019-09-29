import h5py

from config import *
from utils import h36m_skeleton_fit, infer_camera_intrinsics


# read data from gt
def read_joints_from_h36m(annot_dir=data_path):
    cam_id = ['54138969', '55011271', '58860488', '60457274']
    annot_file_path = annot_dir + 'annot-' + cam_id[3] + '.h5'
    with h5py.File(annot_file_path, 'r') as annot:
        j2d = np.array(annot['joints2D'])
        j3d = np.array(annot['joints3D-univ'])      # (f, 32, 3)
    h36m_to_17_index = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    j3d, j2d = j3d[:, h36m_to_17_index], j2d[:, h36m_to_17_index]
    j2d = j2d / 1000 * 2 - 1.
    j3d = j3d / 1000    # mm -> m
    # input_tpose_j3d = mixamo_skeleton_fit(j3d[0]) * 100
    j3d = h36m_skeleton_fit(j3d)
    j_cal_cam = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16])
    cam = infer_camera_intrinsics(j2d[:, j_cal_cam], j3d[:, j_cal_cam])
    j2d = np.dot(j3d / j3d[:, :, 2:], cam.T)[:, :, :2]
    confidence = np.ones((j2d.shape[0], j2d.shape[1], 1))
    j2d = np.concatenate([j2d, confidence], axis=2)
    return j3d, j2d, cam


# read data from prediction by Shuo
def read_joints_from_eval(eval_path=all_eval_path):
    all_data = np.load(eval_path, allow_pickle=True)
    pose_3d = all_data['positions_3d_pred'].item()
    gt_3d = all_data['positions_3d_gt'].item()
    pose_2d = all_data['positions_2d'].item()
    # for subject in pose_3d:
    #     for action in pose_3d[subject]:
    #         for i in range(4):
    #             j3d = pose_3d[subject][action][i]
    #             j2d = pose_2d[subject][action][i]
    subject = data_path.split('/')[-3]
    action = data_path.split('/')[-2]
    j3d = pose_3d[subject][action][3]
    gt3d = gt_3d[subject][action][3]
    j2d = pose_2d[subject][action][3]
    index_f19_t17 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    j3d, j2d, gt3d = j3d[:, index_f19_t17], j2d[:, index_f19_t17], gt3d[:, index_f19_t17]   # j3d: m   j2d: [-1, 1]
    j3d = h36m_skeleton_fit(j3d)
    j_cal_cam = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16])
    cam = infer_camera_intrinsics(j2d[:, j_cal_cam, :2], j3d[:, j_cal_cam])
    j2d[:, :, :2] = np.dot(j3d / j3d[:, :, 2:], cam.T)[:, :, :2]
    return j3d, j2d, cam, gt3d


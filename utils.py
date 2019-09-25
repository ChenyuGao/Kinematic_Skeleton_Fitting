import numpy as np
import h5py

from config import *


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


# not use
def mixamo_skeleton_fit(j3d_h36m):
    j17_ori = input_tpose_j3d / 100
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
    if mixamo:
        head = j3d_h36m[:, 10] - j3d_h36m[:, 9]
        neck = j3d_h36m[:, 9] - j3d_h36m[:, 8]
        j3d_h36m[:, 8] = j3d_h36m[:, 8] - (j3d_h36m[:, 8] - j3d_h36m[:, 7]) / 3
        j3d_h36m[:, 9] = j3d_h36m[:, 8] + neck
        j3d_h36m[:, 10] = j3d_h36m[:, 9] + head
        j3d_h36m[:, 0] = ((j3d_h36m[:, 1] + j3d_h36m[:, 4] + j3d_h36m[:, 7]) / 3 - j3d_h36m[:, 0]) / 2 + j3d_h36m[:, 0]
    else:
        head = j3d_h36m[:, 10] - j3d_h36m[:, 9]
        neck = j3d_h36m[:, 9] - j3d_h36m[:, 8]
        j3d_h36m[:, 0] = j3d_h36m[:, 7] - (j3d_h36m[:, 7] - j3d_h36m[:, 0]) / 10
        j3d_h36m[:, 8] = j3d_h36m[:, 7] + (j3d_h36m[:, 8] - j3d_h36m[:, 7]) / 3
        j3d_h36m[:, 9] = j3d_h36m[:, 8] + neck
        j3d_h36m[:, 10] = j3d_h36m[:, 9] + head

    j17_ori = input_tpose_j3d / 100     # cm -> m
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


# ----------------------------------------------------------------------------------------------------------------------
# eval
def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def align_by_pelvis(joints, get_pelvis=False):
    """
    Assumes joints is 17 x 3 in LSP order.
    Then root are: [0]
    Takes mid point of these points, then subtracts it.
    """
    if joints.shape[0] == 14:
        left_id = 3
        right_id = 2

        pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
        if get_pelvis:
            return joints - np.expand_dims(pelvis, axis=0), pelvis
        else:
            return joints - np.expand_dims(pelvis, axis=0)
    else:
        return joints - np.expand_dims(joints[0, :], axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 or 16 common joints.
    Inputs:
      - gt3ds: N x 17 x 3
      - preds: N x 17 x 3
    """
    errors, errors_pa = [], []
    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds)):

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa


def n_mpjpe(gt3ds, preds):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    errors = []
    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds)):
        norm_pred3d = np.mean(np.sum(pred3d ** 2, axis=1))
        norm_gt3d = np.mean(np.sum(gt3d ** 2, axis=1))
        scale = norm_gt3d / norm_pred3d
        joint_error = np.sqrt(np.sum((gt3d - scale * pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))
    return errors


def mean_velocity_error(gt3ds, preds):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert preds.shape == gt3ds.shape

    velocity_predicted = np.diff(preds, axis=0)
    velocity_target = np.diff(gt3ds, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(gt3ds.shape) - 1))


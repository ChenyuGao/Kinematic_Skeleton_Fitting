import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from config import j17_parents


def plot_keypoints3d(keypoints3d, line, ax, num):
    human_value = keypoints3d.astype('int')
    for i in range(num):
        ax.scatter(human_value[i][0], human_value[i][1], human_value[i][2], c='red')
        ax.text(human_value[i][0]+1, human_value[i][1]+1, human_value[i][2]+1, str(i),
                color='red', size=10, backgroundcolor="none")
    for j in line:
        ax.plot([human_value[j[0]][0], human_value[j[1]][0]],
            [human_value[j[0]][1], human_value[j[1]][1]],
            [human_value[j[0]][2], human_value[j[1]][2]], color='g')
    return ax


def plot_2skeleton(j3d1, j3d2, frame=0, mpjpe=0, save_dir=None):
    line = [[i, j] for i, j in enumerate(j17_parents)][1:]
    j3d1 = j3d1 - j3d1[0]
    j3d2 = j3d2 - j3d2[0]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d')
    j = j3d1 - j3d1[0]
    ax = plot_keypoints3d(j, line, ax, j3d1.shape[0])

    ax.set_title('gt')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.view_init(elev=-90, azim=-90)  # elev俯仰角度  azim是旋转角度
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)

    ax = fig.add_subplot(122, projection='3d')
    j = j3d2 - j3d2[0]
    ax = plot_keypoints3d(j, line, ax, j3d2.shape[0])

    ax.set_title('pre')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.view_init(elev=-90, azim=-90)  # elev俯仰角度  azim是旋转角度
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    title = ('%04d' % frame) + '-MPJPE: ' + ('%.2f' % mpjpe) + ' mm'
    plt.suptitle(title)
    if save_dir:
        save_dir += '/3d_skeleton'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + '/' + ('%04d' % frame) + '.png', dpi=100)
    # plt.show()
    plt.close('all')


def frame_to_video(save_dir):
    print('To Video...')
    fps = 10
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    imgs_name = os.listdir(save_dir)
    imgs_path = sorted([save_dir + '/' + img_name for img_name in imgs_name])
    image = cv2.imread(imgs_path[0])
    videoWriter = cv2.VideoWriter(save_dir + '/../' + save_dir.split('/')[-1] + '.avi', fourcc, fps,
                                 (image.shape[1], image.shape[0]))
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        videoWriter.write(img)
    videoWriter.release()

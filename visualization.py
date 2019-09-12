import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


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
    j17_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
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

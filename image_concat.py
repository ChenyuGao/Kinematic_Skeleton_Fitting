import os
from PIL import Image
import cv2

images_dir = 'E:/Datasets/Human3.6m/processed/S11/WalkingDog-2/imageSequence-54138969'
images_unity_dir = 'E:/Projects/Kinematic_Skeleton_Fitting/out/09_02_14_09/unity_frames'
skeleton_dir = 'E:/Projects/Kinematic_Skeleton_Fitting/out/09_02_14_09/3d_skeleton'
save_dir = 'E:/Projects/Kinematic_Skeleton_Fitting/out/09_02_14_09/image_unity_all'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
images_path = [os.path.join(images_dir, i) for i in sorted(os.listdir(images_dir)) if 'jpg' in i]
images_unity_path = [os.path.join(images_unity_dir, i) for i in sorted(os.listdir(images_unity_dir))][:len(images_path) + 1]
skeleton_path = [os.path.join(skeleton_dir, i) for i in sorted(os.listdir(skeleton_dir)) if 'png' in i]

# assert len(images_render_path) == len(images_unity_path)

width = 1000
height = 1000

for i in range(len(images_path)):
    image = Image.open(images_path[i])
    image = image.resize((width // 2, height // 2))
    image_unity = Image.open(images_unity_path[i])
    image_unity = image_unity.resize((width // 2, height // 2))
    skeleton_img = Image.open(skeleton_path[i])
    skeleton_img = skeleton_img.resize((width, height // 2))
    toImage = Image.new('RGBA', (width, height))
    toImage.paste(image, (0, 0))
    toImage.paste(image_unity, (width // 2, 0))
    toImage.paste(skeleton_img, (0, height // 2))
    image_name = 'image_%04d' % i + '.png'
    toImage.save(os.path.join(save_dir, image_name))

print('To Video...')
fps = 10
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
imgs_name = os.listdir(save_dir)
imgs_path = sorted([os.path.join(save_dir, img_name) for img_name in imgs_name])
image = cv2.imread(imgs_path[0])
videoWriter = cv2.VideoWriter(os.path.join(save_dir, '../', save_dir[save_dir.rfind('/') + 1:] + '.avi'), fourcc,
                              fps, (image.shape[1], image.shape[0]))
for img_path in imgs_path:
    img = cv2.imread(img_path)
    videoWriter.write(img)
videoWriter.release()

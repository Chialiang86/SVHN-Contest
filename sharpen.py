import cv2
import argparse
import numpy as np
import os
import glob


def main(args):
    joint_path = os.path.join(args.root, args.dir)
    img_paths = glob.glob('{}/*.png'.format(joint_path))

    target_dir = os.path.join(args.root, args.target)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print('{} created.'.format(target_dir))

    print('sharpen images saving to {}'.format(target_dir))
    cnt = 0
    total = len(img_paths)
    for img_path in img_paths:

        cnt += 1
        if cnt % 100 == 0:
            print('processing {}/{}'.format(cnt, total))

        id = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        kernel = np.array([[-0.125, -0.125, -0.125, -0.125, -0.125],
                           [-0.125, 0.250, 0.250, 0.250, -0.125],
                           [-0.125, 0.250, 1.000, 0.250, -0.125],
                           [-0.125, 0.250, 0.250, 0.250, -0.125],
                           [-0.125, -0.125, -0.125, -0.125, -0.125]])

        img_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        img_sharp = cv2.fastNlMeansDenoisingColored(img_sharp, None, 10, 10, 7, 21)

        cv2.imwrite('{}/{}'.format(target_dir, id), img_sharp)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='data/datasets/svhn/images', type=str)
    parser.add_argument('--dir', '-d', default='train', type=str)
    parser.add_argument('--target', '-t', default='train_sharp', type=str)
    args = parser.parse_args()

    main(args)

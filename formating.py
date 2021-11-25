import json
import os
import argparse
from PIL import Image
import cv2
import random
import numpy as np


def write_yolo(args, data_dict):
    # generate yolo txt format
    print('generating yolo txt files for each image ...')

    img_path_list = []
    for e in data_dict:
        train_image_path = os.path.join(args.root, args.train_image_prefix, e['filename'])
        label_path = os.path.join(args.root, args.label_prefix, '{}.txt'.format(e['filename'].split('.')[0]))

        assert os.path.exists(train_image_path)
        img_path_list.append(train_image_path)

        f_label = open(os.path.join(label_path, ), 'w')
        image = Image.open(train_image_path)
        (img_w, img_h) = image.size

        for bbox in e['boxes']:
            l = 0 if bbox['label'] == 10 else bbox['label']
            cx = (bbox['left'] + bbox['width'] / 2.0) / img_w
            cy = (bbox['top'] + bbox['height'] / 2.0) / img_h
            w = bbox['width'] / img_w
            h = bbox['height'] / img_h
            f_label.write('{} {} {} {} {}\n'.format(l, cx, cy, w, h))

        f_label.close()

    return img_path_list


def dump_train_val(args, img_path_list):
    # generate train/val .txt file
    print('generating train/val .txt ...')

    length = len(img_path_list)
    thresh = int(args.split * length)
    random_indexes = random.sample(range(length), length)

    kfold = args.kfold
    assert kfold * thresh <= length

    for k in range(kfold):
        train_path = os.path.join(args.root, 'train_{}.txt'.format(k))
        val_path = os.path.join(args.root, 'val_{}.txt'.format(k))
        f_train = open(train_path, 'w')
        f_val = open(val_path, 'w')

        for i in range(length):
            if i >= np.floor(k * thresh) and i < np.floor((k + 1) * thresh):
                f_val.write('{}\n'.format(img_path_list[random_indexes[i]]))
            else:
                f_train.write('{}\n'.format(img_path_list[random_indexes[i]]))

        f_train.close()
        f_val.close()
        print('{} saved'.format(train_path))
        print('{} saved'.format(val_path))


def dump_test(args):
    # writing test
    print('writing test.txt ...')
    f_test = open(os.path.join(args.root, 'test.txt'), 'w')
    test_image_path = os.path.join(args.root, args.test_image_prefix)
    test_imgs = os.listdir(test_image_path)
    test_imgs.sort(key=lambda x: int(x[:-4]))
    for test_img in test_imgs:
        f_test.write('{}\n'.format(os.path.join(args.root, args.test_image_prefix, test_img)))

    f_test.close()
    print('{} saved'.format(os.path.join(args.root, 'test.txt')))


def main(args):
    f_json = open(os.path.join(args.root, args.json), 'r')
    data_dict = json.load(f_json)

    # .txt for each .png
    img_path_list = write_yolo(args, data_dict)
    assert len(img_path_list) == len(data_dict)

    # dump .txt files
    dump_train_val(args, img_path_list)
    dump_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='data/datasets/svhn/', type=str)
    parser.add_argument('--train_image_prefix', '-train', default='images/train', type=str)
    parser.add_argument('--test_image_prefix', '-test', default='images/test', type=str)
    parser.add_argument('--label_prefix', '-lpf', default='labels/train', type=str)
    parser.add_argument('--json', '-j', default='annotations/digitStruct.json', type=str)
    parser.add_argument('--split', '-s', default=0.1, type=float)
    parser.add_argument('--kfold', '-k', default=5, type=int)
    args = parser.parse_args()

    main(args)

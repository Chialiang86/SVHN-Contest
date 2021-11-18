import json
import os
import argparse
from PIL import Image
import cv2
import random


def main(args):
    f_json = open(os.path.join(args.root ,args.json), 'r')
    data_dict = json.load(f_json)

    f_train = open(os.path.join(args.root, 'train.txt'), 'w')
    f_val = open(os.path.join(args.root, 'val.txt'), 'w')
    f_test = open(os.path.join(args.root, 'test.txt'), 'w')
    img_path_list = []


    # generate yolo txt format
    print('generating yolo txt files for each image ...')
    for e in  data_dict:
        train_image_path = os.path.join(args.root, args.train_image_prefix, e['filename'])
        label_path = os.path.join(args.root, args.label_prefix, '{}.txt'.format(e['filename'].split('.')[0]) )
        
        assert os.path.exists(train_image_path)
        img_path_list.append(train_image_path)


        # img = cv2.imread(train_image_path)
        
        f_label = open(os.path.join(label_path, ), 'w')
        image = Image.open(train_image_path)
        (img_w, img_h) = image.size
        
        for bbox in e['boxes']:
            l = 0 if bbox['label'] == 10 else bbox['label']
            cx = (bbox['left'] + bbox['width'] / 2.0) / img_w
            cy = (bbox['top'] + bbox['height'] / 2.0) / img_h
            w = bbox['width'] / img_w
            h = bbox['height']  / img_h
            f_label.write('{} {} {} {} {}\n'.format(l, cx, cy, w, h))
            # cv2.rectangle(img, (int(bbox['left']), int(bbox['top'])), (int(bbox['left'] + bbox['width']), int(bbox['top'] + bbox['height'])), (255, 255, 0), 2, cv2.LINE_AA)

        f_label.close()

        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    assert len(img_path_list) == len(data_dict)

    # generate train/val .txt file
    print('generating train/val .txt ...')
    length = len(img_path_list)
    thresh = int(args.split * length)
    random_indexes = random.sample(range(length), length)
    for i in range(length):
        if i < thresh:
            f_val.write('{}\n'.format(img_path_list[random_indexes[i]]))
        else :
            f_train.write('{}\n'.format(img_path_list[random_indexes[i]]))
    
    f_train.close()
    f_val.close()
    print('{} saved'.format(os.path.join(args.root, 'train.txt')))
    print('{} saved'.format(os.path.join(args.root, 'val.txt')))

    # writing test
    print('writing test.txt ...')
    test_image_path = os.path.join(args.root, args.test_image_prefix)
    test_imgs = os.listdir(test_image_path)
    test_imgs.sort(key = lambda x: int(x[:-4]))
    for test_img in test_imgs:
        f_test.write('{}\n'.format(os.path.join(args.root, args.test_image_prefix, test_img)))
    
    f_test.close()
    print('{} saved'.format(os.path.join(args.root, 'test.txt')))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='data/datasets/svhn/', type=str)
    parser.add_argument('--train_image_prefix', '-train', default='images/train', type=str)
    parser.add_argument('--test_image_prefix', '-test', default='images/test', type=str)
    parser.add_argument('--label_prefix', '-lpf', default='labels/train', type=str)
    parser.add_argument('--json', '-j', default='annotations/digitStruct.json', type=str)
    parser.add_argument('--split', '-s', default=0.1, type=float)
    args = parser.parse_args()

    main(args)
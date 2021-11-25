#!/bin/bash

echo "functions : ${1}"

dataset="data/svhn.yaml"
model="yolov5m.yaml"
pretrained="yolov5m6.pt"
val_task="test"

running_id="11"
img_size=512
batch_size=32
epoch=30

conf_thres=0.001
iou_thres=0.6

weights_0="runs/${running_id}/train/weights/best.pt"
weights_1="runs/9/train/weights/best.pt"
weights_2="runs/8/train/weights/best.pt"
weights_2="runs/7/train/weights/best.pt"
one_img="data/datasets/svhn/images/test/117.png"

# python sharpen.py --dir test --target test_sharp
# python formating.py -train images/train_sharp -test images/test_sharp -lpf labels/train_sharp

if [ ${1} = "train" ]; then
    echo "in training mode"
    python3 train.py --img $img_size --batch-size $batch_size --epochs $epoch --data $dataset --cfg $model --weights $pretrained --project "runs/$running_id" --multi-scale

elif [ ${1} = "test" ]; then
    echo "in testing mode"
    python3 val.py --img $img_size --data $dataset --task $val_task --weights $weights_0 --conf-thres $conf_thres --iou-thres $iou_thres --project "runs/$running_id" --save-json 

elif [ ${1} = "det" ]; then
    echo "in detecting mode"
    python3 detect.py --weights $weights --source $one_img  --imgsz $img_size --conf-thres $conf_thres --iou-thres $iou_thres --project "runs/$running_id"

else
    echo "args error : you should specify an arg (train/test/det)"
fi
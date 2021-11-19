#!/bin/bash

echo "functions : ${1}"

dataset="data/svhn.yaml"
model="yolov5l.yaml"

runnung_id="2"
img_size=320
batch_size=32
epoch=20

conf_thres=0.2
iou_thres=0.5

weights="runs/2/train2/weights/best.pt"
one_img="data/datasets/svhn/images/test/117.png"

if [ ${1} = "train" ]; then
    echo "in training mode"
    python3 train.py --img $img_size --batch-size $batch_size --epochs $epoch --data $dataset --cfg $model --weights '' --project "runs/$runnung_id"

elif [ ${1} = "test" ]; then
    echo "in testing mode"
    python3 val.py --img $img_size --data $dataset --weights $weights --conf-thres $conf_thres --iou-thres $iou_thres --project "runs/$runnung_id" --save-json 

elif [ ${1} = "det" ]; then
    echo "in detecting mode"
    python3 detect.py --weights $weights --source $one_img  --imgsz $img_size --conf-thres $conf_thres --iou-thres $iou_thres --project "runs/$runnung_id"

else
    echo "args error : you should specify an arg (train/test/det)"
fi
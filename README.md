
# CodaLab Competetion : Street View House Numbers detection

> contest link : https://competitions.codalab.org/competitions/35888#learn_the_details
> slide : [link](https://docs.google.com/presentation/d/1uPEuvBi3gyX7tq4MvuPp3d5JePWJa9rmVqfMcDrm4Mg/edit#slide=id.gfd55e7c5d5_0_0)

## Environment
- Ubuntu 20.04.1
- python 3.8.10

## Download
```shell
$ git clone https://github.com/Chialiang86/SVHN-Contest.git
```

## Requirements
```
# Base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0

# Logging -------------------------------------
tensorboard>=2.4.1
# wandb

# Plotting ------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

# Export --------------------------------------
# coremltools>=4.1  # CoreML export
# onnx>=1.9.0  # ONNX export
# onnx-simplifier>=0.3.6  # ONNX simplifier
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1  # TFLite export
# tensorflowjs>=3.9.0  # TF.js export

# Extras --------------------------------------
# albumentations>=1.0.3
# Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
# pycocotools>=2.0  # COCO mAP
# roboflow
thop  # FLOPs computation
```

### Download requirements
```shell 
$ cd SVHN-Contest
$ pip install -r requirements.txt
```

## Training on SVHN dataset

#### Training architecture and the opensource link
- I used [YOLOv5](https://github.com/ultralytics/yolov5) as my main training architecture

#### 1. Add dataset to folder (IMPORTANT!)
- Move `train/test` folder which contrains SVHN image data to `data/datasets/svhn/images/` (feel free to replace old train/test folder in it)
- Copy `digitStruct.mat` to the root of this repo

#### 2. Transfer .mat file to .json file

In order to generate .json files for dataset, I used [this GitHub repo](https://github.com/Bartzi/stn-ocr/blob/master/datasets/svhn/svhn_dataextract_tojson.py) to produce .json file. After running the cammand below, the .json file will be saved in `data/datasets/svhn/annatations/digitStruct.json`

```bash 
$ python read_svhn_mat.py
```

#### 3. Transfer .json file to yolo format

- In yolo format, each training image corresponded to a .txt file with same file name
- Each row in .txt files was : `class x_center y_center width height` format. 
- Besides, bounding box coordinates must be in normalized xywh format (from 0 - 1).
- Class numbers are zero-indexed (start from 0)
```shell 
$ python formating.py
```

#### 4. Pretrained weights for Yolov5
- `train.py` in this repo will download weights automatically
- The pretrined weights I used : 
    - yolov5m6.pt
    - yolov5l6.pt

#### 5. Command line options of train.py 
- img : image size of the given image, ex 320 -> (320, 320)
- batch-size : batch size
- epochs : number of epochs
- data : dataset information in .yaml format (ex : data/svhn.yaml)
- cfg : config files for training archtecture, in .yaml format [`yolov5s.yaml`, `yolov5m.yaml`, `yolov5l.yaml`, `yolov5x.yaml`]
- weights : pretrained model, for example : [`yolov5s6.pt`, `0yolov5m6.pt`, `yolov5l6.pt`, `yolov5x6.pt`]
- project : the output directory to store training/testing/validation results
- multi-scale : whether to train with multiple scale images (store true)

#### 6. Training example
```python 
$ python3 train.py --img 368 --batch-size 32 --epochs 30 --data data/svhn.yaml --cfg yolov5m.yaml --weights yolov5m6.pt --project runs/12 --multi-scale
```

## Generate answer.json
- For example
```python 
$ python3 val.py --img 512 --data data/svhn.yaml --task test --weights runs/11/train/weights/best.pt --conf-thres 0.001 --iou-thres 0.6 --project runs/11 --save-json 
```

## Reproduce the best submission file
1. Add dataset to folder (IMPORTANT!)
    - Move `train/test` folder which contrains SVHN image data to `data/datasets/svhn/images/` (feel free to replace old train/test folder in it)
2. Download the best weight 
-> google drive link : [best-11.pt](https://drive.google.com/file/d/1opcMexv-tEG4jJJ-uS6pjetxSHl3KmPn/view?usp=sharing)
3. Drag the weight file to the root of the project folder
4. Run the command below, then answer.json will be produced in the same direcory
5. The predicting result will be saved in `result/`

```bash 
$ python3 val.py --img 512 --data data/svhn.yaml --task test --weights best-11.pt --conf-thres 0.001 --iou-thres 0.6 --project result --save-json 
```

The predicting result on testing set : 


- Label
![](https://i.imgur.com/TUEw5J6.jpg)
- Predict
![](https://i.imgur.com/S2SCMql.jpg)


## Best Result


| User        | Team Name |Score   |
| ------------|-----------|--------|
| Chialiang86 | 310552027 |0.431096|


## References
- YOLOv5 : https://github.com/ultralytics/yolov5
- From .mat to .json for SVHN dataset : https://github.com/Bartzi/stn-ocr/blob/master/datasets/svhn/svhn_dataextract_tojson.py
# Needle Drop Detection 
Pytorch implementation of a deep learning model to classify whether the needle in each video frame is:
1) grabbed by either left or right surgical tool, or
2) dropped

<img src="./outputs/caseid_000184_fps1.gif" />


## Folder structure
```
.
├── data
│   ├── images              # extracted frames for all videos
│   ├── videos
│   │   └ fps1              # original videos used for training and validation
│   └── test_videos         # test videos
├── outputs                 # output videos created by test.py
├── runs                    # saved checkpoints
├── train.csv               # training dataset annotations
├── val.csv                 # validation dataset annotations
├── extract_frames.py       # script for extracting frames, saved in data/images
├── train.py                # script for train and val
└── test.py                 # script for create the ouput videos

```

## Description

Two models are trained for frame-by-frame classification using CNN (efficient-net):
- Grabbed/Dropped (GD): a binary classifier to identify if the needle is grabbed or dropped
- Grabbed_with_left/Grabbed_eith_right (LR): a multi-label classifier to identify if the needle is grabbed by the left and right tools or not

## Requirements
- python 3.8
```
torch
torchvision
timm
opencv-python
pandas
joblib
imageio
imageio-ffmpeg
tqdm
scikit-learn
wandb
```
## Usage
- create a virtual environment
- install the requirements: 
```shell
pip install -r requirements.txt
```
- Extract the frames of all the videos: All the frames of the videos in datavideos/fps1 will be cropped to remove the black margin, resized to (500,500) and saved in data/images folder
```shell
python extract.py
```
- train the models:
```shell
// for grabbed/dropped model
python train.py --train_dir ".\data\images" --test_dir ".\data\images" --batch_size 128 --num_epochs 30 --device "cuda" --LR 0.001 --checkpoint_path "./runs/CNN_GD.pth" --left_right False --amp True

// for left/right model
python train.py --train_dir ".\data\images" --test_dir ".\data\images" --batch_size 128 --num_epochs 30 --device "cuda" --LR 0.001 --checkpoint_path "./runs/CNN_LR.pth" --left_right True --amp True
```
- test the trained model and save the output video:
```shell
// for grabbed/dropped model
python test.py --video "./data/test_videos/caseid_000070_fps1.mp4" --checkpoint "./runs/CNN_GD1.pth" --output "./outputs" --left_right False --device "cuda" --amp True

// for left/right model
python test.py --video "./data/test_videos/caseid_000070_fps1.mp4" --checkpoint "./runs/CNN_LR1.pth" --output "./outputs" --left_right True --device "cuda" --amp True
```

## Results
The validation set (val.csv) include 8 videos with Case_ids: [000168, 000177, 000057, 000064, 000007, 000010, 000011, 000015]. The reamaining videos were used for training (train.csv).

| Model         | Accuracy | F1-Score |
| ------------- | ------------- | ------------- |
| Grabbed/Dropped  |  0.95 | 0.79
| Left/Right  |  0.94 | 0.79
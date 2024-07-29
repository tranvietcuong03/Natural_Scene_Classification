# Natural Scenes Classification
![pytorch](https://img.shields.io/badge/pytorch-%23013243.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![open-cv](https://img.shields.io/badge/open--cv-%23150458.svg?style=for-the-badge&logo=open-cv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-%23013000.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Introduction
I build a simple CNN-Convolutional Neural Network to classify the natural scenes. The categories in dataset: buildings, forest, glacier, mountain, sea, street.

## Demo
Classify some pictures of natural scenes:

![...](https://github.com/tranvietcuong03/Natural_Scene_Classification/blob/master/Visualization/tests.png). <br>
Detect and classify some natural scenes in video: 

![...](https://github.com/tranvietcuong03/Natural_Scene_Classification/blob/master/Visualization/result.gif)

## Dataset
There are four folders: train, valid, test, video. Train and Valid folder are used to train model, then apply inference with image by Test folder and video by Video folder. <br>
Dataset: [Download here](https://github.com/tranvietcuong03/Natural_Scene_Classification/tree/master/natural_scenes) <br>

## Training Progress
The training progress is presented by tensorboard

![...](https://github.com/tranvietcuong03/Natural_Scene_Classification/blob/master/Visualization/tensorboard.png).
![...](https://github.com/tranvietcuong03/Natural_Scene_Classification/blob/master/Visualization/best_acc.png).

Moreover, the **train_progress.txt** file saved the state of each epoch.

## Requirements
**Python 3.12**<br>
**pytorch**<br>
**cv2**<br>
**sklearn**<br>
**numpy**

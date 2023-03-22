# Retinal Vessel Segmentation through an Active Learning Framework

## Introduction
The aim of the project is the reduction in the number of annotated images for a weakly supervised task of segmentation of retinal vessels, using an active learning framework which provides to the Oracle only the most uncertain images to label.
Images are taken from the [DRIVE dataset](https://paperswithcode.com/dataset/drive)

## Index
1. Preprocessing
2. Classification task with active learning framework
3. Segmentation task

## Preprocessing 
In the preprocessing phase, there is a preliminary extraction of 32 X 32 patches from the input images. Only images with at least a partial not-black background are consider in the training, the others are simply neglected.
<p align="center">
  <img src="imgs/metodo.png" alt="Image of preprocessing"/>
</p> 

## Classification task with active learning framework
we used an Active Learning technique on the classifier that consists in selecting the most useful samples from the unlabeled dataset and send them to an oracle for the annotation.
As a classification network, we mainly used the PNET architecture, but we also tried well known networks as ResNet50 and VGG16, both pre-trained on ImageNet dataset and fine-tuned on eye’s images.
As uncertainty measures we implemented Least Confidence and Entropy, that both returns the sample on which the classifier
is more uncertain on.
Data augmentation is also applied to the images.
<p align="center">
  <img src="imgs/rete.png" alt="Image of classification"/>
</p> 

In order to obtain a first approximation of pixel-level labels we used K-means clustering algorithm and OpenCV Canny method.
<p align="center">
  <img src="imgs/kmeans_canny.png" alt="Image of output from kmeans and Canny"/>
</p> 

## Segmentation task
. As segmentation network we exploited the Unet [7] ;
in particular we used two 2D-Unets in cascade [3] which was extremely useful since the first Unet will recover rough-mask
labels coming from Canny and K-means and the second Unet will produce better segmentations thanks to the output of the
first Unet.

<p align="center">
  <img src="imgs/output.png" alt="Image of output from kmeans and Canny"/>
</p> 

## Train

### 0) Optional - Create and Activate virtual environment (using Conda)
```bash
  $ conda create --name RetinalAL python=3.8
  $ conda activate RetinalAL
```

### 1) Install required packages from requirements.txt
```bash
  $ pip install -r requirements.txt
```

### 2) Create patched images
```bash
  $ python3 create_patches.py
```

### 3) Start training
```bash
  $ python3 models_training.py
```

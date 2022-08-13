# Self Supervised Data Labeling ML
A self-supervised data labeling ML YoloX with multi-GPU training and automatic hyperparameter tuning.
# Introduction
I completed the Intro to Deep Learning with PyTorch and Intro To Self Driving Cars Nanodegree by Udacity on August 9th, 2022. I had the opportunity to sharpen my Python skills and apply C++ while implementing matrices and calculus in code. Additionally, I was able to touch on computer vision and machine learning all in the context of solving self-driving car problems.

<p align="center">
  <img src="https://user-images.githubusercontent.com/86870298/184509548-84a5ce8f-c9e1-4841-8430-7ae33ecb0107.png" alt="cer" height="500"/>
</p>

# Datasets
### LISA Traffic Light Dataset
The dataset used for this project is the [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset). The database consists of continuous test and training video sequences, totaling 43,007 frames and 113,888 annotated traffic lights.

Label Distribution         |  Image Examples
:-------------------------:|:-------------------------:
![LISA dataset](https://user-images.githubusercontent.com/86870298/184504395-c65dced9-92f1-4dad-bdc2-cf92675b7653.png)  |  ![LISA sample](https://user-images.githubusercontent.com/86870298/184509916-2d94ae25-88b2-40a6-bce8-6e3f27c2bb8e.png)

```
go : 46707 images
stop : 44318 images
warning : 2669 images
Percentage of: go : 49.85 %
Percentage of: stop : 47.3 %
Percentage of: warning : 2.85 %
Total: 93694 images
```
### MIT self-driving car course dataset
I wanted to test the self-supervised data labeling and model on an entirely different dataset. I found the dataset from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/).

Label Distribution         |  Image Examples
:-------------------------:|:-------------------------:
![mit dataset](https://user-images.githubusercontent.com/86870298/184504406-36b44cdc-dbb3-4c9b-99f8-9671f52fda32.png)  |  ![mit sample](https://user-images.githubusercontent.com/86870298/184509966-cc17e367-d8c9-4221-9362-d1917d6088e9.png)

```
go : 536 images
stop : 904 images
warning : 44 images
Percentage of: go : 36.12 %
Percentage of: stop : 60.92 %
Percentage of: warning : 2.96 %
Total: 1484 images
```

# Self Supervised Data Labeling
I created a method to self label the traffic lights to "go", "stop", and "warning" using OpenCV masks. This method creates three masks for each image (red, yellow, green). Then it calculates the probability of each label using weights and cutoffs.
### LISA Traffic Light Dataset
For the LISA dataset, the results were:
```
Training set:
Correct:  74096 / 74955  Accuracy: 98.85 %
Will stop when supposed to go accuracy: 1.02%  Go incorrect: 0.12%
```
```
Testing set:
Correct:  18536 / 18739  Accuracy: 98.92 %
Will stop when supposed to go accuracy: 0.99%  Go incorrect: 0.10%
```
#### Example of good results:
Green Hue    |  Green
:-------------------------:|:-------------------------:
![LISA green correct](https://user-images.githubusercontent.com/86870298/184511237-ad4109e8-05d7-43a7-9487-27119f54cd55.png) | ![LISA green correct2](https://user-images.githubusercontent.com/86870298/184511254-340b573f-8ae7-474a-a727-35aaeeef713f.png)
Red Hue        |  Red
![LISA red correct2](https://user-images.githubusercontent.com/86870298/184511721-dcfd4268-2d81-409d-a474-0772491a65d0.png) | ![LISA red correct](https://user-images.githubusercontent.com/86870298/184511728-508db0a8-360a-4b1f-8782-1c42a2ee4e55.png)


#### Example of bad results:
Unreadable by humans |  Color mixup
:-------------------------:|:-------------------------:
![LISA not readable incorrect](https://user-images.githubusercontent.com/86870298/184511291-27b39163-bbec-457d-bd45-7b8c89401eab.png) | ![LISA orange-blue incorrect](https://user-images.githubusercontent.com/86870298/184511318-d20e8be9-56d7-4c49-98b1-0b69f3a1e37a.png)
Blue, not green |
![LISA blue incorrect](https://user-images.githubusercontent.com/86870298/184511341-ec87b9f3-581d-48cd-8635-ccfe58ff0bbb.png) |

### MIT self-driving car course dataset
For the MIT dataset, the results were:
```
Correct:  1408 / 1484  Accuracy: 94.88 %
Will stop when supposed to go accuracy: 3.84%  Go incorrect: 1.28%
```
#### Example of good results:
Yellow side | Yellow
:-------------------------:|:-------------------------:
![mit yellow correct](https://user-images.githubusercontent.com/86870298/184511783-d3ff4590-acf1-4658-ac0c-cec19c742c80.png) | ![mit yellow correct2](https://user-images.githubusercontent.com/86870298/184511785-077c1bcb-5983-47ee-b7ad-7c3b2fc0a0ab.png)
Red Hue        |  Red
![mit red correct2](https://user-images.githubusercontent.com/86870298/184511816-b34c64c4-b2ec-40ed-accf-4c060c2a2e7d.png) | ![mit red correct3](https://user-images.githubusercontent.com/86870298/184511821-bb3536b3-aa48-4bcc-b650-d49f4715cc1b.png)

#### Example of bad results:
Light off | Grey
:-------------------------:|:-------------------------:
![mit green correct2](https://user-images.githubusercontent.com/86870298/184511847-1d8d7dac-2257-4bf1-99c4-26f6887bd8b1.png) | ![mit gray incorrect](https://user-images.githubusercontent.com/86870298/184511617-f581d896-0d9f-42ec-af15-246bbc725a54.png)
Unreadable by humans        |  Low brightness
![mit green incorrect](https://user-images.githubusercontent.com/86870298/184511861-739b8cb1-3dd5-4c92-ba8a-11fb823eec61.png) | ![mit orange incorrect](https://user-images.githubusercontent.com/86870298/184511876-4531ae59-fb1f-4a78-9dce-879460043896.png)

# YoloX

# Multi-GPU Training and Automatic Hyperparameter Tuning

# Requirements and file architecture:
### Imports:
```python
import os
import cv2
import numpy as np
import pickle
import multiprocessing as mp
import re
import pandas as pd
import matplotlib.pyplot as plt
import logging
import progressbar

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
```
### File Architecture:
```
├── README.md
├── archive 
│   ├── lisa-traffic-light-dataset
│   ├── traffic_light_images
├── dataset # contains the data used for training and testing after preprocessing
│   ├── LISA_train_dataset
│   ├── LISA_test_dataset
│   ├── LISA_self_supravised_train_labels
│   ├── LISA_self_supravised_test_labels
│   ├── MIT_test_dataset
├── preprocessing.py # contains the code for preprocessing the dataset
├── train.py # contains the code for training the model
├── self_supervised_data_labeling.py # contains the code for self supervised data labeling
```

## Instructions:
- Download the LISA Traffic Light Dataset from https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset into the archive folder.
- Download the images from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) into the archive folder.
- run the preprocessing.py file to preprocess the dataset.
- run the self_supervised_data_labeling.py file to generate the self supervised data labeling.
- run the train.py file to train the model.

# What does it do?
asdsdasd

## Example:

![2_3 1](https://user-images.githubusercontent.com/86870298/180622921-41e9b082-9fb9-4ad9-a46e-7acd7e16bcc7.png)

# Limitations and known bugs:
jkkj

## Example of bad detection:
sdz

## Next steps:
- [ ] Improve the model's accuracy.
- [ ] Improve relabeling accuracy by taking exposure into account.
- [ ] Move preprocessing to the GPU.
- [ ] Add more self supervised data labeling methods.
  - [ ] [Scikit-learn classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)
  - [ ] [Scikit-learn Comparison of Calibration of Classifiers](https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py)

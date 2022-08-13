# Self Supervised Data Labeling ML
A self-supervised data labeling ML YoloX with multi-GPU training and automatic hyperparameter tuning.
# Introduction
I completed the Intro to Deep Learning with PyTorch and Intro To Self Driving Cars Nanodegree by Udacity on August 9th, 2022. I had the opportunity to sharpen my Python skills and apply C++ while implementing matrices and calculus in code. Additionally, I was able to touch on computer vision and machine learning all in the context of solving self-driving car problems.

![k;j](https://user-images.githubusercontent.com/86870298/183752079-14f63e23-31ed-4200-aea0-d924909e9557.png)

# Datasets
The dataset used for this project is the LISA Traffic Light Dataset. The database consists of continuous test and training video sequences, totaling 43,007 frames and 113,888 annotated traffic lights. The dataset is available for download at https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset. 

![LISA dataset](https://user-images.githubusercontent.com/86870298/184504395-c65dced9-92f1-4dad-bdc2-cf92675b7653.png)
![LISA sample](https://user-images.githubusercontent.com/86870298/184504404-dd80bb96-030d-47a7-ab9d-33d6b1c4467f.png)

```
go : 46707 images
stop : 44318 images
warning : 2669 images
Percentage of: go : 49.85 %
Percentage of: stop : 47.3 %
Percentage of: warning : 2.85 %
Total: 93694 images
```

I wanted to test the self-supervised data labeling and model on an entirely different dataset. I found the dataset from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/).

![mit dataset](https://user-images.githubusercontent.com/86870298/184504406-36b44cdc-dbb3-4c9b-99f8-9671f52fda32.png)
![mit sample](https://user-images.githubusercontent.com/86870298/184504407-227f7759-87f3-4dac-8320-7ee0a1e7592e.png)

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
I created a method...

# YoloX

# Multi-GPU Training and Automatic Hyperparameter Tuning

# Requirements and file architecture:
### Imports:
```python
import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
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
- [ ] Move preprocessing to the GPU.
- [ ] Add more self supervised data labeling methods.

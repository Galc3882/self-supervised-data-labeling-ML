import os
import re
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def getAnnotationDataframe():
    """
    Get annotation dataframe from train data folders
    """
    # TODO: comment the function

    # get all folders in archive
    folders = os.listdir(os.path.abspath(os.getcwd())+r"\archive")

    # folders to use
    trainFolderList = [
        'dayTrain',
        'daySequence1',
        'daySequence2',
        #     'sample-dayClip6',
        'nightTrain',
        'nightSequence1',
        'nightSequence2',
        #     'sample-nightClip1',
    ]

    # build dataframe on allowed folders
    annotation_list = list()
    for folder in folders:
        if folder in trainFolderList:
            annotation_path = os.path.abspath(
                os.getcwd())+r"\archive\Annotations\Annotations" + "\\" + folder
            image_frame_path = os.path.abspath(
                os.getcwd()) + "\\archive\\" + folder + "\\" + folder
            df = pd.DataFrame()
            if 'Clip' in os.listdir(annotation_path)[0]:
                clip_list = os.listdir(annotation_path)
                for clip_folder in clip_list:
                    df = pd.read_csv(
                        annotation_path + "\\" + clip_folder + r'\frameAnnotationsBOX.csv', sep=";")
                    df['image_path'] = image_frame_path + \
                        "\\" + clip_folder + r'\frames' + "\\"
                    annotation_list.append(df)
            else:
                df = pd.read_csv(annotation_path +
                                 r'\frameAnnotationsBOX.csv', sep=";")
                df['image_path'] = image_frame_path + r'\frames' + "\\"
                annotation_list.append(df)

    df = pd.concat(annotation_list)
    df = df.drop(['Origin file', 'Origin frame number',
                 'Origin track', 'Origin track frame number'], axis=1)
    df.columns = ['filename', 'target', 'x1', 'y1', 'x2', 'y2', 'image_path']
    df = df[df['target'].isin(target_classes)]
    df['filename'] = df['filename'].apply(
        lambda filename: re.findall("\/([\d\w-]*.jpg)", filename)[0])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def showDataset(df):
    """
    Show dataset distribution
    """
    index, counts = np.unique(df['target'], return_counts=True)
    colors = [color_map[target] for target in index]
    plt.bar(index, counts, color=colors)
    print(df['target'].value_counts())
    plt.show()


def crop(img, row):
    """
    Crop image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[row['y1']:row['y2'], row['x1']:row['x2']]
    # img = cv2.resize(img, (64, 64))
    return img


def imagesCrop(df):
    """
    Crop images from dataframe
    """
    # init the progress bar
    pbar = progressbar.ProgressBar(maxval=len(df), widgets=["Cropping images: ", progressbar.Percentage(
    ), " ", progressbar.Bar(), " ", progressbar.ETA()]).start()

    # save cropped images in a dictionary for each label
    imgs = dict()
    for i, row in df.iterrows():
        pbar.update(i)
        img = cv2.imread(row['image_path']+row['filename'])
        img = crop(img, row)
        if row['target'] not in imgs:
            imgs[row['target']] = list()
        imgs[row['target']].append(img)
    pbar.finish()
    return imgs


def showRandomImage(imgs):
    """
    Show a set of random images from each label in dataset
    """
    _, ax = plt.subplots(1, 3)
    for i, target in enumerate(target_classes):
        ax[i].imshow(imgs[target][np.random.randint(0, len(imgs[target]))])
        ax[i].set_title(target)
    plt.show()


def relabelImages(df):
    """
    Relabel images to be used in training
    return a list of labels
    """
    # init the progress bar
    pbar = progressbar.ProgressBar(maxval=len(df), widgets=["Relabeling images: ", progressbar.Percentage(
    ), " ", progressbar.Bar(), " ", progressbar.ETA()]).start()

    # relabel images
    labels = list()
    for i, row in df.iterrows():
        pbar.update(i)
        img = cv2.imread(row['image_path']+row['filename'])
        img = crop(img, row)
        label = relabel(img)
        if label != row['target']:
            print('labels incorrect', ' label: ',
                  label, ' target: ', row['target'])
            relabel(img, show=True)
        labels.append(label)
    pbar.finish()
    return labels

#cv2.cvtColor(np.uint8([[[255,255,255 ]]]),cv2.COLOR_RGB2HSV)
def relabel(img, show=False):
    """
    give a label to an image
    """
    # convert to HSV
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # create three masks for each color (red, green, yellow)
    red_mask = cv2.bitwise_or(cv2.inRange(hsvImg, (0, 80, 80), (20, 255, 255)), cv2.inRange(
        hsvImg, (170, 80, 80), (190, 255, 255)))
    green_mask = cv2.inRange(hsvImg, (45, 80, 80), (100, 255, 255))
    yellow_mask = cv2.inRange(hsvImg, (20, 80, 80), (45, 255, 255))

    # save only the biggest contour
    # red
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        red_mask = np.zeros_like(red_mask)
        cv2.drawContours(red_mask, [contours[0]], 0, 255, -1)
    # green
    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        green_mask = np.zeros_like(green_mask)
        cv2.drawContours(green_mask, [contours[0]], 0, 255, -1)
    # yellow
    contours, _ = cv2.findContours(
        yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        yellow_mask = np.zeros_like(yellow_mask)
        cv2.drawContours(yellow_mask, [contours[0]], 0, 255, -1)

    if show:
        # show masks with title and original image
        _, ax = plt.subplots(1, 4)
        ax[0].imshow(img)
        ax[0].set_title('Original')
        ax[1].imshow(red_mask, cmap='gray')
        ax[1].set_title('Red')
        ax[2].imshow(green_mask, cmap='gray')
        ax[2].set_title('Green')
        ax[3].imshow(yellow_mask, cmap='gray')
        ax[3].set_title('Yellow')
        plt.show()
        i = 0  # ! remove this line

    # check which mask has the most pixels and if red is at the top, yellow in the middle and green at the bottom
    # ! np.sum(green_mask) > np.sum(red_mask) and np.sum(green_mask) > np.sum(yellow_mask) and
    if np.sum(green_mask[int(len(green_mask)*(2/3)):, :]) > np.sum(red_mask[0:int(len(red_mask)*(1/3)), :]) and np.sum(green_mask[int(len(green_mask)*(2/3)):, :]) > np.sum(yellow_mask[int(len(yellow_mask)*(1/9)):int(len(yellow_mask)*(8/9)), :]):
        return 'go'
    elif np.sum(yellow_mask[int(len(yellow_mask)*(1/9)):int(len(yellow_mask)*(8/9)), :]) > np.sum(green_mask[int(len(green_mask)*(2/3)):, :]) and np.sum(yellow_mask[int(len(yellow_mask)*(1/9)):int(len(yellow_mask)*(8/9)), :]) > np.sum(red_mask[0:int(len(red_mask)*(1/3)), :]):
        return 'warning'
    else:
        return 'stop'


def compareLabels(recalculatedLabels, originalLabels):
    """
    Compare recalculated labels with original labels
    """
    # compare each label in the lists and print the results
    correct = 0
    stopIncorrect = 0
    goIncorrect = 0
    for i, target in enumerate(originalLabels):
        if target == recalculatedLabels[i]:
            correct += 1
        else:
            if recalculatedLabels[i] == 'stop':
                stopIncorrect += 1
            else:
                goIncorrect += 1
    print("Correct: ", correct, "/", len(originalLabels),
          " Accuracy: {:.2f}".format(100*correct/len(originalLabels)), "%")
    print("Will stop when supposed to go accuracy: {:.2f}".format(100*stopIncorrect/len(
        originalLabels))+"%", " Go incorrect: {:.2f}".format(100*goIncorrect/len(originalLabels))+"%")


if __name__ == '__main__':
    # initialize parameters
    target_classes = ['go', 'stop', 'warning']
    color_map = {'go': 'green', 'stop': 'red', 'warning': 'yellow'}
    rgb_color_map = {'go': (0, 255, 0), 'stop': (
        255, 0, 0), 'warning': (255, 255, 0)}

    # initialize hyperparameters
    # TODO: init hyperparameters

    # get annotation dataframe
    annotationDf = getAnnotationDataframe()

    # # decrease dataset size while keeping the same distribution and ratios
    # # TODO: delete this part later
    annotationDf = resample(annotationDf, n_samples=1000,
                            random_state=42).reset_index(drop=True)

    # recreate labels using openCV and
    selfSupravisedLables = relabelImages(annotationDf)

    # compare with original labels
    compareLabels(selfSupravisedLables, annotationDf['target'])

    # show dataset distribution
    showDataset(annotationDf)

    # # split into train and test
    # trainAnnotationDf, testAnnotationDf = train_test_split(
    #     annotationDf, test_size=0.2, random_state=42)
    # trainAnnotationDf = trainAnnotationDf.reset_index(drop=True)
    # testAnnotationDf = testAnnotationDf.reset_index(drop=True)

    # # show dataset distribution
    # #! showDataset(trainAnnotationDf)
    # #! showDataset(testAnnotationDf)

    # # get the cropped images
    # trainImages = imagesCrop(trainAnnotationDf)
    # testImages = imagesCrop(testAnnotationDf)

    # # Show some images
    # # for i in range(20):
    # #     showRandomImage(trainImages)

    # # create Yolo5 model

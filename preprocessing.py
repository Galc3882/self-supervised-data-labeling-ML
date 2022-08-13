from cProfile import label
import os
import pickle
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def getAnnotationDataframe():
    """
    Get annotation dataframe from train data folders
    """

    # get all folders in archive
    folders = os.listdir(os.path.abspath(os.getcwd()) +
                         r"\archive\LISA Traffic Light Dataset")

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
            # build dataframe for each folder
            df = pd.DataFrame()
            # get both clips and others
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

    # concat all dataframes
    df = pd.concat(annotation_list)
    # remove unwanted columns
    df = df.drop(['Origin file', 'Origin frame number',
                 'Origin track', 'Origin track frame number'], axis=1)
    # organize columns in a better way and reset index
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
    # create a color map for each label
    color_map = {target_classes[0]: 'green',
                 target_classes[1]: 'red', target_classes[2]: 'yellow'}
    # get the target classes and the number of images for each class
    index, counts = np.unique(df['target'], return_counts=True)
    # create a list of colors to corrolate with the labels
    colors = [color_map[target] for target in index]

    # plot the distribution
    plt.bar(index, counts, color=colors)
    plt.title('Dataset distribution')

    # print the number of images for each label and the total number of images and precentage of each label
    for i in range(len(index)):
        print(index[i], ':', counts[i], 'images')
    print('Percentage of:', index[0], ':', round(
        counts[0]/np.sum(counts)*100, 2), '%')
    print('Percentage of:', index[1], ':', round(
        counts[1]/np.sum(counts)*100, 2), '%')
    print('Percentage of:', index[2], ':', round(
        counts[2]/np.sum(counts)*100, 2), '%')
    print('Total:', np.sum(counts), 'images')

    plt.show()


def crop(row):
    """
    load image from path, change from BGR to RGB, crop, resize and return
    """
    img = cv2.imread(row['image_path']+row['filename'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[row['y1']:row['y2'], row['x1']:row['x2']]
    img = cv2.resize(img, (30, 50), interpolation=cv2.INTER_NEAREST)
    return img


def imagesCrop(df):
    """
    Crop images from dataframe
    """
    # init the progress bar
    pbar = progressbar.ProgressBar(maxval=len(df), widgets=["Cropping images: ", progressbar.Percentage(
    ), " ", progressbar.Bar(), " ", progressbar.ETA()])

    # multiprocessing to speed up the process of cropping images
    pool = mp.Pool()
    results = pool.imap(crop, [x[1] for x in list(df.iterrows())])

    # wait for the process to finish
    pbar.start()
    while results._length == None:
        pbar.update(results._index)
    pbar.finish()
    pool.close()

    # change from generator to numpy array
    arr = np.empty((len(df), 50, 30, 3), dtype=np.uint8)
    for i, img in enumerate(results):
        arr[i] = img

    # return a list of images and a list of labels
    return arr, list(df['target'])


def showRandomImages(imgs, labels):
    """
    Show a set of random images from each label in dataset in a 3x3 grid
    """
    # number of images to show from each label
    n = 3

    # set of unique labels
    labelsSet = list(set(labels))

    # get a list of random indexes for each label
    indexes = [np.random.choice(np.where(np.array(labels) == label)[
                                0], n, replace=False) for label in labelsSet]

    # show the images
    for i in range(n):
        for j in range(len(labelsSet)):
            plt.subplot(n, len(labelsSet), i*len(labelsSet)+j+1)
            plt.imshow(imgs[indexes[j][i]])
            if i == 0:
                plt.title(labelsSet[j])
            plt.axis('off')
            plt.axis("tight")
            plt.axis("image")
    plt.show()


def getMITData():
    """
    Get the dataset from the mit dataset
    """
    images = list()
    labels = list()
    # iterate over the folders and get the images and labels as lists
    for folder in os.listdir(os.path.abspath(os.getcwd())+r"\archive\traffic_light_images"):
        for target in os.listdir(os.path.abspath(os.getcwd())+r"\archive\traffic_light_images"+"\\"+folder):
            for img in os.listdir(os.path.abspath(os.getcwd())+r"\archive\traffic_light_images"+"\\"+folder+"\\"+target):
                img = cv2.imread(os.path.abspath(
                    os.getcwd())+r"\archive\traffic_light_images"+"\\"+folder+"\\"+target+"\\"+img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img, (30, 50), interpolation=cv2.INTER_NEAREST)
                images.append(img)
                if target == "green":
                    labels.append(target_classes[0])
                elif target == "red":
                    labels.append(target_classes[1])
                elif target == "yellow":
                    labels.append(target_classes[2])
    return images, labels


if __name__ == '__main__':
    # initialize parameters
    target_classes = ['go', 'stop', 'warning']

    # get annotation dataframe
    annotationDf = getAnnotationDataframe()

    # get mit dataset
    mitImages, mitLabels = getMITData()

    # # decrease dataset size while keeping the same distribution and ratios
    # annotationDf = resample(annotationDf, n_samples=1000,
    #                         random_state=42).reset_index(drop=True)

    # # show dataset distribution
    # ! showDataset(annotationDf)
    # ! showDataset({'target': mitLabels})

    # split into train and test
    trainAnnotationDf, testAnnotationDf = train_test_split(
        annotationDf, test_size=0.2, random_state=42)
    trainAnnotationDf = trainAnnotationDf.reset_index(drop=True)
    testAnnotationDf = testAnnotationDf.reset_index(drop=True)

    # get the cropped images
    trainImages, trainLabels = imagesCrop(trainAnnotationDf)
    testImages, testLabels = imagesCrop(testAnnotationDf)

    # show some images
    # ! showRandomImages(trainImages, trainLabels)
    # ! showRandomImages(mitImages, mitLabels)

    # if dataset folder doesn't exist, create it
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # pickle the train data
    with open('dataset/LISA_train_dataset.pickle', 'wb') as f:
        pickle.dump((trainImages, trainLabels), f)
        f.close()
    # pickle the test data
    with open('dataset/LISA_test_dataset.pickle', 'wb') as f:
        pickle.dump((testImages, testLabels), f)
        f.close()
    # pickle the mit data
    with open('dataset/MIT_test_dataset.pickle', 'wb') as f:
        pickle.dump((mitImages, mitLabels), f)
        f.close()

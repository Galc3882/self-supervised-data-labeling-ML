import pickle
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import preprocessing



if __name__ == '__main__':
    # get pickled train data
    with open('dataset/train.pickle', 'rb') as f:
        trainImages, trainLabels = pickle.load(f)
        f.close()
    
    # get pickled test data
    with open('dataset/test.pickle', 'rb') as f:
        testImages, testLabels = pickle.load(f)
        f.close()
    
    # get pickled self supervised labels
    with open('dataset/selfSupravisedTrainLabels.pickle', 'rb') as f:
        selfSupravisedTrainLables = pickle.load(f)
        f.close()
    with open('dataset/selfSupravisedTestLabels.pickle', 'rb') as f:
        selfSupravisedTestLables = pickle.load(f)
        f.close()

    # show some images
    # ! preprocessing.showRandomImages(trainImages, trainLabels)

    


    # initialize hyperparameters
    # TODO: init hyperparameters

    # create Yolo5 model

import pickle
import progressbar
import multiprocessing as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging


def relabelImages(imgs, func):
    """
    Relabel images to be used in training
    return a list of labels
    """
    # init the progress bar
    pbar = progressbar.ProgressBar(maxval=len(imgs), widgets=["Relabeling images: ", progressbar.Percentage(
    ), " ", progressbar.Bar(), " ", progressbar.ETA()])

    # multiprocessing to speed up the process of relabeling
    pool = mp.Pool()
    labels = pool.imap(func, imgs)

    # wait for the process to finish
    pbar.start()
    while labels._length == None:
        pbar.update(labels._index)
    pbar.finish()
    pool.close()

    return list(labels)


def relabelOpenCV(img, debug=False, args=None):
    """
    give a label to an image
    """
    # convert to HSV
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # parameters for initial masks
    if args is None:
        topRangeRed, topRangeYellow, bottomRangeYellow, bottomRangeGreen = (15, 20, 35, 33)
    else:
        topRangeRed, topRangeYellow, bottomRangeYellow, bottomRangeGreen = args

    # create three masks for each color (red, green, yellow)
    red_mask = cv2.bitwise_or(cv2.inRange(hsvImg, (0, 50, 50), (15, 255, 255)), cv2.inRange(
        hsvImg, (154, 50, 50), (190, 255, 255)))
    yellow_mask = cv2.inRange(hsvImg, (9, 80, 80), (45, 255, 255))
    green_mask = cv2.inRange(hsvImg, (45, 60, 60), (105, 255, 255))

    # remove the top and bottom parts of the image from the masks using cutoffs
    red_mask[topRangeRed+1:, :] = 0
    yellow_mask[:topRangeYellow, :] = 0
    yellow_mask[bottomRangeYellow+1:, :] = 0
    green_mask[:bottomRangeGreen, :] = 0

    if debug:
        # show masks with title and original image and hsv image
        _, ax = plt.subplots(1, 5)

        ax[0].imshow(hsvImg)
        ax[0].set_title('HSV')
        ax[0].axis('off')
        ax[0].axis("tight")
        ax[0].axis("image")

        ax[1].imshow(img)
        ax[1].set_title('Original')
        ax[1].axis('off')
        ax[1].axis("tight")
        ax[1].axis("image")

        ax[2].imshow(red_mask, cmap='gray')
        ax[2].set_title('Red ' + str(np.sum(red_mask)))
        ax[2].axis("tight")
        ax[2].axis("image")
        ax[2].yaxis.set_major_locator(plt.FixedLocator([topRangeRed]))
        ax[2].xaxis.set_major_locator(plt.NullLocator())


        ax[3].imshow(green_mask, cmap='gray')
        ax[3].set_title('Green ' + str(np.sum(green_mask)))
        ax[3].axis("tight")
        ax[3].axis("image")
        ax[3].yaxis.set_major_locator(plt.FixedLocator([bottomRangeGreen]))
        ax[3].xaxis.set_major_locator(plt.NullLocator())
        
        ax[4].imshow(yellow_mask, cmap='gray')
        ax[4].set_title('Yellow ' + str(np.sum(yellow_mask)))
        ax[4].axis("tight")
        ax[4].axis("image")
        ax[4].yaxis.set_major_locator(plt.FixedLocator([topRangeYellow, bottomRangeYellow]))
        ax[4].xaxis.set_major_locator(plt.NullLocator())

        plt.show()

    # check which mask has the most pixels and if red is at the top, yellow in the middle and green at the bottom
    if np.sum(green_mask) > np.sum(red_mask) and np.sum(green_mask) > np.sum(yellow_mask):
        return "go"
    elif np.sum(yellow_mask) > np.sum(green_mask) and np.sum(yellow_mask) > np.sum(red_mask):
        return "warning"
    else:
        return "stop"


def evaluateParameters(imgs, labels):
    """
    evaluate best parameters for relabel function
    """
    # create log
    logging.basicConfig(filename='evaluateParamsLog.log', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # best known parameters
    best = (16, 20, 31, 37)
    # best known score
    bestScore = len(imgs)*0.95
    # number of variations
    var = 10

    # progress bar
    pbar = progressbar.ProgressBar(maxval=var, widgets=["Evaluating parameters: ", progressbar.Percentage(
    ), " ", progressbar.Bar(), " ", progressbar.ETA()]).start()

    # make slight variations of the parameters and evaluate them

    for topRangeRed in range(best[0]-var//2, best[0]+var//2):
        pbar.update(topRangeRed - (best[0]-var//2))
        for topRangeYellow in range(best[1]-var//2, best[1]+var//2):
            for bottomRangeYellow in range(best[2]-var//2, best[2]+var//2):
                for bottomRangeGreen in range(best[3]-var//2, best[3]+var//2):
                    # check if the ranges are correct
                    if topRangeYellow < bottomRangeYellow:
                        correct = 0
                        for index, img in enumerate(imgs):
                            # check if the label is correct
                            if relabelOpenCV(img, args=(topRangeRed, topRangeYellow, bottomRangeYellow, bottomRangeGreen)) == labels[index]:
                                correct += 1
                            if correct + (len(imgs) - index) < bestScore:
                                break
                        # check if the score is better than the best score
                        if correct >= bestScore:
                            bestScore = correct
                            newBest = (topRangeRed, topRangeYellow,
                                       bottomRangeYellow, bottomRangeGreen)

                            # log results
                            logging.info(
                                f"New best score: {bestScore} with parameters: {newBest}")
    pbar.finish()
    return newBest


def compareLabels(recalculatedLabels, originalLabels, imgs, func, debug=False):
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
            # If debug is true, show the image with the wrong label
            if debug:
                print("Correct label: ", target,
                      " Incorrect label: ", recalculatedLabels[i])
                func(imgs[i], debug=True)

            if recalculatedLabels[i] == 'stop':
                stopIncorrect += 1
            else:
                goIncorrect += 1

    # print how many labels were correct and the accuracy
    print("Correct: ", correct, "/", len(originalLabels),
          " Accuracy: {:.2f}".format(100*correct/len(originalLabels)), "%")
    # print the percentages of stop and go labels that were incorrect
    print("Will stop when supposed to go accuracy: {:.2f}".format(100*stopIncorrect/len(
        originalLabels))+"%", " Go incorrect: {:.2f}".format(100*goIncorrect/len(originalLabels))+"%")


if __name__ == "__main__":
    # get pickled train data
    with open('dataset/LISA_train_dataset.pickle', 'rb') as f:
        trainImages, trainLabels = pickle.load(f)
        f.close()

    # get pickled test data
    with open('dataset/LISA_test_dataset.pickle', 'rb') as f:
        testImages, testLabels = pickle.load(f)
        f.close()

    # get mit test data
    with open('dataset/MIT_test_dataset.pickle', 'rb') as f:
        mitImages, mitLabels = pickle.load(f)
        f.close()

    # evaluate best parameters
    # ! print(evaluateParameters(trainImages, trainLabels))

    # recreate labels using openCV
    selfSupravisedTrainLables = relabelImages(trainImages, relabelOpenCV)
    selfSupravisedTestLables = relabelImages(testImages, relabelOpenCV)
    selfSupravisedMitLables = relabelImages(mitImages, relabelOpenCV)

    # save the new labels
    with open('dataset/selfSupravisedTrainLabels.pickle', 'wb') as f:
        pickle.dump(selfSupravisedTrainLables, f)
        f.close()
    with open('dataset/selfSupravisedTestLabels.pickle', 'wb') as f:
        pickle.dump(selfSupravisedTestLables, f)
        f.close()

    # compare with original labels
    compareLabels(selfSupravisedTrainLables, trainLabels, trainImages, relabelOpenCV)
    compareLabels(selfSupravisedTestLables, testLabels, testImages, relabelOpenCV)
    compareLabels(selfSupravisedMitLables, mitLabels, mitImages, relabelOpenCV)
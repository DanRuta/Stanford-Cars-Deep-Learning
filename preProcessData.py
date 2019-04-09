import os
import sys
import shutil
import argparse
import math
import random

import numpy as np
import cv2 as cv
# https://forums.fast.ai/t/image-normalization-in-pytorch/7534/6



def preprocess(overwriteData=False, skipWritingFiles=False):

    totalImages = 0
    meanR = 0
    meanG = 0
    meanB = 0
    stdR = 0
    stdG = 0
    stdB = 0

    labels = []
    classes = []
    splits = [[50,25,25], [60,20,20], [70,15,15], [80,10,10]]

    # Read the classifications
    with open("data_pre/classes.txt") as f:
        classes = [x for x in f.read().split("\n") if x is not ""]


    if not skipWritingFiles:

        # Check if the output has already been created, and skip this
        if os.path.exists("data") is False:
            os.mkdir("data")
            os.mkdir("data/augmentations")
            prepareDataFolder(classes, splits)

        elif os.path.exists("./data/augmentations"):
            if not overwriteData:
                print("Data has already been pre-processed.")
                return
            else:
                # Clear out the folder
                print("Clearing out old data")
                prepareDataFolder(classes, splits)


        print("Pre-processing data")

    with open("data_pre/labels.txt") as f:
        labels = [x.split(",") for x in f.read().split("\n")]

        for l in range(len(labels)):

            sys.stdout.write("\r- Processing... {}%".format(math.floor(l / len(labels)*100)) )
            sys.stdout.flush()

            # Take care of end of blank lines, usually at the end of the file
            if len(labels[l])==1:
                continue

            imgName = "{}.jpg".format(labels[l][0].zfill(6))
            imagePath = "./data_pre/images/{}".format(imgName)
            classification = labels[l][1]

            image = cv.imread(imagePath)
            totalImages += 1
            b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]


            meanR += np.mean(r)
            meanG += np.mean(g)
            meanB += np.mean(b)
            stdR += np.std(r)
            stdG += np.std(g)
            stdB += np.std(b)


            if skipWritingFiles:
                continue

            newImg = cv.resize(image, (int(224*1.5), int(224*1.5)))
            for [training, validation, test] in splits:

                r = random.random()*100
                splitFolder = "{}-{}-{}".format(training, validation, test)
                folder = ""

                # Move to training
                if r<training:
                    folder = "train"
                # Move to Test
                elif r>training+validation:
                    folder = "test"
                # Move to Validation
                else:
                    folder = "val"


                cv.imwrite("./data/{}/{}/{}/{}".format(splitFolder, folder, classification, imgName), newImg)

                # Copy each image 10 times, to allow plenty of different transformations to happen, to each one
                if folder=="train":
                    for i in range(0, 10):
                        cv.imwrite("./data/augmentations/{}/{}/{}/{}_{}".format(splitFolder, folder, classification, i, imgName), newImg)
                else:
                    # The validation and test data does not get augmented, and therefore only 1 copy of each image is needed
                    cv.imwrite("./data/augmentations/{}/{}/{}/{}_{}".format(splitFolder, folder, classification, i, imgName), newImg)

    print("\nFinished pre-processing data ({} images)".format(totalImages))

    meanR /= totalImages
    meanG /= totalImages
    meanB /= totalImages
    stdR /= totalImages
    stdG /= totalImages
    stdB /= totalImages

    print("meanR: {} ({})".format(meanR, meanR/255))
    print("meanG: {} ({})".format(meanG, meanG/255))
    print("meanB: {} ({})".format(meanB, meanB/255))
    print("stdR: {} ({})".format(stdR, stdR/255))
    print("stdG: {} ({})".format(stdG, stdG/255))
    print("stdB: {} ({})".format(stdB, stdB/255))


def prepareDataFolder (classes, splits):

    shutil.rmtree("./data")
    os.mkdir("data")
    os.mkdir("data/augmentations")

    print("Preparing folder")
    for [training, validation, test] in splits:
        os.mkdir("data/{}-{}-{}".format(training, validation, test))
        os.mkdir("data/{}-{}-{}/train".format(training, validation, test))
        os.mkdir("data/{}-{}-{}/val".format(training, validation, test))
        os.mkdir("data/{}-{}-{}/test".format(training, validation, test))

        os.mkdir("data/augmentations/{}-{}-{}".format(training, validation, test))
        os.mkdir("data/augmentations/{}-{}-{}/train".format(training, validation, test))
        os.mkdir("data/augmentations/{}-{}-{}/val".format(training, validation, test))
        os.mkdir("data/augmentations/{}-{}-{}/test".format(training, validation, test))

        # Class folders
        for c in range(0, len(classes)):
            os.mkdir("data/{}-{}-{}/train/{}".format(training, validation, test, c+1))
            os.mkdir("data/{}-{}-{}/val/{}".format(training, validation, test, c+1))
            os.mkdir("data/{}-{}-{}/test/{}".format(training, validation, test, c+1))

            os.mkdir("data/augmentations/{}-{}-{}/train/{}".format(training, validation, test, c+1))
            os.mkdir("data/augmentations/{}-{}-{}/val/{}".format(training, validation, test, c+1))
            os.mkdir("data/augmentations/{}-{}-{}/test/{}".format(training, validation, test, c+1))


if __name__=="__main__":
    print("Starting...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--o', default=False, type=bool)
    parser.add_argument('--sw', default=False, type=bool, help="Skip file writing")
    args = parser.parse_args()

    preprocess(args.o, args.sw)
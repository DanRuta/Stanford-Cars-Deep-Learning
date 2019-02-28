import os
import sys
import shutil
import argparse
import math
import random


def preprocess(overwriteData=False):

    labels = []
    classes = []
    splits = [[50,25,25], [60,20,20], [70,15,15], [80,10,10]]

    # Read the classifications
    with open("data_pre/classes.txt") as f:
        classes = [x for x in f.read().split("\n") if x is not ""]


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


            for [training, validation, test] in splits:
                # print([training, validation, test])

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

                shutil.copy(imagePath, "./data/{}/{}/{}/{}".format(splitFolder, folder, classification, imgName))

                # Copy each image 10 times, to allow plenty of different transformations to happen, to each one
                for i in range(0, 10):
                    shutil.copy(imagePath, "./data/augmentations/{}/{}/{}/{}_{}".format(splitFolder, folder, classification, i, imgName))

    print("\nFinished pre-processing data")


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
    args = parser.parse_args()

    preprocess(args.o)
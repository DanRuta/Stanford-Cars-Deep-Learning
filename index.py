import argparse
import logging
from logging.handlers import RotatingFileHandler

from preProcessData import preprocess
from model import Model

def main(model):
    log("Start")

    # Initializing logger
    logger = initLogger()

    # Read the classifications
    classes = []
    with open("data_pre/classes.txt") as f:
        classes = [line for line in f.read().split("\n") if line is not ""]

    splits = [[50,25,25], [60,20,20], [70,15,15], [80,10,10]]
    learningRates = [0.01, 0.05, 0.1]
    # L1s = [0.001, 0.005, 0.01]
    L2s = [0.0005, 0.001, 0.005]
    dropouts = [1, 0.95, 0.9]
    optimisers = ["SGD", "RMSprop", "Adam"]
    # ? momentum ?

    # Random Crop, Flip, Colour Jitter, Rotation
    noAugmentations = [False, False, False]
    someAugmentations = [True, True, False]
    allAugmentations = [True, True, True]

    for split in splits:
        for augs in [noAugmentations, someAugmentations, allAugmentations]:
            for lr in learningRates:
                for l2 in L2s:
                    for optim in optimisers:
                        trainParams(model, classes, split, augs, lr, l2, optim)
                # for l1 in L1s:
                    # for drop in dropouts:
                        # trainParams(classes, split, augs, lr, l1, l2, drop)
                        # trainParams(model, classes, split, augs, lr, l2, drop)
                    # trainParams(model, classes, split, augs, lr, l2)




def trainParams (model, classes, split, augs, lr, l2, optim):

    [train, val, test] = splits

    log("\n-----")
    log("Split: {}-{}-{}\tAugmentations: {}\tLearning Rate: {}\tRegularization: {}\tOptimiser: {}".format(train, val, test, augs, lr, l2, optim))
    log("-----\n")

    augType = 0 if augs[0] is False else (2 if augs[3] else 1)
    name = "{}-{}-{}_A{}_LR{}_R{}_O-{}".format(train, val, test, lr, augType, l2, optim)

    log("\nNew {} model".format(model))
    model = Model(model, False, classes, name)
    model.loadData(split, augs)
    model.train(lr, weight_decay=l2, optim=optim, epochs=3)
    confMatrix, testLoss = model.test()


    log("\nPretrained {} model".format(model))
    model = Model(model, True, classes, name)
    model.loadData(split, augs)
    model.train(lr, weight_decay=l2, optim=optim, epochs=3)
    confMatrix, testLoss = model.test()



def initLogger ():
    logger = logging.getLogger("trainingLog")
    logger.setLevel(logging.DEBUG)
    fh = RotatingFileHandler('{}/training.log'.format(os.path.dirname(os.path.realpath(__file__))), maxBytes=5*1024*1024, backupCount=2)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("New session")

    return logger

def log (string):
    print(string)
    logger.info(string)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="alexnet")
    args = parser.parse_args()

    main(args.m)
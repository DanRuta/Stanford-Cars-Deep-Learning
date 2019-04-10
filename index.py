import os
import traceback
import argparse
import logging
from logging.handlers import RotatingFileHandler

from tensorboardX import SummaryWriter

from preProcessData import preprocess
from model import Model


class Logger(object):

    def __init__(self):
        super(Logger, self).__init__()
        self.logger = logging.getLogger("trainingLog")
        self.logger.setLevel(logging.DEBUG)
        fh = RotatingFileHandler("{}/training.log".format(os.path.dirname(os.path.realpath(__file__))), maxBytes=5*1024*1024, backupCount=2)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info("\n\n\n\nNew session")

        self.writer = SummaryWriter()

    def log(self, message=""):
        print(message)
        self.logger.info(message)



def main(architecture):

    # Initializing logger
    logger = Logger()
    logger.log("Start")

    # Read the classifications
    classes = []
    with open("data_pre/classes.txt") as f:
        classes = [line for line in f.read().split("\n") if line is not ""]

    splits = [[50,25,25], [70,15,15], [80,10,10]]
    learningRates = [0.01, 0.005, 0.001]
    L2s = [0.005, 0.001, 0.0005]
    optimisers = ["SGD", "RMSprop", "Adam"]

    # Flip, Colour Jitter, Rotation
    noAugmentations = [False, False, False]
    allAugmentations = [True, True, True]

    for split in splits:
        for augs in [noAugmentations, allAugmentations]:
            for lr in learningRates:
                for l2 in L2s:
                    trainParams(logger, architecture, classes, split, augs, lr, l2, "SGD")

    # trainParams(logger, architecture, classes, [50,25,25], [False, False, False], 0.01, 0.001, "SGD")
    logger.writer.close()



def trainParams (logger, architecture, classes, split, augs, lr, l2, optim):

    [train, val, test] = split

    logger.log("\n-----")
    logger.log("Split: {}-{}-{}\tAugmentations: {}\tLearning Rate: {}\tRegularization: {}\tOptimiser: {}".format(train, val, test, augs, lr, l2, optim))
    logger.log("-----\n")

    augType = 0 if augs[0] is False else 1
    name = "{}_new_{}-{}-{}_A{}_LR{}_R{}_O-{}".format(architecture, train, val, test, augType, lr, l2, optim)

    logger.log("\nNew {} model".format(architecture))
    try:
        model = Model(architecture, False, classes, name, logger.writer)
        model.setLogger(logger)
        model.loadData(split, augs)
        model.train(lr, weight_decay=l2, optimFn=optim, epochs=50)
        model.test()
    except:
        logger.log("{} ERROR {}".format("="*10, "="*10))
        logger.log(traceback.format_exc())


    name = "{}_pt_{}-{}-{}_A{}_LR{}_R{}_O-{}".format(architecture, train, val, test, augType, lr, l2, optim)
    logger.log("\nPretrained {} model".format(architecture))
    try:
        model = Model(architecture, True, classes, name, logger.writer)
        model.setLogger(logger)
        model.loadData(split, augs)
        model.train(lr, weight_decay=l2, optimFn=optim, epochs=50)
        model.test()
    except:
        logger.log("{} ERROR {}".format("="*10, "="*10))
        logger.log(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="alexnet") # alexnet, vgg19, <googlenet/some resnet>
    args = parser.parse_args()

    main(args.m)
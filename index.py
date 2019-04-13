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

    splits = [[80,10,10], [50,25,25], [70,15,15]]
    learningRates = [0.001, 0.01, 0.005]
    L2s = [0.0005, 0.001, 0.005]
    optimisers = ["SGD", "RMSprop", "Adam"]

    # Flip, Colour Jitter, Rotation
    noAugmentations = [False, False, False]
    allAugmentations = [True, True, True]

    for l2 in L2s:
        for split in splits:
            for augs in [allAugmentations, noAugmentations]:
                for lr in learningRates:
                    trainParams(logger, False, architecture, classes, split, augs, lr, l2, "SGD")
                    trainParams(logger, True, architecture, classes, split, augs, lr, l2, "SGD")

    logger.writer.close()



def trainParams (logger, isPretrained, architecture, classes, split, augs, lr, l2, optim):

    [train, val, test] = split

    logger.log("\n===============================")
    logger.log("Split: {}-{}-{}\tAugmentations: {}\tLearning Rate: {}\tRegularization: {}\tOptimiser: {}\tPre-trained: {}".format(train, val, test, augs, lr, l2, optim, isPretrained))
    logger.log("===============================\n")

    augType = 0 if augs[0] is False else 1

    name = "{}_{}_{}-{}-{}_A{}_LR{}_R{}_O-{}".format(architecture, "pt" if isPretrained else "new", train, val, test, augType, lr, l2, optim)
    try:
        model = Model(architecture, isPretrained, classes, name, logger.writer)
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
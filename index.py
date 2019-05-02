import os
import traceback
import argparse
import logging
from logging.handlers import RotatingFileHandler

from tensorboardX import SummaryWriter

from preProcessData import preprocess
from model import Model
from ensemble import EnsembleVoter


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
    L2s = [0, 0.0005, 0.001]
    optimisers = ["SGD", "RMSprop", "Adam"]

    # Flip, Colour Jitter, Rotation
    noAugmentations = [False, False, False]
    allAugmentations = [True, True, True]

    # for l2 in L2s:
    #     for split in splits:
    #         for augs in [allAugmentations, noAugmentations]:
    #             for lr in learningRates:
    #                 trainParams(logger, False, architecture, classes, split, augs, lr, l2, "SGD")
    #                 trainParams(logger, True, architecture, classes, split, augs, lr, l2, "SGD")

    runs = [
        ["resnet50", [80,10,10], allAugmentations, 0.001, 0, True],
        ["resnet50", [80,10,10], noAugmentations, 0.005, 0, False],
        ["resnet50", [70,15,15], noAugmentations, 0.005, 0, False],
        ["resnet50", [70,15,15], allAugmentations, 0.005, 0, False],
    ]

    for run in runs:
        trainParams(logger, run[5], run[0], classes, run[1], run[2], run[3], run[4], "SGD")

    logger.writer.close()




def doEnsemble():

    print("Doing ensemble")

    # Initializing logger
    logger = Logger()
    logger.log("Start")

    # Read the classifications
    classes = []
    with open("data_pre/classes.txt") as f:
        classes = [line for line in f.read().split("\n") if line is not ""]

    architecture = "resnet50"
    noAugmentations = [False, False, False]
    allAugmentations = [True, True, True]

    bestConfiguration = [[80,10,10], allAugmentations, 0.001, 0, True]
    train, val, test = bestConfiguration[0]


    lr, l2, optim = bestConfiguration[2], bestConfiguration[3], "SGD"
    augType = 0 if bestConfiguration[1][0] is False else 1
    name = "{}-ensemble-{}-{}-{}_A{}_LR{}_R{}_O-{}".format(architecture, train, val, test, augType, lr, l2, optim)


    logger.log(name)
    logger.log("\n===============================")
    logger.log("Ensemble for 1 Model")
    logger.log("===============================\n")

    models = []
    newModel = Model(architecture, bestConfiguration[4], classes, writer=logger.writer)
    newModel.setLogger(logger)
    newModel.loadData(bestConfiguration[0], bestConfiguration[1])
    newModel.train(bestConfiguration[2], weight_decay=bestConfiguration[3], optimFn="SGD", epochs=100)
    models.append(newModel)


    for mI in range(0, 30):

        if mI!=0:
            logger.log("\n===============================")
            logger.log("Ensemble for {} Models".format(len(models)+1))
            logger.log("===============================\n")

            newModel = Model(architecture, bestConfiguration[4], classes, writer=logger.writer)
            newModel.setLogger(logger)
            newModel.loadData(bestConfiguration[0], bestConfiguration[1])

            logger.log("\nTraining new model\n")
            newModel.train(bestConfiguration[2], weight_decay=bestConfiguration[3], optimFn="SGD", epochs=100)

            models.append(newModel)


        logger.log("\nTesting Ensemble\n")
        voter = EnsembleVoter(models, classes, name, writer=logger.writer)
        voter.setLogger(logger)
        voter.loadData(bestConfiguration[0])
        voter.test()

    logger.writer.close()




def trainParams (logger, isPretrained, architecture, classes, split, augs, lr, l2, optim):

    [train, val, test] = split

    logger.log("\n===============================")
    logger.log("Architecture: {}\tSplit: {}-{}-{}\tAugmentations: {}\tLearning Rate: {}\tRegularization: {}\tOptimiser: {}\tPre-trained: {}".format(architecture, train, val, test, augs, lr, l2, optim, isPretrained))
    logger.log("===============================\n")

    augType = 0 if augs[0] is False else 1

    name = "{}_{}_{}-{}-{}_A{}_LR{}_R{}_O-{}".format(architecture, "pt" if isPretrained else "new", train, val, test, augType, lr, l2, optim)
    try:
        model = Model(architecture, isPretrained, classes, name, logger.writer)
        model.setLogger(logger)
        model.loadData(split, augs)
        model.train(lr, weight_decay=l2, optimFn=optim, epochs=200)
        model.test()
    except KeyboardInterrupt:
        raise
    except:
        logger.log("{} ERROR {}".format("="*10, "="*10))
        logger.log(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="alexnet") # alexnet, vgg19, <googlenet/some resnet>
    parser.add_argument("--e", default=False, help="Do ensembles or not")
    args = parser.parse_args()

    if args.e:
        doEnsemble()
    else:
        main(args.m)


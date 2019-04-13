
import os
import argparse
import copy
import math

import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
# Confusion Matrix
from torchnet import meter
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

from sklearn.metrics import classification_report

from tensorboardX import SummaryWriter

class Model():

    def __init__ (self, architecture, pretrained, classes, name=None, writer=None):

        self.classes = classes
        self.architecture = architecture
        self.batch_size = 4
        self.model = getattr(models, architecture)(pretrained=pretrained)
        self.name = name or architecture
        self.writer = writer or SummaryWriter()

        self.model.cuda()
        self.model.eval()

        # Freeze training for all pretrained layers
        if pretrained:
            for param in self.model.features.parameters():
                param.require_grad = False

        # Remove the last layer, and add one with <classes> length
        num_features = self.model.classifier[6].in_features
        lastLayers = list(self.model.classifier.children())[:-1]
        lastLayers.extend([nn.Linear(num_features, len(self.classes))])

        # Overwrite with new topology
        self.model.classifier = nn.Sequential(*lastLayers)
        self.criterion = nn.CrossEntropyLoss()
        self.bestEpoch = 1
        self.totalTrainingIts = 0
        self.totalValidationIts = 0
        self.totalTestingIts = 0

        self.validationPatience = 5

        self.log = print

    def train (self, lr=0.001, weight_decay=0, optimFn="SGD", epochs=1):

        self.log("\nTraining...")

        self.model.cuda()
        self.optimizer = getattr(optim, optimFn)(self.model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)

        # Back up current weights
        self.bestModelWeights = copy.deepcopy(self.model.state_dict())
        self.validationPatienceCounter = 0
        self.bestLoss = math.inf
        correct = 0
        total = 0


        for epoch in range(epochs):

            self.log("Epoch {}/{}".format(epoch+1, epochs))
            lastLoss = None # For TensorboardX
            self.trainingLoss = 0
            self.validationLoss = 0
            self.trainingAccuracy = 0
            self.validationAccuracy = 0

            self.model.train()

            for i, data in enumerate(self.dataLoaders[0]):
                if i % 25 == 0 or i == int(self.numTrainingSamples/self.batch_size)-1:
                    print("\rTraining iteration: {}/{}".format(i*self.batch_size, self.numTrainingSamples), end='', flush=True)
                    if i>0:
                        self.writer.add_scalar("{}/1.training/loss".format(self.name), lastLoss/self.batch_size, self.totalTrainingIts)
                        self.writer.add_scalar("{}/1.training/accuracy".format(self.name), self.trainingAccuracy*100, self.totalTrainingIts)

                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                if math.isnan(loss):
                    raise("Loss is nan")

                loss.backward()
                self.optimizer.step()

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                lastLoss = loss.data.item()
                self.trainingLoss += loss.data.item()
                self.trainingAccuracy = correct/total

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
                self.totalTrainingIts += self.batch_size

            self.model.eval()

            # Do the same for validation
            if self.numValidationSamples>0:
                self.log()
                self.validate()
                self.writer.add_scalar("{}/2.validation/perEpochLoss".format(self.name), self.validationLoss / self.numValidationSamples, epoch)

            self.log()
            self.log("Epoch {} result: ".format(epoch+1))
            self.log("Average loss (train): {:.4f}".format(self.trainingLoss / self.numTrainingSamples))
            self.log("Average accuracy (train): {:.4f}%".format(self.trainingAccuracy*100))
            self.log("Average loss (val): {:.4f}".format(self.validationLoss / self.numValidationSamples))
            self.log("Average accuracy (val): {:.4f}%".format(self.validationAccuracy*100))
            self.writer.add_scalar("{}/1.training/perEpochLoss".format(self.name), self.trainingLoss / self.numTrainingSamples, epoch)

            if self.validationLoss / self.numValidationSamples < self.bestLoss:
                self.log("Deep copying new best model. (Loss of {}, over {})".format(self.validationLoss / self.numValidationSamples, self.bestLoss))
                self.bestLoss = self.validationLoss / self.numValidationSamples
                self.bestModelWeights = copy.deepcopy(self.model.state_dict())
                self.bestEpoch = epoch + 1
                self.validationPatienceCounter = 0
            else:
                self.validationPatienceCounter += 1
                self.log("Validation loss not improved. Now: {:.4f}, Old: {:.4f}\tPatience: {}/{}".format(self.validationLoss, self.bestLoss, self.validationPatienceCounter, self.validationPatience))

                if self.validationPatienceCounter >= self.validationPatience:
                    self.log("Validation loss has not improved for {} epochs. Stopping training, and saving the best weights.".format(self.validationPatience))
                    break


            self.log("-" * 10)
            self.log()

        torch.save(self.bestModelWeights, "checkpoints/{}__{}-{}.pt".format(self.bestLoss, self.name, self.bestEpoch))


    def validate (self):

        lastLoss = None # For TensorboardX
        correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.dataLoaders[1]):

                if i % 25 == 0 or i == int(self.numValidationSamples/self.batch_size)-1:
                    print("\rValidation iteration: {}/{}".format(i*self.batch_size, self.numValidationSamples), end='', flush=True)
                    if i>0:
                        self.writer.add_scalar("{}/2.validation/loss".format(self.name), lastLoss/self.batch_size, self.totalValidationIts)
                        self.writer.add_scalar("{}/2.validation/accuracy".format(self.name), self.validationAccuracy*100, self.totalValidationIts)

                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                lastLoss = loss.data.item()
                self.validationLoss += loss.data.item()
                self.validationAccuracy = correct/total

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
                self.totalValidationIts += self.batch_size


    def test (self):

        # For sklearn classification report
        self.correctLabels = []
        self.predictedLabels = []

        self.testLoss = 0
        self.testAccuracy = 0
        self.top5 = 0

        lastLoss = None
        correct = 0
        total = 0

        # Confusion Matrix
        confusionMatrix = meter.ConfusionMeter(len(self.classes))

        self.log("\nTesting...")
        with torch.no_grad():

            for i, data in enumerate(self.dataLoaders[2]):
                if i % 25 == 0 or i == int(self.numTestSamples/self.batch_size)-1:
                    print("\rTest iteration: {}/{}".format(i*self.batch_size, self.numTestSamples), end='', flush=True)
                    if i>0:
                        self.writer.add_scalar("{}/3.test/loss".format(self.name), lastLoss, self.totalTestingIts)
                        self.writer.add_scalar("{}/3.test/top5".format(self.name), self.top5/(i*self.batch_size), self.totalTestingIts)
                        self.writer.add_scalar("{}/3.test/accuracy".format(self.name), self.testAccuracy*100, self.totalTestingIts)

                self.model.eval()

                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                outputs = self.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # Aggregate the total loss and accuracy values
                lastLoss = loss.item()
                self.testLoss += loss.item()

                self.testAccuracy = correct/total
                self.top5 += self.getTopKAccuracy(outputs, labels, 5)

                # Add to confusion matrix
                if len(outputs) == self.batch_size:
                    confusionMatrix.add(outputs.data.squeeze(), labels.type(torch.LongTensor))

                # Collect data for the classification report
                labelVals = np.array(labels.data.cpu())
                predVals = np.array(preds.data.cpu())


                for b in range(min(len(labelVals), len(predVals))):
                    self.correctLabels.append(labelVals[b])
                    self.predictedLabels.append(predVals[b])


                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
                self.totalTestingIts += self.batch_size

            self.log()

        loss = self.testLoss / self.numTestSamples
        top5 = self.top5 / self.numTestSamples
        self.log("Average loss (test): {:.4f}".format(loss))
        self.log("Average accuracy (test): {:.4f}%".format(self.testAccuracy*100))
        self.log("Top 5 accuracy (test): {:.4f}%".format(top5 * 100))

        self.getMetrics(confusionMatrix, loss, self.testAccuracy, top5)

        return loss, self.testAccuracy, top5


    def loadData (self, split, augmentations):
        # Look in the augmentations sub-directory, if any augmentations have been selected
        augPath = augmentations[0] or augmentations[1] or augmentations[2]
        augPath = "augmentations/" if augPath else ""
        split = "{}-{}-{}".format(split[0], split[1], split[2])

        traindir = os.path.join(os.getcwd(), "data/{}{}/train".format(augPath, split))
        valdir = os.path.join(os.getcwd(), "data/{}{}/val".format(augPath, split))
        testdir = os.path.join(os.getcwd(), "data/{}{}/test".format(augPath, split))
        # Expects as input normalized x * H * W images, where H and W have to be at least 224
        # Also needs mean and std as follows:
        normalize = transforms.Normalize(mean=[0.46989, 0.45955, 0.45476], std=[0.266161, 0.265055, 0.269770])

        # Build up the required augmentations
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        augmentationTransforms = []

        # The data input must be of this dimensions
        augmentationTransforms.append(transforms.RandomResizedCrop(224))

        if augmentations[0]:
            augmentationTransforms.append(transforms.RandomHorizontalFlip())
        if augmentations[1]:
            augmentationTransforms.append(transforms.ColorJitter(0.5, 0.5, 0.5, 0.1))
        if augmentations[2]:
            augmentationTransforms.append(transforms.RandomRotation(45))
        augmentationTransforms.append(transforms.ToTensor())
        augmentationTransforms.append(normalize)

        datasetGroups = []
        datasetGroups.append(datasets.ImageFolder(traindir, transforms.Compose(augmentationTransforms)))
        datasetGroups.append(datasets.ImageFolder(valdir, transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize])))
        datasetGroups.append(datasets.ImageFolder(testdir, transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize])))

        self.numTrainingSamples = len(datasetGroups[0])
        self.numValidationSamples = len(datasetGroups[1])
        self.numTestSamples = len(datasetGroups[2])

        self.dataLoaders = []
        self.dataLoaders.append(torch.utils.data.DataLoader(datasetGroups[0], batch_size=self.batch_size, shuffle=True, num_workers=8))
        self.dataLoaders.append(torch.utils.data.DataLoader(datasetGroups[1], batch_size=self.batch_size, shuffle=False, num_workers=8))
        self.dataLoaders.append(torch.utils.data.DataLoader(datasetGroups[2], batch_size=self.batch_size, shuffle=False, num_workers=8))


    def getMetrics(self, conf, loss, accuracy, top5):

        # Classification report
        report = classification_report(self.correctLabels, self.predictedLabels, target_names=self.classes)

        with open("checkpoints/{}__{}-{}.txt".format(self.bestLoss, self.name, self.bestEpoch), "w+") as f:
            f.write("Model: {}\tEpochs: {}\tLoss: {}\tAccuracy: {:.4f}%\tTop 5: {:.4f}%\n".format(self.name, self.bestEpoch, loss, accuracy*100, top5*100))
            f.write(report)

        # Confusion Matrix

        # Too many classes break the confusion matrix plot, so manually render a heatmap using OpenCV instead
        if len(self.classes) > 20:
            conf.normalized = True
            self.plotConfHeatmap(conf.value(), "checkpoints/{}__{}-{}_n".format(self.bestLoss, self.name, self.bestEpoch))
        else:
            self.plotConfMatrix(conf.value(), "checkpoints/{}__{}-{}".format(self.bestLoss, self.name, self.bestEpoch))
            conf.normalized = True
            self.plotConfMatrix(conf.value(), "checkpoints/{}__{}-{}_n".format(self.bestLoss, self.name, self.bestEpoch))

    # https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    def getTopKAccuracy (self, output, labels, k):

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t() # Transpose?

        # Set one hot vector for correct predictions, within the ordered list of highest predictions
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        # Check if the correct predictions are within the top k highest predictions, and sum them up
        correct_k = correct[:k].view(-1).float().sum(0)

        return correct_k.item()


    def plotConfHeatmap (self, conf, path):
        largestVal = -math.inf

        for row in conf:
            for col in row:
                if col>largestVal:
                    largestVal = col

        outputFrame = []
        for row in conf:
            outputFrameRow = []
            for col in row:
                outputFrameRow.append(col/largestVal*255)
            outputFrame.append(outputFrameRow)

        outputFrame = cv.UMat(np.uint8(np.array(outputFrame, dtype=int)))
        cv.imwrite("{}.jpg".format(path), cv.resize(outputFrame, (1600,1600), interpolation=cv.INTER_NEAREST ))


    def plotConfMatrix (self, conf, path):
        df_cm = pd.DataFrame(conf, self.classes, self.classes)
        plt.figure(figsize=(15,15))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, fmt="g")
        plt.savefig("{}.jpg".format(path))
        sn.reset_orig()


    def loadCheckpoint (self, path):
        self.log("Loading checkpoint at {}".format(path))
        self.model.load_state_dict(torch.load(path))

    def setLogger (self, logger):
        self.log = logger.log


if __name__ == "__main__":

    # Read the classifications
    classes = []
    with open("data_pre/classes.txt") as f:
        classes = [line for line in f.read().split("\n") if line is not ""]

    parser = argparse.ArgumentParser()
    parser.add_argument("--e", default=5, type=int, help="Epochs to train for")
    parser.add_argument("--m", default="alexnet")
    parser.add_argument("--pt", help="Checkpoint loading")
    args = parser.parse_args()

    model = Model(args.m, True, classes)
    print(model.model)

    if args.pt is not None:
        model.loadCheckpoint(args.pt)

    model.loadData([70, 15, 15], [True, True, True])
    model.train(0.001, 0, "SGD", epochs=args.e)
    model.test()
    model.writer.close()



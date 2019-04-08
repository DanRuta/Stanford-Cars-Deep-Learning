
import os
import argparse
import copy
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


class Model():

    def __init__ (self, architecture, pretrained, classes, name=None):

        self.classes = classes
        self.architecture = architecture
        self.batch_size = 4
        self.model = getattr(models, architecture)(pretrained=pretrained)
        self.name = name or architecture

        self.model.cuda()
        self.model.train(False)
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

    def train (self, lr=0.001, weight_decay=0, optimFn="SGD", epochs=1):

        print("training...")

        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.optimizer = getattr(optim, optimFn)(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.model.cuda()

        # Back up current weights
        self.bestModelWeights = copy.deepcopy(self.model.state_dict())
        bestAccuracy = 0.0
        self.bestEpoch = 1


        for epoch in range(epochs):

            print("Epoch {}/{}".format(epoch+1, epochs))
            self.trainingLoss = 0
            self.validationLoss = 0
            self.trainingAccuracy = 0
            self.validationAccuracy = 0

            self.model.train(True)

            for i, data in enumerate(self.dataLoaders[0]):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, int(self.numTrainingSamples/self.batch_size)), end='', flush=True)

                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                self.trainingLoss += loss.data.item()
                self.trainingAccuracy += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            self.model.train(False)
            self.model.eval()

            # Do the same for validation
            if self.numValidationSamples>0:
                print()
                self.validate()

            print()
            print("Epoch {} result: ".format(epoch+1))
            print("Average loss (train): {:.4f}".format(self.trainingLoss / self.numTrainingSamples))
            print("Average accuracy (train): {:.4f}".format(self.trainingAccuracy / self.numTrainingSamples))
            print("Average loss (val): {:.4f}".format(self.validationLoss / self.numValidationSamples))
            print("Average accuracy (val): {:.4f}".format(self.validationAccuracy / self.numValidationSamples))
            print("-" * 10)
            print()

            if self.validationAccuracy / self.numValidationSamples > bestAccuracy:
                print("Deep copying new best model")
                bestAccuracy = self.validationAccuracy / self.numValidationSamples
                self.bestModelWeights = copy.deepcopy(self.model.state_dict())
                self.bestEpoch = epoch + 1


        torch.save(self.model.state_dict(), "checkpoints/{}-{}.pt".format(self.name, self.bestEpoch))


    def validate (self):
        for i, data in enumerate(self.dataLoaders[1]):

            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, int(self.numValidationSamples/self.batch_size)), end='', flush=True)

            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            self.validationLoss += loss.data.item()
            self.validationAccuracy += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()


    def test (self):
        # Confusion Matrix
        confusionMatrix = meter.ConfusionMeter(len(self.classes))

        print("Testing model")
        with torch.no_grad():

            self.testLoss = 0
            self.testAccuracy = 0

            for i, data in enumerate(self.dataLoaders[2]):
                if i % 100 == 0:
                    print("\rTest batch {}/{}".format(i, int(self.numTestSamples/self.batch_size)), end='', flush=True)

                self.model.train(False)
                self.model.cuda()
                self.model.eval()

                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                outputs = self.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                # Add to confusion matrix
                confusionMatrix.add(outputs.data.squeeze(), labels.type(torch.LongTensor))

                self.testLoss += loss.data.item()
                self.testAccuracy += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            print()
            print("Average loss (test): {:.4f}".format(self.testLoss / self.numTestSamples))
            print("Average accuracy (test): {:.4f}".format(self.testAccuracy / self.numTestSamples))

        return self.plotConfMatrix(confusionMatrix.conf), self.testLoss / self.numTestSamples


    def loadData (self, split, augmentations):
        # Look in the augmentations sub-directory, if any augmentations have been selected
        augPath = augmentations[0] or augmentations[1] or augmentations[2] or augmentations[3]
        augPath = "augmentations/" if augPath else ""
        split = "{}-{}-{}".format(split[0], split[1], split[2])

        traindir = os.path.join(os.getcwd(), "data/{}{}/train".format(augPath, split))
        valdir = os.path.join(os.getcwd(), "data/{}{}/val".format(augPath, split))
        testdir = os.path.join(os.getcwd(), "data/{}{}/test".format(augPath, split))
        # Expects as input normalized x * H * W images, where H and W have to be at least 224
        # Also needs mean and std as follows:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Build up the required augmentations
        augmentationTransforms = []

        if augmentations[0]:
            augmentationTransforms.append(transforms.RandomResizedCrop(224))
        else:
            # The data input must be of this dimensions
            augmentationTransforms.append(transforms.Resize(224))

        if augmentations[1]:
            augmentationTransforms.append(transforms.RandomHorizontalFlip())
        if augmentations[2]:
            augmentationTransforms.append(transforms.ColorJitter(0.5, 0.5, 0.5, 0.1))
        if augmentations[3]:
            augmentationTransforms.append(transforms.RandomRotation(45))
        augmentationTransforms.append(transforms.ToTensor())
        augmentationTransforms.append(normalize)

        datasetGroups = []
        datasetGroups.append(datasets.ImageFolder(traindir, transforms.Compose(augmentationTransforms)))
        datasetGroups.append(datasets.ImageFolder(valdir, transforms.Compose([transforms.ToTensor(), normalize])))
        datasetGroups.append(datasets.ImageFolder(testdir, transforms.Compose([transforms.ToTensor(), normalize])))

        self.numTrainingSamples = len(datasetGroups[0])
        self.numValidationSamples = len(datasetGroups[1])
        self.numTestSamples = len(datasetGroups[2])

        self.dataLoaders = []
        self.dataLoaders.append(torch.utils.data.DataLoader(datasetGroups[0], batch_size=self.batch_size, shuffle=True, num_workers=8))
        self.dataLoaders.append(torch.utils.data.DataLoader(datasetGroups[1], batch_size=self.batch_size, shuffle=True, num_workers=8))
        self.dataLoaders.append(torch.utils.data.DataLoader(datasetGroups[2], batch_size=self.batch_size, shuffle=False, num_workers=8))


    def plotConfMatrix (self, conf):
        df_cm = pd.DataFrame(conf, self.classes, self.classes)
        plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt="g")
        plt.show()
        plt.savefig("checkpoints/{}-{}.png".format(self.name, self.bestEpoch))
        return plt


if __name__ == "__main__":

    # Read the classifications
    classes = []
    with open("data_pre/classes.txt") as f:
        classes = [line for line in f.read().split("\n") if line is not ""]

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="alexnet")
    args = parser.parse_args()

    model = Model(args.m, True, classes)
    model.loadData([50, 25, 25], [True, True, True, True])
    model.train(0.001, 0, "SGD", 1)
    model.test()

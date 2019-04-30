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

from model import Model


class EnsembleVoter(Model):

    def __init__ (self, models, classes, name=None, writer=None):
        self.classes = classes
        self.batch_size = 4
        self.models = models
        self.name = name or "Ensemble of {} models".format(len(self.models))
        self.writer = writer or SummaryWriter()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        for model in self.models:
            model.model.to(torch.device(self.device))
            model.model.eval()


        self.criterion = nn.CrossEntropyLoss()
        self.totalTestingIts = 0

        self.log = print

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):

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
            for i, data in enumerate(self.dataLoaders[0]):
                if i % 25 == 0 or i == int(self.numTestSamples/self.batch_size)-1:
                    print("\rEnsemble iteration: {}/{}".format(i*self.batch_size, self.numTestSamples), end='', flush=True)
                    if i>0:
                        self.writer.add_scalar("{}/loss".format(self.name), lastLoss, self.totalTestingIts+(self.totalTestingIts*(len(self.models)-1)))
                        self.writer.add_scalar("{}/top5".format(self.name), self.top5/(i*self.batch_size), self.totalTestingIts+(self.totalTestingIts*(len(self.models)-1)))
                        self.writer.add_scalar("{}/accuracy".format(self.name), self.testAccuracy*100, self.totalTestingIts+(self.totalTestingIts*(len(self.models)-1)))

                inputs, labels = data
                inputs, labels = Variable(inputs.to(torch.device(self.device))), Variable(labels.to(torch.device(self.device)))

                total += labels.size(0)
                lastLoss = 0
                # votes = [] # Vote indices. Size: (noModels x batch_size)
                # Vote counts for classes. Size: (noModels x batchSize x noClasses)
                # maxClassVotes = np.zeros((len(self.models), self.batch_size, len(self.classes)))


                # Final output needs to be 1 class index per batch
                # Initially, each sub-item will be an array of votes (array length being number of models), before they get counted
                # ensembleVotes = [[] for i in range(self.batch_size)]
                ensembleVotes = [[] for i in range(len(labels))]

                # pre_bestVotes = np.zeros((self.batch_size, len(self.models)))
                # bestVotes = np.zeros((self.batch_size))
                for model in self.models:

                    outputs = model.model(inputs)

                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                    # Get the predicted indeces from each item in this batch size, and add it to the respective sub-array
                    preds = preds.data
                    for b in range(len(preds)):
                        ensembleVotes[b].append(preds[b])



                    # correct += (preds == labels).sum().item()

                    # Aggregate the total loss and accuracy values
                    lastLoss += loss.item()
                    self.testLoss += loss.item()

                    del outputs, preds




                # Get the maximum occuring integer in each of the ensembleVotes array's sub-arrays, to end up with a batch_size sized array
                # ensembleVotes = [max(set(subArr), key=subArr.count) for subArr in ensembleVotes if len(subArr)>0]
                maxEnsembleVotes = [max(set(subArr), key=subArr.count) for subArr in ensembleVotes if len(subArr)>0]
                # print("test")
                # print(test)

                # maxEnsembleVotes = []
                # for subArr in ensembleVotes:
                #     try:
                #         # print(subArr)
                #         if len(subArr) > 0:
                #             maxEnsembleVotes.append(max(set(subArr), key=subArr.count))
                #         else:
                #             maxEnsembleVotes.append([])
                #         # ensembleVotes = maxEnsembleVotes

                #     except:
                #         print("subArr")
                #         # print(ensembleVotes)
                #         print(subArr)
                #         raise

                # Create a fake 'output'-like data structure from the ensembleVotes, encoded as one-hot vectors, for use in other metrics.
                # It should still be fine, as only the top value is important, which corresponds to a 1, in the one-hot vector
                # ensembleOutput = np.zeros((self.batch_size, len(self.classes)))
                ensembleOutput = np.zeros((len(maxEnsembleVotes), len(self.classes)))
                # for b in range(self.batch_size):
                for b in range(len(maxEnsembleVotes)):
                    if len(ensembleVotes[b]) > 0:
                        ensembleOutput[b][maxEnsembleVotes[b]] = 1
                    # try:
                    # except:
                    #     print("ensembleVotes")
                    #     print("b: {}".format(b))
                    #     print(ensembleVotes[b])
                    #     raise

                # Continue as normal, using ensembleVotes and ensembleOutput instead of a normal single model's batch set of predictions
                correct += (torch.Tensor(maxEnsembleVotes).long() == labels.cpu()).sum().item()

                self.testAccuracy = correct/total
                self.top5 += self.getTopKAccuracy(torch.Tensor(ensembleOutput).long(), labels.cpu(), 5)

                # Add to confusion matrix
                if len(ensembleOutput) == self.batch_size:
                    confusionMatrix.add(torch.Tensor(ensembleOutput).long().data.squeeze(), labels.type(torch.LongTensor).cpu())

                # Collect data for the classification report
                labelVals = np.array(labels.data.cpu())
                # predVals = np.array(maxEnsembleVotes.data.cpu())
                predVals = np.array(maxEnsembleVotes)

                for b in range(min(len(labelVals), len(predVals))):
                    self.correctLabels.append(labelVals[b])
                    self.predictedLabels.append(predVals[b].item())


                del inputs, labels
                torch.cuda.empty_cache()
                self.totalTestingIts += self.batch_size



                # for mv in range(len(votes)):
                #     modelVotes = votes[mv]

                #     for b in range(self.batch_size):
                #         vote = modelVotes[b]
                #         maxClassVotes[mv][b][vote] += 1


                # # Get an index value for the top vote, in each numClasses sized array, in each batch, for each model
                # voteIndeces = [[max(maxClassVotes[m][b]) for b in range(self.batch_size)] for m in range(len(self.models))]



                lastLoss /= len(self.models)
            self.log()


        loss = self.testLoss / self.numTestSamples
        top5 = self.top5 / self.numTestSamples
        self.log("Average loss (Ensemble): {:.4f}".format(loss))
        self.log("Average accuracy (Ensemble): {:.4f}%".format(self.testAccuracy*100))
        self.log("Top 5 accuracy (Ensemble): {:.4f}%".format(top5 * 100))

        print("DE THING {}".format(len(self.models)))
        self.writer.add_scalar("{}/averageLoss".format(self.name), loss, len(self.models))
        self.writer.add_scalar("{}/averageAccuracy".format(self.name), self.testAccuracy*100, len(self.models))
        self.writer.add_scalar("{}/averageTop5".format(self.name), top5 * 100, len(self.models))


        self.bestLoss = loss
        self.bestEpoch = "-"
        self.getMetrics(confusionMatrix, loss, self.testAccuracy, top5)

        return loss, self.testAccuracy, top5





    def loadData (self, split):

        split = "{}-{}-{}".format(split[0], split[1], split[2])
        testdir = os.path.join(os.getcwd(), "data/{}/test".format(split))

        # Expects as input normalized x * H * W images, where H and W have to be at least 224
        # Also needs mean and std as follows:
        normalize = transforms.Normalize(mean=[0.46989, 0.45955, 0.45476], std=[0.266161, 0.265055, 0.269770])

        # Build up the required augmentations
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        augmentationTransforms = []

        # The data input must be of this dimensions
        augmentationTransforms.append(transforms.RandomResizedCrop(224))
        augmentationTransforms.append(transforms.ToTensor())
        augmentationTransforms.append(normalize)

        datasetGroups = []
        datasetGroups.append(datasets.ImageFolder(testdir, transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize])))
        self.numTestSamples = len(datasetGroups[0])

        self.dataLoaders = []
        self.dataLoaders.append(torch.utils.data.DataLoader(datasetGroups[0], batch_size=self.batch_size, shuffle=False, num_workers=8))

    def loadCheckpoint(self):
        pass



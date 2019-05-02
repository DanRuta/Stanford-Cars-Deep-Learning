
# Stanford Cars Deep Learning

This repository contains the implementation of the *Image Processing and Deep Learning* assignment, from the 2018-2019 *Computer Vision, Robotics, and Machine Learning* course, at *University of Surrey*.


## Content

The assignment implementation covers the training and evaluation of a range of CNN architectures, hyper-parameters, data splits/augmentation, and training strategies. The dataset chosen was an augmented version of the Stanford Cars dataset, where 4 additional classes were added, rounding the count to 200.


Evaluation is performed with and without transfer learning, through use of the following metrics:

* Confusion matrix/heatmap
* Loss
* Top-1 Accuracy
* Top-5 Accuracy
* sklearn classification report

TensorboardX plots are generated from the following:
* Training accuracy
* Training loss
* Training loss (per epoch)
* Validation accuracy
* Validation loss
* Validation loss (per epoch)
* Test accuracy
* Test loss
* Test Top-5


Early stopping is employed using the patience policy, with the best stored weights being reverted. Checkpoints are saved/loaded in/from the `checkpoints` directory, along with the confusion matrix, and `.txt` report containing the above metrics.


Ensemble training is performed using maximum vote, and TensorboardX plots are generated for accuracy and loss metrics, per ensemble size.

A rotating file logger is used, logging the training process to the `training.log` file.



## Files

The repository consists of the following files:

- *preProcessData.py* - This file reads in the dataset, gets data statistics, resizes, and splits the data into several pre-defined training/validation/test splits
- *model.py* - This file contains the bulk of the model code, where the model training and evaluation takes place
- *ensemble.py* - This file contains a wrapper class for managing and evaluating trained model ensembles
- *index.py* - The main entry point for experiments

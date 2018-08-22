#
# required arguments list
#

# python train.py data_dir --save_dir save_directory
# python train.py data_dir --arch "vgg13"
# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# python train.py data_dir --gpu

# Import libraries
import io
import json
import torch
import argparse
import requests
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from io import BytesIO
from torch import optim
from collections import OrderedDict

if __name__ == "__main__":
    #-----------------#
    #    CLI setup    #
    #-----------------#

    ### Default values ###
    # args.learning_rate = 0.001
    # args.gpu = True
    # args.epochs = 20
    # args.save_dir = "./"
    # args.arch = "None"

    ### Required inputs ###
    # args.hidden_units : list values i.e 1024 256

    parser = argparse.ArgumentParser(description="Train a predictive model using images")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate of the optimizer")
    parser.add_argument("-u", "--hidden_units", type=int, nargs='+', required=True, help="number of hidden units in the neural network")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of iterations to perform during the training")
    parser.add_argument("-g", "--gpu", help="use GPU", default=True, action="store_true")
    parser.add_argument("-a", "--arch", type=str, default="None", help="Specify the pretrained model to use. Supporting models: vgg16, vgg19")
    parser.add_argument("-s", "--save_dir", type=str, default="./", help="directory to save the trained model")
    args = parser.parse_args()

    print("Training will be performed with the setting as followed:")
    print("GPU: " + str(args.gpu))
    print("epochs: " + str(args.epochs))
    print("Learning Rate: " + str(args.learning_rate))
    print("Pretrained model: " + args.arch)
    print("Hidden units: " + str(args.hidden_units))

    # Data Preparation
    dataloader_training, dataloader_validation, dataloader_testing = data_preparation()

    # Download predefined model

    # Define Classifier

    # classifier needs to be assigned to the model before moving it to GPU

    # Move model to GPU

    # Optimizer should defined after moving model to GPU

    # Train

    # Test

    # Save model


def data_preparation():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    transforms_training = transforms.Compose([
                                            transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))
                                            ])


    transforms_validation = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))
                                            ])

    transforms_testing = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))
                                            ])
    training_batch = 64
    validation_testing_batch = 32

    # TODO: Load the datasets with ImageFolder
    dataset_training = torchvision.datasets.ImageFolder(train_dir, transform = transforms_training)
    dataset_validation = torchvision.datasets.ImageFolder(valid_dir, transform = transforms_validation)
    dataset_testing = torchvision.datasets.ImageFolder(test_dir, transform = transforms_testing)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=training_batch, shuffle=True, drop_last=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=validation_testing_batch)
    dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=validation_testing_batch)

    return dataloader_training, dataloader_validation, dataloader_testing



def validation(model, dataloader_validation, criterion):
    valid_correct = 0
    valid_loss = 0

    for images, labels in (iter(dataloader_validation)):
        images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass
        output = model.forward(images)
        loss = criterion(output, labels)

        # Track loss
        valid_loss += loss.item()
        _, output_v = torch.max(output.data, 1)

        # Track Accuracy
        valid_correct += (output_v == labels).sum()
    
    return valid_loss/len(dataloader_validation), (100*valid_correct)/(validation_testing_batch*len(dataloader_validation))
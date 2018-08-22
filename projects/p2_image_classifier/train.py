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
    # CLI setup
    parser = argparse.ArgumentParser(description="Train a predictive model using images")
    parser.add_argument("-l", "--learning_rate", type=float, help="learning rate of the optimizer")
    parser.add_argument("-u", "--hidden_units", type=int, help="number of hidden units in the neural network")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of iterations to perform during the training")
    parser.add_argument("-g", "--gpu", help="use GPU", default=True, action="store_true")
    parser.add_argument("-a", "--arch", type=str, help="specify the pretrained model to use")
    parser.add_argument("-s", "--save_dir", type=str, help="directory to save the trained model")
    args = parser.parse_args()

    # Default values
    # args.gpu = True or False
    # args.epochs = 20
    # args

    print(args.gpu)
    print(args.epochs)
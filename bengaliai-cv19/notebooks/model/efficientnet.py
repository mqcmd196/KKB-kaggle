#importするmoduleの一覧
import numpy as np
import pandas as pd
#import cupy as cp
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import cv2
# from tqdm.auto import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, utils
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
import torch.optim as optim
import torchvision

import PIL
# from torchsummary import summary
import gc

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.name = "efficientnet"

        self.efficient_imagenet = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Linear(1000, 512)
        self.head_root = nn.Linear(512, 168) # + softmax
        self.head_vowel = nn.Linear(512, 11) # + softmax
        self.head_consonant = nn.Linear(512, 7) # + softmax

    def forward(self, x):
        x = self.efficient_imagenet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        head_root = self.head_root(x)
        head_vowel = self.head_vowel(x)
        head_consonant = self.head_consonant(x)

        return head_root, head_vowel, head_consonant

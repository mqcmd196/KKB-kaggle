#importするmoduleの一覧
import numpy as np
import pandas as pd
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
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import PIL
# from torchsummary import summary
import gc

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.resnet152 = models.resnet152(pretrained = True)
        self.fc = nn.Linear(1024,512)
        self.head_root = nn.Linear(512, 168) # + softmax
        self.head_vowel = nn.Linear(512, 11) # + softmax
        self.head_consonant = nn.Linear(512, 7) # + softmax
    
    def forward(self, x):
        x = self.resnet152(x)
        x = self.fc(x)
        head_root = self.head_root(x)
        head_vowel = self.head_vowel(x)
        head_consonant = self.head_consonant(x)
        return head_root, head_vowel, head_consonant # not sure..
    
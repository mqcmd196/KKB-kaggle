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
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.b1 = nn.BatchNorm2d(32, momentum=0.15)     
        self.pool = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv5_dropout = nn.Dropout(p=0.3)

        self.conv6 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)
        self.b2 = nn.BatchNorm2d(64, momentum=0.15)
        # self.pool = nn.MaxPool2d(2,2) same as upper pool
        self.conv10 = nn.Conv2d(64, 64, 5, padding=2)
        self.b3 = nn.BatchNorm2d(64, momentum=0.15)
        self.conv10_dropout = nn.Dropout(p=0.3)

        self.conv11 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv13 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv14 = nn.Conv2d(128, 128, 3, padding=1)
        self.b4 = nn.BatchNorm2d(128, momentum=0.15)
        # self.pool = nn.MaxPool2d(2,2) same as upper pool
        self.conv15 = nn.Conv2d(128, 128, 5, padding=2)
        self.b5 = nn.BatchNorm2d(128, momentum=0.15)
        self.conv15_dropout = nn.Dropout(p=0.3)

        self.conv16 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv17 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv18 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv19 = nn.Conv2d(256, 256, 3, padding=1)
        self.b6 = nn.BatchNorm2d(256, momentum=0.15)
        # self.pool = nn.MaxPool2d(2,2) same as upper pool
        self.conv20 = nn.Conv2d(256, 256, 5, padding=2)
        self.b7 = nn.BatchNorm2d(256, momentum=0.15)
        self.conv20_dropout = nn.Dropout(p=0.3)

        self.dense1 = nn.Linear(256*4*4, 1024)
        self.dense1_dropout = nn.Dropout(p=0.3) # +relu
        self.dense2 = nn.Linear(1024, 512) # +relu
        

        self.head_root = nn.Linear(512, 168) # + softmax
        self.head_vowel = nn.Linear(512, 11) # + softmax
        self.head_consonant = nn.Linear(512, 7) # + softmax

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.b1(self.conv4(x))))
        x = F.relu(self.conv5(x))
        x = self.conv5_dropout(x)
            
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(F.relu(self.b2(self.conv9(x))))
        x = F.relu(self.b3(self.conv10(x)))
        x = self.conv10_dropout(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool(F.relu(self.b4(self.conv14(x))))
        x = F.relu(self.b5(self.conv15(x)))
        x = self.conv15_dropout(x)

        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = self.pool(F.relu(self.b6(self.conv19(x))))
        x = F.relu(self.b7(self.conv20(x)))
        x = self.conv20_dropout(x)

        x = x.view(-1, 256*4*4)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        
        
        
        head_root = self.head_root(x)
        head_vowel = self.head_vowel(x)
        head_consonant = self.head_consonant(x)

        return head_root, head_vowel, head_consonant # not sure..

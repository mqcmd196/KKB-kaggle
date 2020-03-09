#importするmoduleの一覧
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import cv2
# from tqdm.auto import tqdm
import copy
import datetime
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

def load_model(model,optimizer,model_path):
    device = torch.device('cpu')
    checkpoint = torch.load(model_path,map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print("loaded from {}".format(model_path))
    return model, optimizer, epoch

def save_model(model,optimizer,model_dir, epoch):
    print("---saving model of epoch {}---".format(epoch))
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=9)
    now = now + diff
    path = model_dir+"/"+model.name+'_epoch'+str(epoch)+'_'+now.strftime('%Y-%m-%d_%H-%M-%S')+'.pth'
    #torch.save(model.state_dict(), model_dir + '/'+model.name + f'_{epoch}epoch' +'.pth')
    torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()
          },path)
    print('save finished')
    return 0

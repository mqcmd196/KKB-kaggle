import numpy as np
import pandas as pd
import cv2
import copy
import gc

import torch
from torchvision import transforms, utils
import torchvision
import PIL


### 文字を切り抜いて、希望のサイズにresizeする関数（前処理）

def resize(X, out_height=64, out_width=64, in_height=137, in_width=236):
    print('Resizing raw image... / 前処理実行中…')
    resized = np.zeros((len(X), out_height * out_width))

    for i in range(len(X)):
        image = X.iloc[[i]].values.reshape(in_height, in_width)
        _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        left = 1000
        right = -1
        top = 1000
        bottom = -1

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            left = min(x, left)
            right = max(x+w, right)
            top = min(y, top)
            bottom = max(y+h, bottom)

        roi = image[top:bottom, left:right]
        resized_roi = cv2.resize(roi, (out_height, out_width),interpolation=cv2.INTER_AREA)
        resized[i] = resized_roi.reshape(-1)

    return resized


def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e



### Augmentation のための transform

# トリミング
transform_crop64 = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomResizedCrop((64,64), scale=(0.80, 0.90)),
                                transforms.ToTensor()
])
transform_crop224 = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomResizedCrop((224,224), scale=(0.80, 0.90)),
                                transforms.ToTensor()
])

# 回転
rotate_left = transforms.RandomAffine((-20, -10), fillcolor=255, resample=PIL.Image.BILINEAR)
rotate_right = transforms.RandomAffine((10, 20), fillcolor=255, resample=PIL.Image.BILINEAR)
transform_rotate = transforms.Compose([
                                       transforms.ToPILImage(),
                                       transforms.RandomChoice([rotate_left, rotate_right]),
                                       transforms.ToTensor()
])

# ガウスノイズ
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        with_noise = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.max(torch.min(with_noise, torch.tensor(1.)), torch.tensor(0.))
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transform_noise = AddGaussianNoise(0., 0.01)

# 何もしない、という transform（恒等写像）
class IdentityTransform(object):
    def __call__(self, tensor):
        return tensor

transform_none = IdentityTransform() 


### PyTorch式のデータセットクラスを定義

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, transform=None, enable_3d=False, H=64, W=64):
        self.transform = transform
        self.X = X
        self.Y = Y
        self.enable_3d = enable_3d
        self.H = H 
        self.W = W 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        if self.enable_3d:
            out_data = cv2.resize(self.X[idx].reshape(self.H,self.W), (224, 224),interpolation=cv2.INTER_AREA)
            out_data = out_data.reshape(224, 224, 1).astype(np.uint8)
            out_data = cv2.cvtColor(out_data, cv2.COLOR_GRAY2RGB)
            out_data = np.transpose(out_data, (2,0,1)) / 255.0
            out_data = out_data.reshape(3,224,224) 
        
        else:
            out_data = self.X[idx].reshape(1,self.H,self.W) / 255.0
        
        out_data = torch.tensor(out_data, dtype=torch.float)

        root_label = torch.tensor(np.argmax(self.Y[0][idx]), dtype=torch.long)
        vowel_label = torch.tensor(np.argmax(self.Y[1][idx]), dtype=torch.long)
        cons_label = torch.tensor(np.argmax(self.Y[2][idx]), dtype=torch.long)

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, root_label, vowel_label, cons_label


class TransformDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        out_data, root_label, vowel_label, cons_label = self.dataset[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, root_label, vowel_label, cons_label


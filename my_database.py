import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from io import BytesIO 
import os
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms

import make_db
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class ImageTransform():
    def __init__(self,resize,mean,std):
        self.data_transform = {
            'train':transforms.Compose([
                transforms.RandomResizedCrop(
                    resize,scale=(0.5, 1.0)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ]),
            'val':transforms.Compose([
                #transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.RandomResizedCrop(
                    resize,scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ]),
            'test':transforms.Compose([
                #transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.RandomResizedCrop(
                    resize,scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])}
    def __call__(self,img,phase):
        #phasel'train'or'val'で前処理のモードを指定する
        return self.data_transform[phase](img)

def make_datapath_list(phase):
    train_data=make_db.make(phase='train')
    test_data=make_db.make(phase='test')
    val_data=make_db.make(phase='val')
    path_list = []
    if phase=='train':
        path_list=train_data
        return path_list
    elif phase=='test':
        path_list=test_data
        return path_list
    if phase=='val':
        path_list= val_data
        return path_list





# image_file_path='../fashiondata_tng/photos/100/10037428289-10037428289_400.jpg'
# img=Image.open(image_file_path)
# plt.imshow(img)
# plt.show()
# img=Image.open(image_file_path)
# size=224
# mean=(0.485,0.456,0.406)
# std=(0.229,0.224,0.225)

# transform=ImageTransform(size,mean,std)
# img_transformed=transform(img,phase='train')

# img_transformed=img_transformed.numpy().transpose((1,2,0))
# img_transformed=np.clip(img_transformed,0,1)
# plt.imshow(img_transformed)
# plt.show()

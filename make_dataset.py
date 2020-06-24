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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
import make_db
import my_database


class FashionDataset(data.Dataset):
    def __init__(self,file_list,transform,phase):
        self.file_list=file_list['img_name']
        self.label=file_list['season']
        self.transform=transform
        self.phase=phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        img_path=self.file_list[index]
        label = self.label[index]
        img=Image.open(img_path)
        img = img.convert('RGB')
        img_transformed=self.transform(
            img,self.phase)
        return img_transformed,label

def main_move():
    size=224
    mean=(0.485,0.456,0.406)
    std=(0.229,0.224,0.225)

    train_list=my_database.make_datapath_list(phase='train')
    val_list=my_database.make_datapath_list(phase='val')
    test_list = my_database.make_datapath_list(phase='test')
    train_dataset= FashionDataset(
        file_list=train_list,transform=my_database.ImageTransform(size,mean,std),phase='train')
    val_dataset= FashionDataset(
        file_list=val_list,transform=my_database.ImageTransform(size,mean,std),phase='val')
    test_dataset= FashionDataset(
        file_list=test_list,transform=my_database.ImageTransform(size,mean,std),phase='')
    #print(train_dataset.label)
    #index=0
    # print(train_dataset.__getitem__(index)[0].size())
    # print(train_dataset.__getitem__(index)[1])


    batch_size=32
    train_dataloader=torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size,shuffle=True
    )
    val_dataloader=torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size,shuffle=False
    )
    test_dataloader=torch.utils.data.DataLoader(
        test_dataset,batch_size=batch_size,shuffle=False
    )
    dataloaders_dict={"train": train_dataloader,"val":val_dataloader,"test":test_dataloader}
    batch_iterator=iter(dataloaders_dict['train'])
    inputs,labels=next(
        batch_iterator
    )
    return dataloaders_dict

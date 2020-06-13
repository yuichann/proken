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
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
import make_db
import my_database

class FashionDataset(data.Dataset):
    def __init__(self,file_list,transform,phase='train'):
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

        # if self.phase =='train':
        #     label = self.label
        # elif self.phase=='val':
        #     label = self.label

        if label=='spring':
            label=0
        elif label=='summer':
            label=1
        elif label=='autumn':
            label=2
        elif label=='winter':
            label=3

        return img_transformed,label

def main_move():
    size=224
    mean=(0.485,0.456,0.406)
    std=(0.229,0.224,0.225)

    train_list=my_database.make_datapath_list(phase='train')
    val_list=my_database.make_datapath_list(phase='val')

    train_dataset= FashionDataset(
        file_list=train_list,transform=my_database.ImageTransform(size,mean,std),phase='train')
        
    val_dataset= FashionDataset(
        file_list=val_list,transform=my_database.ImageTransform(size,mean,std),phase='val')

    index=0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])


    batch_size=32
    train_dataloader=torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size,shuffle=True
    )
    val_dataloader=torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size,shuffle=False
    )

    dataloaders_dict={"train": train_dataloader,"val":val_dataloader}

    batch_iterator=iter(dataloaders_dict['train'])
    inputs,labels=next(
        batch_iterator
    )
    print(inputs.size())
    print(labels)

    return dataloaders_dict,inputs,labels

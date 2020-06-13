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
import make_dataset
import new_model


def make_optim():
    criterion=nn.CrossEntropyLoss()
    params_to_update=[]
    
    update_params_names=['classifier.6.weight','classifier.6.bias']
    for name, param in net.named_parameters():
        if name in update_params_names:
            param.requires_grad=True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = False

    print('--------')
    print(params_to_update)

    return params_to_update
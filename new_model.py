from torch import nn
from torch import optim
import make_dataset
from torchvision import models,transforms
import make_db
import my_database

def main():
    use_pretrained=True
    net=models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=4)
    net.train()
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

    return params_to_update,net



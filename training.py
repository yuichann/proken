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
import model
import new_model
from tqdm import tqdm
import cnn

def train_model(net,dataloaders_dict,criterion,optimizer,num_epochs):
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch+1,num_epochs))
        print('------------')
    
        for phase in ['train','val']:
            if phase=='train':
                net.train()
            else:
                net.eval()
            
            epoch_loss=0.0
            epoch_corrects=0

            if (epoch==0)and(phase=='train'):
                continue
            
            
            for inputs,labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    
                    outputs=net(inputs)
                    print(outputs.size())
                    loss = criterion(outputs,labels)
                    
                    _, preds = torch.max(outputs,1)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                    print(preds)
                    print(labels.data)
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)

            epoch_loss=epoch_loss/len(dataloaders_dict[phase].dataset)
            epoch_acc=epoch_corrects.double(
            )/len(dataloaders_dict[phase].dataset)
        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase,epoch_loss,epoch_acc
            ))

num_epochs=20
params_to_update, net = new_model.main()
net = cnn.Net()

criterion=nn.CrossEntropyLoss()

dataloaders_dict,inputs,labels = make_dataset.main_move()
optimizer=optim.SGD(params=params_to_update,lr=0.001,momentum=0.9)
train_model(net,dataloaders_dict,criterion,optimizer,num_epochs=num_epochs)


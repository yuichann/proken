import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import make_db
import make_dataset
import model
import new_model
import cnn
import matplotlib.pyplot as plt
from PIL import Image
import csv
import my_database
import pandas as pd

class CSVDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list,transform=None):
        super().__init__()  # 親クラスのtorch,Datasetの__init__()を呼び出している
        self.transform = transform
        self.file_list=file_list['img_name']
        self.label=file_list['season']

    def __getitem__(self, index):
        img_path=self.file_list[index]
        label = self.label[index]
        img=Image.open(img_path)
        img = img.convert('RGB')
        img_transformed=self.transform(
            img)
        return img_transformed,label

    def __len__(self):
        return len(self.file_list)

def transform():
    Dtransform = torchvision.transforms.Compose([
    torchvision.transforms.Resize( size=256 ),
    torchvision.transforms.RandomCrop( size=(224,224) ),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (1.,)),
    ])

    train_list=pd.read_csv('train_dataset.csv')
    val_list=pd.read_csv('val_dataset.csv')
    test_list =pd.read_csv('test_dataset.csv')

    Dtrain = CSVDataset(file_list=train_list, transform=Dtransform )
    Dvalid = CSVDataset( file_list=val_list,  transform=Dtransform )
    Dtest  = CSVDataset( file_list=test_list, transform=Dtransform )

    batchsize = 128
    train_loader = torch.utils.data.DataLoader( Dtrain, batch_size=batchsize, shuffle=True )
    valid_loader = torch.utils.data.DataLoader( Dvalid, batch_size=batchsize )
    test_loader  = torch.utils.data.DataLoader( Dtest,  batch_size=batchsize )
    dataloaders_dict = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
    return dataloaders_dict

def create_csv(epoch,train_loss,val_loss,optimizer,acc):
    note=pd.read_csv('note.csv', mode='a', header=False)
    return 0



def new_train(device,model,dataloaders_dict,optimizer):
    iter_counter = 0
    running_loss = 0
    train_loss = []
    valid_loss = []
    train_loader = dataloaders_dict['train']
    valid_loader = dataloaders_dict['val']
    test_loader = dataloaders_dict['test']
    for epoch in range(2):
        model.train() #modelをtrainモードに切り替え
        t = tqdm(train_loader, desc=("Epoch %d"%(epoch+1))) #tにはinput,outsputが入る
        for x,y in t:
            running_loss = 0
            optimizer.zero_grad() #購買の初期化
            x,y = x.to(device), y.to(device)
            pred = model(x) #predは予測値
            loss = F.cross_entropy(pred, y ) #交差エントロピー誤差の計算
            t.set_postfix( loss=loss.item() ) #バーに表示する値はloss
            running_loss += loss.item() #一回のinputで出るlossを足していく

            loss.backward()
            optimizer.step()
            iter_counter+=1
            if iter_counter > 30:
                train_loss.append( running_loss/30 )

                model.eval()
                with torch.no_grad():
                    iter_counter = 0
                    running_loss = 0
                    for x,y in valid_loader:
                        x, y = x.to(device), y.to(device)
                        pred = model(x)
                        loss = F.cross_entropy( pred, y )
                        running_loss += loss.item()
                valid_loss.append( running_loss/len(valid_loader) )
                model.train()
                running_loss = 0

    model.eval()
    sp_all=0
    su_all=0
    au_all=0
    win_all=0
    sp=0
    su=0
    au=0
    win=0
    model.eval()
    with torch.no_grad():
        correct = 0
        for x,y in test_loader:
            # x, y = x.to(device), y.to(device)
            pred = model(x).argmax( dim=1, keepdim=True )
            print(pred)
            print(y)
            correct += pred.eq( y.view_as(pred) ).sum().item()
        print( correct / len(test_loader.dataset) )


    with torch.no_grad():
        for x,y in t:
            pred = model(x).argmax( dim=1, keepdim=True )
            pred=pred.view(1,-1).squeeze() #predは予測値
            for i in range(len(pred)):
                if pred[i]==0:
                    sp_all+=1
                elif pred[i]==1:
                    su_all+=1
                elif pred[i]==2:
                    au_all+=1
                elif pred[i]==3:
                    win_all+=1
                if pred[i]==y[i]:
                    if pred[i]==0:
                        sp+=1
                    elif pred[i]==1:
                        su +=1
                    elif pred[i]==2:
                        au +=1
                    elif pred[i]==3:
                        win+=1
        print(sp/sp_all)
        print(su/su_all)
        print(au/au_all)
        print(win/win_all)

    print(train_loss)
    print(valid_loss)
    model_save_path='resnet18_Adagrad.pth'
    torch.save(model.to('cpu').state_dict(), model_save_path)
    plt.plot( train_loss, label='Train')
    plt.plot( valid_loss, label='Validation')
    plt.savefig("sgd_image.png")
    # plt.clf()

    # plt.plot(correct / len(test_loader.dataset) ,label='accuracy')
    # plt.savefig("acc_image.png")
    # plt.show()

dataloaders_dict=transform()
model = torchvision.models.resnet18( pretrained=True )
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
device = torch.device("cuda" if torch.cuda.device_count()>0 else "cpu")
model = model.to(device)
optimizer = torch.optim.SGD(
    [
        {"params": model.fc.parameters(), "lr": 1e-2},
    ],
    lr = 0.01,
)

new_train(device,model,dataloaders_dict,optimizer)
model_save_path='resnet18_SGD.pth'
torch.save(model.to('cpu').state_dict(), model_save_path)
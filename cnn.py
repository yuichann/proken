import torch
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,12,5,padding=2,stride=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(12,48,5,padding=2,stride=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(3 * 224* 224,16)
        self.fc2 = torch.nn.Linear( 16, 64)
        self.fc3 = torch.nn.Linear(64, 4)


    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        # print(x.size())
        x = self.pool1(x)
        # print(x.size())
        x = torch.nn.functional.relu(self.conv2(x))
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())
        x = x.view(-1,3 * 224* 224)
        # print(x.size())
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

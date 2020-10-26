import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()

        self.conv01 = nn.Conv2d(8, 32, 3,padding=1)#64
        self.conv02 = nn.Conv2d(32, 32, 3,padding=1)#64
        self.bn1=nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)#32

        self.conv03 = nn.Conv2d(32, 64, 3,padding=1)#32
        self.conv04 = nn.Conv2d(64, 64, 3,padding=1)#32
        self.bn2=nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)#16

        self.conv05 = nn.Conv2d(64, 128, 3,padding=1)#16
        self.conv06 = nn.Conv2d(128, 128, 3,padding=1)#16
        self.conv07 = nn.Conv2d(128, 128, 3,padding=1)#16
        self.bn3=nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)#8

        self.conv08 = nn.Conv2d(128, 256, 3,padding=1)#8
        self.conv09 = nn.Conv2d(256, 256, 3,padding=1)#8
        self.conv10 = nn.Conv2d(256, 256, 3,padding=1)#8
        self.bn4=nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)#4

        self.conv11 = nn.Conv2d(256, 256, 3,padding=1)#4
        self.conv12 = nn.Conv2d(256, 256, 3,padding=1)#4
        self.conv13 = nn.Conv2d(256, 256, 3,padding=1)#4
        self.bn5=nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)#2

       
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x):
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = self.pool1(self.bn1(x))

        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = self.pool2(self.bn2(x))

        x = F.relu(self.conv05(x))
        x = F.relu(self.conv06(x))
        x = F.relu(self.conv07(x))
        x = self.pool3(self.bn3(x))

        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(self.bn4(x))

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool5(self.bn5(x))


        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return self.softmax(x)   
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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

    def before_lstm(self, x):
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
        xx = F.relu(self.fc1(x))
        xx = self.dropout1(xx)
        xx = F.relu(self.fc2(xx))
        xx = self.dropout2(xx)
        xx = self.fc3(xx)
        return x, xx

class LSTMNet(nn.Module):

    def __init__(self, is_cuda, input_dim, output_dim, hidden_dim, n_layers,drop_prob=0.3):
        super(LSTMNet, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.is_cuda =is_cuda

    def forward(self, input, hidden):
        batch_size = input.size(0)
        # input shape : (batch size, sequence length, input dimension)
        # output shape : (batch size, sequence length, hidden_dim)
        lstm_out, hidden = self.lstm(input, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, hidden

    def init_hidden(self, batch_size):
        if self.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

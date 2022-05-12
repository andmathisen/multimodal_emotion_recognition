
import torch.nn as nn


class Net1D(nn.Module):
    def __init__(self):
        super(Net1D,self).__init__()

       
        self.conv1 = nn.Conv1d(11, 64,kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        
    
        self.conv2 = nn.Conv1d(64, 128,kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(128,256,kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2)


        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.25))
        self.fc = nn.Linear(64,7)


    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc(x)

 
        return x


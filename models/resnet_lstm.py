import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet,BasicBlock
from torchvision import models
import torch



class ResNet_LSTM(nn.Module):
    def __init__(self, params_model):
        super(ResNet_LSTM, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate = params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        features = params_model["features"]

        
        if features == "pretrained":
            baseModel = models.resnet50(pretrained=True)
            for param in baseModel.parameters():
                param.requires_grad = False
            num_features = baseModel.fc.in_features
            baseModel.fc = nn.Sequential(nn.Linear(num_features,300))
        else:
            baseModel = ResNet(BasicBlock, [2, 2, 2, 2])
            num_features = baseModel.linear.in_features
            baseModel.linear = nn.Sequential(nn.Linear(num_features,300))

        self.baseModel = baseModel
        self.dropout = nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(300, rnn_hidden_size, rnn_num_layers,batch_first=True,dropout=dr_rate)
        self.fc1 = nn.Linear(rnn_hidden_size, 128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        hidden = None
        for ii in range(ts):
            with torch.no_grad():
                y = self.baseModel((x[:, ii]))
            self.rnn.flatten_parameters()
            out, hidden = self.rnn(y.unsqueeze(0), hidden)
        

        out = self.fc1(out[-1,:,:])
        out = F.relu(out)
        out = self.fc2(out)

        return out
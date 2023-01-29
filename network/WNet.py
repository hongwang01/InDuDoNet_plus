import torch
import torch.nn as nn
import torch.nn.functional as F

class WNet(nn.Module):
    def __init__(self, n_filter=32):
        super(WNet, self).__init__()
        self.conv0 = nn.Conv2d(1,n_filter,3,1,1)
        self.dncnn_mid = self.dncnn_block(1, n_filter)
        self.convf = nn.Conv2d(n_filter,1,3,1,1)
    def dncnn_block(self, block_num, channel_num):
        layers = []
        for i in range(block_num):
            layers.append(nn.Conv2d(channel_num, channel_num, 3,1,1))
            layers.append(nn.BatchNorm2d(channel_num))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, input):
        x  = F.relu(self.conv0(input))
        x = self.dncnn_mid(x)
        w = self.convf(x)
        return  w


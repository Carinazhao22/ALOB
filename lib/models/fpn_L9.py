import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# baseline L9
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class network_9layers(nn.Module):
    def __init__(self, num_classes=512):
        super(network_9layers, self).__init__()
        self.bn2=nn.BatchNorm1d(512)
        self.features = nn.Sequential(
            mfm(3, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(128 * 7* 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(128 * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )
    def lightCNN(self,x, GRL):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return  x


class CNet(network_9layers):
  def __init__(self):
    super(CNet,self).__init__()
  def forward(self,x):
      cfmap=self.lightCNN(x,GRL=True)
      return cfmap

def LResNet50E_IR_Occ(common):
    model = CNet()
    return model
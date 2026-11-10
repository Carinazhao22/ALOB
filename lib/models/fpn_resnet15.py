import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

 
# baseline resnet15    
class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out



class Whole_Net(nn.Module):
    def __init__(self, block, layers, filter_list, is_gray=False):
        super(Whole_Net, self).__init__()
        if is_gray:
            self.conv1 = nn.Conv2d(
                1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(
                3, filter_list[0], kernel_size=3, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        
        self.bn2= nn.BatchNorm1d(512)
        self.prelu1 = nn.PReLU(filter_list[0])
         
        
        self.layer1 = self._make_layer(
            block, filter_list[0], filter_list[2], layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, filter_list[2], filter_list[4], layers[1], stride=2)
        
         
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 10* 8),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 10 * 8, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def resnet(self, x, GRL=False):
        
        x = self.conv1(x)  # 64 x 38 x 32
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)  # 128 x 19 x 16
        fmap=self.layer2(x1) # 512 x 10 x 8 

  
        fmap = self.fc(fmap.view(fmap.size(0), -1))
        return fmap

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class CNet(Whole_Net):
  def __init__(self, block,  layers, filter_list, is_gray=False):
    super(CNet,self).__init__(block, layers, filter_list, is_gray)
  def forward(self,x):
      cfmap=self.resnet(x,True)
      return cfmap


def LResNet50E_IR_Occ(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4]

 
    model = CNet(BlockIR, layers, filter_list, is_gray)
 
      
    return model
 
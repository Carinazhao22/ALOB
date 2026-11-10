import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from lib.models.cbam import CBAM


# twonets and twonets with-cbam
class Matrix(object):
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.num_samples = coordinates.size(dim=0)
        self.dimension = coordinates.size(dim=1)

    def __str__(self):
        return self.coordinates

    def plus(self, v):
        return self.coordinates + v.coordinates

    def minus(self, v):
        return self.coordinates - v.coordinates

    def magnitude(self):
        return torch.reshape(torch.sqrt(torch.sum(torch.square(self.coordinates), axis=-1)), (self.num_samples, 1))

    def normalized(self):   
        magnitude = self.magnitude()
        weight = torch.reshape(1.0/magnitude, (self.num_samples, 1))
        return self.coordinates * weight

    def component_parallel_to(self, basis):   
        u = basis.normalized()  # y/|y|
        weight = torch.reshape(torch.sum(
            self.coordinates * u, axis=-1), (self.num_samples, 1))
        return u * weight  # (x·y)/|y| · y/|y|

    def component_orthogonal_to(self, basis):   
        projection = self.component_parallel_to(basis)  # fp∗
        return self.coordinates - projection  # fp–fp∗


def NB_algorithm(original_feature,trivial_featue):
    original_feature = Matrix(original_feature)
    trivial_featue = Matrix(trivial_featue)
    d = original_feature.component_orthogonal_to(trivial_featue)
    d = Matrix(d)
    f = original_feature.component_parallel_to(d)
    return f


class FlipGradientBuilder(Function):

    # ignore warning
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        #　pass dict{'lambd' = lambd}
        ctx.lambd = lambd
        return x.view_as(x)
    @ staticmethod
    def backward(ctx, grad_output):
        # pass gradient* -c
        return grad_output * -ctx.lambd, None
 

class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match,use_cbam=False):
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
        
        if use_cbam:
            self.cbam=CBAM(planes,16)
        else:
            self.cbam=None

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

        if self.cbam is not None:
            out=self.cbam(out)
            out=self.prelu1(out)
        
        out += residual

        return out



class Whole_Net(nn.Module):
    def __init__(self, block, layers, filter_list, is_gray=False,att_type=None):
        super(Whole_Net, self).__init__()
        if is_gray:
            self.conv1 = nn.Conv2d(
                1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(
                3, filter_list[0], kernel_size=3, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.bn2= nn.BatchNorm1d(512)

        # change for ablation study
        # self.bn2= nn.BatchNorm1d(1024)
        
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(
            block, filter_list[0], filter_list[2], layers[0], stride=2,att_type=att_type)
        self.layer2 = self._make_layer(
            block, filter_list[2], filter_list[4], layers[1], stride=2,att_type=att_type)

        
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
    
    def _make_layer(self, block, inplanes, planes, blocks, stride,att_type=None):
        layers = []
        layers.append(block(inplanes, planes, stride, False,use_cbam=att_type=='CBAM'))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True,use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def resnet(self, x, GRL=False):
        
        x = self.conv1(x)  # 64 x 38 x 32
        # print("size: ")
        # print(x.size())
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x) # 128 x 19 x 16
        # print(x1.size())
        fmap=self.layer2(x1) # 512 x 10 x 8 
        # print(fmap.size())
        
        if GRL == True:
            fmap = FlipGradientBuilder.apply(fmap, 1.0)
        
        fmap = self.fc(fmap.view(fmap.size(0), -1))
            
        # print(fmap.size()) # [128, 512]
        return fmap

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class CNet(Whole_Net):
  def __init__(self, block,  layers, filter_list, is_gray=False,att_type=None):
    super(CNet,self).__init__(block, layers, filter_list, is_gray,att_type)
  def forward(self,x):
    
    cfmap=self.resnet(x,True)
      # abaltion study - twonets-resnet15-P-G(concat)
      # cfmap=self.resnet(x,False)
      
    return cfmap

class PNet(Whole_Net):
  def __init__(self,block, layers, filter_list, is_gray=False,att_type=None):
    super(PNet,self).__init__(block, layers, filter_list, is_gray,att_type)
  def forward(self,x,cfmap):
      
      fmap=self.resnet(x,False)
      fmap = NB_algorithm(fmap, cfmap)
      
      # ablation study for twonets-Resnet15-P (f1-f2)
      # fmap=fmap-cfmap
      # fmap = torch.cat((fmap,cfmap),-1)
      
      fmap = self.bn2(fmap)
      return fmap

def LResNet50E_IR_Occ(is_gray=False, num_mask=101, common=True,att_type=None):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4]

    if common:
      model = CNet(BlockIR, layers, filter_list, is_gray,att_type)
    else:
      model = PNet(BlockIR, layers, filter_list, is_gray,att_type)
      
    return model
 
 




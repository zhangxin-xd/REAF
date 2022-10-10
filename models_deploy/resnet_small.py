import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#from .res_utils import DownsampleA, DownsampleC, DownsampleD
import math,time

class conv_1(nn.Module):  

  def __init__(self, rate, index_conv1_value, bias_conv1_value):
    super(conv_1, self).__init__() 
    self.conv = nn.Conv2d(3, rate, kernel_size=3, stride=1, padding=1, bias=True) 
    self.index_conv1_value = index_conv1_value
    self.bias_conv1_value = bias_conv1_value

  def forward(self, x):   
    x = self.conv(x)  
    out_with_bias = self.bias_conv1_value.cuda() # 64
    out_with_bias_expand = out_with_bias.view(1, out_with_bias.shape[0],1,1) # 1 64 1 1
    a = torch.ones(x.shape[0], out_with_bias.shape[0], x.shape[2], x.shape[3]).cuda()
    b =  a * out_with_bias_expand # 128 16 32 32
    b.index_add_(1, self.index_conv1_value.cuda(), x)
    return b  

class DownsampleA(nn.Module):  

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__() 
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

  def forward(self, x):   
    x = self.avg(x)  
    return torch.cat((x, x.mul(0)), 1)  


class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, supply, index, index_block_i_conva_value, index_block_i_convb_value
      , bias_layer_i_value_a, bias_layer_i_value_b, stride=1, downsample=False):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

    self.conv_b = nn.Conv2d(supply, planes, kernel_size=3, stride=1, padding=1, bias=True)

    self.downsample = downsample
    self.inplanes = inplanes
    self.index = index
    self.supply = supply
    self.size = 0
    self.out = torch.autograd.Variable(torch.rand(128, self.supply, 16*32//self.supply, 16*32//self.supply))
    self.i = 0
    self.time = []
    self.sum = []
    self.index_block_i_conva_value = index_block_i_conva_value
    self.index_block_i_convb_value = index_block_i_convb_value
    self.bias_layer_i_value_a = bias_layer_i_value_a
    self.bias_layer_i_value_b = bias_layer_i_value_b    
    
  def forward(self, x):
        
    residual = x

    basicblock = self.conv_a(x)
    out_with_bias_a = self.bias_layer_i_value_a.cuda() # 64
    out_with_bias_a_expand = out_with_bias_a.view(1,out_with_bias_a.shape[0],1,1) # 1 64 1 1
    a = torch.ones(basicblock.shape[0], out_with_bias_a.shape[0], basicblock.shape[2], basicblock.shape[3]).cuda()
    b =  a * out_with_bias_a_expand
    b.index_add_(1, self.index_block_i_conva_value.cuda(), basicblock)

    basicblock = F.relu(b, inplace=True)

    basicblock = self.conv_b(basicblock)
    out_with_bias_b = self.bias_layer_i_value_b.cuda() # 64
    out_with_bias_b_expand = out_with_bias_b.view(1,out_with_bias_b.shape[0],1,1) # 1 64 1 1
    a = torch.ones(basicblock.shape[0], out_with_bias_b.shape[0], basicblock.shape[2], basicblock.shape[3]).cuda()
    b =  a * out_with_bias_b_expand
    b.index_add_(1, self.index_block_i_convb_value.cuda(), basicblock)
    # basicblock = F.relu(b, inplace=True)
    if self.downsample:
      out = b
    else:
      out = residual + b

    return F.relu(out, inplace=True)

class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes, index, bias_value, rate=[16, 16, 32, 64, 16, 32, 64]):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model   
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    self.stage_num = (depth - 2) // 3
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    self.num_classes = num_classes
    self.rate = rate
    print(layer_blocks)

    # self.bn_1 = nn.BatchNorm2d(rate[0])
  

    self.index_conv1 = {key: index[key] for key in index.keys() if 'conv_1_3x3' in key} #保留filter的index
    self.bias_conv1 = {key: bias_value[key] for key in bias_value.keys() if 'conv_1_3x3' in key}
    self.index_conv1_value = list(self.index_conv1.values())[0]
    self.bias_conv1_value = list(self.bias_conv1.values())[0]
    self.conv_1_3x3 = conv_1(rate[0], self.index_conv1_value, self.bias_conv1_value)

    self.index_layer1 = {key: index[key] for key in index.keys() if 'stage_1' in key}
    self.index_layer2 = {key: index[key] for key in index.keys() if 'stage_2' in key}
    self.index_layer3 = {key: index[key] for key in index.keys() if 'stage_3' in key}


    self.bias_layer1 = {key: bias_value[key] for key in bias_value.keys() if 'stage_1' in key}
    self.bias_layer2 = {key: bias_value[key] for key in bias_value.keys() if 'stage_2' in key}
    self.bias_layer3 = {key: bias_value[key] for key in bias_value.keys() if 'stage_3' in key}



    self.stage_1 = self._make_layer(block, 16, rate[1], 16, self.index_layer1, self.bias_layer1, layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 32, rate[3], 32, self.index_layer2, self.bias_layer2, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 64, rate[5], 64, self.index_layer3, self.bias_layer3, layer_blocks, 2)
   
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(64*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, inplanes, planes, supply, index, bias_layer, blocks, stride=1):
    downsample = False
    # if stride != 1 :
    #   downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
    layers = []
    index_block_i_conva_dict = {key: index[key] for key in index.keys() if (str(0) + '.conv_a') in key} #保留的
    index_block_i_conva_value = list(index_block_i_conva_dict.values())[0]
    index_block_i_convb_dict = {key: index[key] for key in index.keys() if (str(0) + '.conv_b') in key}
    index_block_i_convb_value = list(index_block_i_convb_dict.values())[0]

    bias_layer_i_a = {key: bias_layer[key] for key in bias_layer.keys() if (str(0) + '.conv_a' + '.bias') in key }
    bias_layer_i_value_a = list(bias_layer_i_a.values())[0]  #剪掉要补上的
    bias_layer_i_b = {key: bias_layer[key] for key in bias_layer.keys() if (str(0) + '.conv_b' + '.bias') in key}
    bias_layer_i_value_b = list(bias_layer_i_b.values())[0]
    if stride == 1 :
       layers.append(block(inplanes, planes, supply, index, index_block_i_conva_value, index_block_i_convb_value
        , bias_layer_i_value_a, bias_layer_i_value_b, stride, downsample=False))
    else:
       layers.append(block(inplanes//2, planes, supply, index, index_block_i_conva_value, index_block_i_convb_value
        , bias_layer_i_value_a, bias_layer_i_value_b, stride, downsample=True))

    for i in range(1, blocks):
      index_block_i_conva_dict = {key: index[key] for key in index.keys() if (str(i) + '.conv_a') in key}
      index_block_i_conva_value = list(index_block_i_conva_dict.values())[0]
      index_block_i_convb_dict = {key: index[key] for key in index.keys() if (str(i) + '.conv_b') in key}
      index_block_i_convb_value = list(index_block_i_convb_dict.values())[0]

      bias_layer_i_a = {key: bias_layer[key] for key in bias_layer.keys() if (str(i) + '.conv_a' + '.bias') in key }
      bias_layer_i_value_a = list(bias_layer_i_a.values())[0]
      bias_layer_i_b = {key: bias_layer[key] for key in bias_layer.keys() if (str(i) + '.conv_b' + '.bias') in key}
      bias_layer_i_value_b = list(bias_layer_i_b.values())[0]

      layers.append(block(inplanes, planes, supply, index, index_block_i_conva_value, index_block_i_convb_value
      , bias_layer_i_value_a, bias_layer_i_value_b, stride=1, downsample=False))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x) 
    x = F.relu(x, inplace=True)
#    print(x.size())
#    x = torch.autograd.Variable(torch.rand(x.size()[0], 16, x.size()[2], x.size()[3])).cuda()

    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

def resnet20_small(index, rate,num_classes=10):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 20, num_classes, index, rate)
  return model

def resnet32_small(index, rate,num_classes=10):
  """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 32, num_classes,index, rate)
  return model

def resnet44_small(index, rate,num_classes=10):
  """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 44, num_classes,index, rate)
  return model

def resnet56_small(index, bias_value, num_for_construct, num_classes=10):
  """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 56, num_classes, index, bias_value, num_for_construct)
  return model

def resnet110_small(index, bias_value, num_for_construct, num_classes=10):
  """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 110, num_classes, index, bias_value, num_for_construct)
  return model

##############################################################################################
#
#   MemNet: A Persistent Memory Network for Image Restoration
#   ICCV,2017
#   Date: 2018/3/30
#   Author: Rosun
#
##############################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels, True)  #FENet: staic(bn)+relu+conv1
        self.reconstructor = BNReLUConv(channels, in_channels, True)      #ReconNet: static(bn)+relu+conv 
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )
        #ModuleList can be indexed like a regular Python list, but modules it contains are 
        #properly registered, and will be visible by all Module methods.
        
        
        self.weights = nn.Parameter((torch.ones(1, num_memblock)/num_memblock), requires_grad=True)  
        #output1,...,outputn corresponding w1,...,w2


    #Multi-supervised MemNet architecture
    def forward(self, x):
        residual = x
        out = self.feature_extractor(x)
        w_sum=self.weights.sum(1)  
        mid_feat=[]   # A lsit contains the output of each memblock
        ys = [out]  #A list contains previous memblock output(long-term memory)  and the output of FENet
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)  #out is the output of GateUnit  channels=64
            mid_feat.append(out);
        #pred = Variable(torch.zeros(x.shape).type(dtype),requires_grad=False)
        pred = (self.reconstructor(mid_feat[0])+residual)*self.weights.data[0][0]/w_sum
        for i in range(1,len(mid_feat)):
            pred = pred + (self.reconstructor(mid_feat[i])+residual)*self.weights.data[0][i]/w_sum

        return pred

    #Base MemNet architecture
    '''
    def forward(self, x):
        residual = x   #input data 1 channel
        out = self.feature_extractor(x)
        ys = [out]  #A list contains previous memblock output and the output of FENet
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual
        
        return out
    '''


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        #self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, True)  #kernel 3x3
        self.gate_unit = GateUnit((num_resblock+num_memblock) * channels, channels, True)   #kernel 1x1

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
       
        
        #gate_out = self.gate_unit(torch.cat([xs,ys], dim=1))
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))  #where xs and ys are list, so concat operation is xs+ys
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, True)
        self.relu_conv2 = BNReLUConv(channels, channels, True)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))  #tureL: direct modified x, false: new object and the modified
        self.add_module('conv', nn.Conv2d(in_channels, channels, 3, 1, 1))  #bias: defautl: ture on pytorch, learnable bias

class GateUnit(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(GateUnit, self).__init__()
        self.add_module('bn',nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels,1,1,0))


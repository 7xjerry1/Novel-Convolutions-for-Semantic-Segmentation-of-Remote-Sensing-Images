import torch
import torch.nn as nn
import numpy as np
class LargeField_9(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(LargeField_9,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        
        self.padA=nn.ZeroPad2d(padding=(3, 3, 0, 0))
        self.padB=nn.ZeroPad2d(padding=(0, 0, 3, 3))
        self.padC=nn.ZeroPad2d(padding=(3, 3, 3, 3))
        
        self.avgPool=nn.AvgPool2d(stride=1,kernel_size=3,padding=1)
        
        self.conv2=nn.Conv2d(in_channels,out_channels,kernel_size=(1,2),stride=1,padding=0,dilation=6)
        self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=(2,1),stride=1,padding=0,dilation=6)
        self.conv4=nn.Conv2d(in_channels,out_channels,kernel_size=(2,2),stride=1,padding=0,dilation=6)
        
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        
        
    def forward(self,x):
        
        x1=self.conv1(x)
        
        x2=self.avgPool(x)
        
        
        x2_1=self.padA(x2)
        x2_1=self.conv2(x2_1)
        
        x2_2=self.padB(x2)
        x2_2=self.conv3(x2_2)
        
        x2_3=self.padC(x2)
        x2_3=self.conv4(x2_3)
        
        out=self.bn(x1+x2_1+x2_2+x2_3)
        out=self.relu(out)
        
        return out

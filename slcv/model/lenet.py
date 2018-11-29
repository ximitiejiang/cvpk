#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:03:19 2018

@author: suliang
"""

from torch import nn   
 

class LeNet5(nn.Module):
    
    def __init__(self, num_classes=10, input_layers=3):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(input_layers, 6, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.classifier = nn.Sequential(
                nn.Linear(16*5*5, 120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes)
                )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), 16*5*5)
        x = self.classifier(x)
        return x
    
    
class LeNet5_bn(nn.Module):
    
    def __init__(self, num_classes=10, input_layers=3):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(input_layers, 6, kernel_size=5),
                nn.BatchNorm2d(6),
                
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.BatchNorm2d(16),
                
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.classifier = nn.Sequential(
                nn.Linear(16*5*5, 120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes)
                )
        
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), 16*5*5)
        x = self.classifier(x)
        return x

class LeNet5_super(nn.Module):
    '''TODO:
    基于LeNet5的5层结构，但修改卷积传递层数，构建一个超级LeNet5
    原有LeNet5的层数变化是1 -> 6 -> 16 -> 400 -> 120 -> 84 -> 10
    现参考Alexnet为更多层 1 -> 64 -> 128->25088->1024->10
    (参考pytorch实战计算机视觉P177)
    '''
    def __init__(self, num_classes=6, input_layers=3):  # 3x64x64
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(input_layers, 64, 3, 1, 1),   # 64x64x64
                nn.BatchNorm2d(64),
                
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64x32x32
                
                nn.Conv2d(64, 128, 3, 1, 1),            # 128x32x32
                nn.BatchNorm2d(128),
                
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)   # 128x16x16
                )
        self.classifier = nn.Sequential(
                nn.Linear(128*16*16, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes)
                )
        
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), 128*16*16)
        x = self.classifier(x)
        return x

                
if __name__ == '__main__':
    lnsp = LeNet5_super(num_classes=6,input_layers=3)
    print(lnsp)
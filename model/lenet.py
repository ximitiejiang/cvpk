#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:03:19 2018

@author: suliang
"""

from torch import nn   
 

class LeNet5(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5),
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
                
if __name__ == '__main__':
    ln = LeNet5(num_classes=10)
    print(ln)
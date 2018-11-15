#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:16:51 2018

@author: ubuntu
"""

from torch import nn
import torch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.randn(3,3))
        
        self.submodel1 = nn.Linear(3,4)
        
    def forward(self, input):
        x = self.param1.input
        x = self.submodel1(x)
        return x
    
net = Net()
net

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:16:51 2018

@author: ubuntu
"""
from model.vgg16 import VGG16
from model.Pretrainedmodels import pretrained_models
from torch import nn
import torch

    



inputs = torch.randn(1,3,224,224)
labels = torch.tensor([[1,0]])
model = pretrained_models('vgg16', num_classes=2)
criterion = torch.nn.CrossEntropyLoss()    
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        

optimizer.zero_grad()
outputs = model(imgs)            
loss = criterion(outputs, labels)     # 这一步有可能很复杂的扩展
loss.backward()                     
optimizer.step() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:16:51 2018

@author: ubuntu
"""
from model.vgg16 import VGG16
from model.Pretrainedmodels import pretrained_models
from torch import nn
from collections import OrderedDict
import torch

''' 这一段可以看到怎么把参数加到_parameters变量中
    1. model对象调用__setattr__()方法，pytorch重写了这个方法
    2. 在__setattr__()方法中调用register_parameters()函数
    3. 在register_parameters()函数中，赋值self._parameters[name] = value
    
model = nn.Module()   # 创建所有变量包括：_parameters, _buffers
model.param = nn.Parameter(torch.ones(2,2))
model._parameters
'''


''' 这一段可以看到怎么把
'''
#model = nn.Module()
#sub_model1 = nn.Linear(2,2)
#sub_model2 = nn.Linear(2,2)
#model_list = [sub_model1, sub_model2]

#model = nn.Sequential(
#          nn.Conv2d(1,20,5),
#          nn.ReLU(),
#          nn.Conv2d(20,64,5),
#          nn.ReLU()
#        )
inputs = torch.randn(1,1,16,16)
labels = torch.tensor([[1]])

model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


print(model.__dict__)

getattr(model, 'training')
getattr(model, '_parameters')

optimizer.zero_grad()
outputs = model(inputs)            
loss = criterion(outputs, labels)     # 这一步有可能很复杂的扩展
loss.backward()    # 更新梯度                     
optimizer.step()   # 更新参数

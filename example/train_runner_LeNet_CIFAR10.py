#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:49:59 2018

@author: ubuntu
"""

from slcv.dataset.existdata import exist_datasets, data_transform
from torch.utils.data import DataLoader
from slcv.runner.runner import Runner
from slcv.model.lenet import LeNet5
from slcv.cfg.config import Config
import sys
import torch

# 0. 固定设置
if sys.platform == 'linux':    # for ubuntu
    data_root = '/home/ubuntu/MyDatasets/CIFAR10'     # for ubuntu 
else:                          # for Mac os
    data_root = '/Users/suliang/MyDatasets/CIFAR10'  # for mac os

cfg = Config().fromfile('../slcv/cfg/cfg_lenet_cifar10.py')  # 需要写相对路径

# 1. 数据
transform = data_transform(train=True, input_size = 32, mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
trainset = exist_datasets('cifar10', root=data_root, train=True, transform =transform, download=False)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# 2. 模型
model = LeNet5(num_classes=cfg.num_classes, input_layers=cfg.input_layers)
optimizer = dict(type = 'SGD', lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()
# 3. 训练
runner = Runner(trainloader, model, optimizer, cfg) # cfg对象也先传进去，想挂参数应该是需要的
runner.register_hooks(cfg.optimizer_config)
runner.train()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:27:53 2018

@author: ubuntu
"""

'''
本测试用于查看在多分类问题上，交叉熵损失函数的输入输出特性
'''
from torch.utils import datasets
import torch
from torch import nn

# 导入数据（mnist为多分类）
train_data_root = '/home/ubuntu/MyDatasets/DogsVSCats/train/'

trainset = datasets.MNIST(train_data_root, train=True)
train_dataloader = torch.utils.data.DataLoader(trainset,
                                               batch_size = 8,
                                               shuffle = True,
                                               num_workers = 2)

# 定义模型/函数
model
criterion = 
optimizer

# 训练
num_epoch=1
for i in range(num_epoch):
    for j in train_dataloader():
        

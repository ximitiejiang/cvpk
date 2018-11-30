#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:39:11 2018

@author: ubuntu

benchmark的训练精度参考陈云的书P48

debug记录：
    * 用batch_size=4, 8个epoch,损失可到1.56,精度43.3%
    * 改成batchsize=32后，损失到1.69,精度37.8%————batch_size加大反而精度下降
    * 从sgd/0.001改为adam/0.001/batchsize=4, 损失1.59, 精度42.2%————精度没有sgd好
    * 

"""
from slcv.dataset.existdata import exist_datasets, data_transform
from torch.utils.data import DataLoader
from slcv.runner.trainer import Trainer
from slcv.model.lenet import LeNet5
import sys

# 1. 数据
num_classes = 10
if sys.platform == 'linux':    # for ubuntu
    data_root = '/home/ubuntu/MyDatasets/CIFAR10'     # for ubuntu 
else:                          # for Mac os
    data_root = '/Users/suliang/MyDatasets/CIFAR10'  # for mac os
transform = data_transform(train=True, input_size = 32, mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
trainset = exist_datasets('cifar10', root=data_root, train=True, transform =transform, download=False)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


# 2. 模型
model = LeNet5(num_classes=num_classes, input_layers=3)

# 2. 训练
trainer = Trainer(model=model, num_classes= num_classes)
num_epoch=8
for i in range(num_epoch):
    print('epoch position: {}/{}'.format(i+1, num_epoch))
    for j,(imgs, labels) in enumerate(trainloader):
        if j%1000 ==0:
            print('...{}k batchs...'.format(j/1000))
        
        trainer.step(j, imgs, labels)
    
    trainer.epoch_show(i)

        
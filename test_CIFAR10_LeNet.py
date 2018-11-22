#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:39:11 2018

@author: ubuntu

benchmark的训练精度参考陈云的书P48

debug记录：
    * 

"""
from dataset.existdata import exist_datasets, data_transform
from torch.utils.data import DataLoader
from utils.trainer import Trainer
from model.lenet import LeNet5

# 1. 数据
num_classes = 10
root = '/Users/suliang/MyDatasets/CIFAR10'
transform = data_transform(train=True, input_size = 32, mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
trainset = exist_datasets('cifar10', root, train=True, transform =transform, download=False)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


# 2. 模型
model = LeNet5(num_classes=num_classes)

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

        
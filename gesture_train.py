#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:01:59 2018

@author: ubuntu

debug记录：
1. 模型设计错误：卷积层输出跟全连接层输入不匹配
2. 自己写的trainset放在trainloader读取报错list index out of range
"""

from dataset.gesture import GestureDataset
from utils.trainer import Trainer
from model.lenet import LeNet5_super
from torch.utils.data import DataLoader

# -------------1. data ------------------
root = '/home/ubuntu/MyDatasets/HandsDataset'
trainset = GestureDataset(root, split='train')
valset = GestureDataset(root, split='trainval')
testset = GestureDataset(root, split='test')
        
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

#(img, label)

print('total length of trainset is: {}'.format(len(trainset)))

# -------------2. model ------------------
model = LeNet5_super(num_classes=6, input_layers=3)


# -------------3. training ------------------
trainer = Trainer(model,num_classes=6)
num_epoch = 1
for i in range(num_epoch):
    print('epoch position: {}/{}'.format(i+1, num_epoch))
    for j, (imgs, labels) in enumerate(trainloader):
        if j%100 ==0:
            print('...{}k batchs...'.format(j/100))
        trainer.step(j, imgs, labels)
    
    trainer.epoch_show(i)
    

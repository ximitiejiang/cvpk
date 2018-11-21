#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:47:26 2018

@author: suliang

对比有无bn的alexnet在训练时的梯度值，是否bn能够很好的放置梯度值消失

"""

from model.lenet import LeNet5, LeNet5_bn
from dataset.existdata import exist_datasets, data_transform
from torch.utils.data import DataLoader
from tqdm import tqdm



def test_bn_in_alexnet():
    '''标准alexnet没有bn，尝试对比增加bn，看看bn对梯度值的影响有多大
    '''
    
    
    '''定义模型
    '''
    num_classes = 10
    ln = LeNet5(num_classes=num_classes)
    ln_bn = LeNet5_bn(num_classes=num_classes)

    
    '''定义MNIST数据
    '''
#    root = '/Users/suliang/MyDatasets/MNIST'  # for mac os
    root = '/home/ubuntu/MyDatasets/MNIST'     # for ubuntu

    transform = data_transform(train=True, input_size = 32, 
                               mean = [0.5,0.5,0.5],
                               std = [0.5,0.5,0.5])
    trainset = exist_datasets(name='mnist', 
                              root=root, 
                              train=True, 
                              transform =transform)

    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
    
    len_loader = len(trainloader)
    
    '''开始训练         
    '''    
    from utils.trainer import Trainer
    trainer = Trainer(model=ln, num_classes=num_classes)
    
    num_epoch = 4
    
    for i in range(num_epoch):
        print('epoch position: {}/{}'.format(i+1, num_epoch))
#        for j, (imgs, labels) in tqdm(enumerate(trainloader), total = len_loader):
        for j,(imgs, labels) in enumerate(trainloader):
            
            if j%1000 ==0:
                print('...{}k batchs...'.format(j/1000))

            trainer.step(j, imgs, labels)
            
        trainer.epoch_show(i)
    

if __name__ == '__main__':

    
    test_bn_in_alexnet()    
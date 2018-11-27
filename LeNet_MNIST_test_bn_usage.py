#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:47:26 2018

@author: suliang

对比有无bn的alexnet在训练时的梯度值，是否bn能够很好的防止梯度值消失
debug 记录
    * batch_size 从2加到8后，loss有变好的趋势
    * op采用adam或sgd差别不明显
    * lr从0.001改为0.01后，发生梯度消失
    * bn加进Lenet后效果不明显
    * 前4个epoch的loss和精度数据始终不理想：loss在1.2->0.7, accuracy在16.5->18.3
      可能跑个100个循环能好点
    * batch_size 从8直接到64, epoch=8: loss继续好转到0.65。
      但始终无法达到别人在epoch=8时0.1的水平，而且我的训练精度只有可怜的18%
    * batchsize=64, epoch=30, loss达到0.554，精度达到81.0%
      问题还没找到！
"""

from model.lenet import LeNet5, LeNet5_bn
from dataset.existdata import exist_datasets, data_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys


def test_bn_in_alexnet():
    '''标准alexnet没有bn，尝试对比增加bn，看看bn对梯度值的影响有多大
    '''
    
    
    '''1. 定义模型
    '''
    num_classes = 10
    ln = LeNet5(num_classes=num_classes, input_layers=1)
    ln_bn = LeNet5_bn(num_classes=num_classes, input_layers=1)
    
    '''2. 定义MNIST数据
    '''

    if sys.platform == 'linux':    # for ubuntu
        data_root = '/home/ubuntu/MyDatasets/MNIST'     # for ubuntu 
    else:                          # for Mac os
        data_root = '/Users/suliang/MyDatasets/MNIST'  # for mac os
    
    transform = data_transform(train=True, input_size = 32, 
                               mean = [0.5,0.5,0.5],
                               std = [0.5,0.5,0.5])
    trainset = exist_datasets(name='mnist', 
                              root=data_root, 
                              train=True, 
                              transform =transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    
    len_loader = len(trainloader)
    
    '''3. 开始训练         
    '''    
    from utils.trainer import Trainer
    trainer = Trainer(model=ln, num_classes=num_classes)
    
    num_epoch = 30
    
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
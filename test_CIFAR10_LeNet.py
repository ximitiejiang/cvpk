#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:39:11 2018

@author: ubuntu
"""
from dataset.existdata import exist_datasets

num_classes = 2
root = ''
trainset = exist_datasets('cifar10', root, train=True, transform =None, download=False)
trainloader

model

trainer = Trainer()

for i in range(num_epoch):
    for j,(imgs, labels) in enumerate(trainloader):
        
        trainer.step()
    
    trainer.
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:03:28 2018

@author: ubuntu

实践一下自己的框架：基于trainer类进行迁移学习

"""



from model.Pretrainedmodels import pretrained_models
from model.vgg16 import VGG16
from utils.trainer import Trainer
import torch
from dataset.dogcat import DogCat

import time


# data
train_data_root = '/home/ubuntu/MyDatasets/DogsVSCats/train/'
test_data_root = '/home/ubuntu/MyDatasets/DogsVSCats/test/'

trainset = DogCat(train_data_root, transform=None, train=True, test=False)
train_dataloader = torch.utils.data.DataLoader(trainset,
                                               batch_size = 8,
                                               shuffle = True,
                                               num_workers = 2)
#imgs, labels= next(iter(train_dataloader))
#imgs = imgs.squeeze(0)
#from utils.image_tensor import tensor2img,imgshow
#img = tensor2img(imgs)
#imgshow(img)

# model
model = VGG16(num_classes=2)

# training
trainer = Trainer(model = model)

n_epoch= 1
for i in range(n_epoch):
    
    for ii, (imgs, labels) in enumerate(train_dataloader):
        since = time.time()
        
        loss = trainer.train_step(imgs,labels)
        print(ii, 'last time:', time.time()-since, 'loss = ', loss)
        
        
        


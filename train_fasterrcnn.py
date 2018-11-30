#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:37:24 2018

@author: ubuntu

尝试在slcv框架里搭建faster rcnn, 仿造simple-faster-rcnn(by chenyun)
"""

from dataset.voc import VOCTrainDataset
from torch.utils.data import DataLoader
from utils.trainer_faster_rcnn import Trainer
from model.vgg_divide import vgg16
from model.faster_rcnn import FasterRCNN
from model.roi_head import VGG16RoIHead

# ------------1. data------------
# 基于源码确保该目录下有ImageSets/Main/文件夹
import sys
if sys.platform == 'linux':    # for ubuntu
    data_root = '/home/ubuntu/MyDatasets/VOCdevkit/VOC2007'  
else:                          # for Mac os
    data_root = '/Users/suliang/MyDatasets/PASCAL_VOC_2007/VOCdevkit2/VOC2007'   

trainset = VOCTrainDataset(root=data_root, split='trainval', use_difficult=False)
trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

'''trainset出来的4个参数中，img为float list[C,H,W], bbox为list
'''
img, bbox, label, scale = next(iter(trainloader))

# ------------2. model------------
extractor, classifier = vgg16()
print(extractor)
print(classifier)

rpn = 

head = VGG16RoIHead(n_class=512, roi_size=512, spatial_scale=, classifier=classifier)  # 1个分类器+2个并列全连接层

model = FasterRCNN(extractor, rpn, head,
                   loc_normalize_mean = (0., 0., 0., 0.),
                   loc_normalize_std = (0.1, 0.1, 0.2, 0.2))

# ------------3. training------------

trainer = Trainer(model, num_classes=20)
num_epoch =1

for i in range(num_epoch):
    print('epoch position: {}/{}'.format(i+1, num_epoch))
    for j, (img, bbox, label, scale) in enumerate(trainloader):
        
        trainer.step(j, img, bbox, label, scale)
    
    trainer.epoch_show(i)

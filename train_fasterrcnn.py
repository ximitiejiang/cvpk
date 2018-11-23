#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:37:24 2018

@author: ubuntu

尝试在slcv框架里搭建faster rcnn, 仿造simple-faster-rcnn(by chenyun)

"""

from dataset.voc import VOCBboxDataset
from torch.utils.data import DataLoader

# 1. data
# 基于源码确保该目录下有ImageSets/Main/文件夹

#data_root = '/home/ubuntu/MyDatasets/VOCdevkit/VOC2007'                     # for ubuntu
data_root = '/Users/suliang/MyDatasets/PASCAL_VOC_2007/VOCdevkit2/VOC2007'   # for Mac os
trainset = VOCBboxDataset(data_dir=data_root, split='trainval')
trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

img, bbox, label, _ = trainset[12]


next(iter(trainloader))

# 2. model



# 3. training

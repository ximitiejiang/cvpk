#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:10:30 2018

@author: ubuntu

整理打开文件的相对路径和绝对路径
1. 打开文件的思路：
    (1)获得当前入口文件的绝对路径头
    (2)输入目标文件的相对路径
    (3)拼接目标文件的绝对路径后打开
    
2. 对比包与模块的导入逻辑
    (1)明确__main__文件的位置
    (2)基于__main__文件输入目标文件的相对路径
    (3)采用相对路径进行包与模块的导入。

"""

from slcv.model.yolov3 import Darknet
from slcv.utils.init import weights_init_normal
import os
from torchvision import transforms

from slcv.dataset.voc_yolov3 import VOCDataset, voc_classes
from torch.utils.data import DataLoader
from slcv.runner.trainer_yolov3 import Trainer

# ---------1. data----------------
root = '/home/ubuntu/MyDatasets/voc/VOCdevkit'
trainset = VOCDataset(base_path=root, dataset_name='VOC2007',dataset_type='train', 
                      classes=voc_classes, img_size=416, img_ext=".jpg")
input_img, filled_labels = trainset[5]

dataloader = DataLoader(trainset, batch_size=16, shuffle=False, num_workers=1)

imgs, labels = next(iter(dataloader))

# --------2. model----------------
# darknet比较大，总共106个sequential
model_dir = '/model/yolov3.cfg'
model_config_path = os.path.dirname(__file__)  + model_dir

model = Darknet(model_config_path, img_size=416)
#print(model)

model.apply(weights_init_normal)
# 待验证weight init的效果

# --------3. training----------------
trainer = Trainer(model, 20)
num_epoch = 1  # 源码用了30
for i in range(num_epoch):
    
    for j,(imgs, labels) in enumerate(dataloader):
        
        loss = trainer.step(j, imgs, labels)
        
        
        print("[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
              % (i+1,
                 num_epoch,
                 j+1,
                 len(dataloader),
                 model.losses["x"],
                 model.losses["y"],
                 model.losses["w"],
                 model.losses["h"],
                 model.losses["conf"],
                 model.losses["cls"],
                 # loss计算出来如果是一个标量，则不能进行.item(),除非是一个单值tensor
                 loss.item(),
                 model.losses["recall"],
                 model.losses["precision"],
                 )
              )
        model.seen += imgs.size(0)
    
    trainer.epoch_show(i)
    
    checkpoint_dir = '/checkpoints'
    checkpoint_interval = 10
    if i % checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (checkpoint_dir, i))    



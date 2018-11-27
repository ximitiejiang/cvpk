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
import torch
from model.yolov3 import Darknet
from utils.init import weights_init_normal
import os
from torchvision import transforms

from dataset.coco import CocoDetection
from torch.utils.data import DataLoader
from utils.trainer_yolov3 import Trainer

# ---------1. data----------------
dataDir='/media/ubuntu/4430C54630C53FA2/SuLiang/MyDatasets/coco'
dataType='train2017'  # 也可定义成'val2017'
    
root = os.path.join(dataDir, dataType)
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# darknet模型要求的图片输入尺寸为416
# 这里增加的transform包含了尺寸（确认是416），totensor（pytorch要求），
# 归一化（github上看到有人实现时加了这个，参考：
# https://github.com/BinWang-shu/yolo_v3_pytorch/blob/master/train.py）
transform = transforms.Compose([transforms.Resize(416),
                                transforms.ToTensor(),
                                transforms.Normalization(mean = [0.485, 0.456, 0.406], 
                                                         std = [0.229, 0.224, 0.225])
                                ])
trainset = CocoDetection(root, annFile, transform=transform)
dataloader = DataLoader(trainset, batch_size=16, shuffle=False, num_workers=1)

# --------2. model----------------
# darknet比较大，总共106个sequential
model_dir = '/model/yolov3.cfg'
model_config_path = os.path.dirname(__file__)  + model_dir

model = Darknet(model_config_path, img_size=416)
#print(model)

model.apply(weights_init_normal)
# 待验证weight init的效果

# --------3. training----------------
trainer = Trainer()
num_epoch = 1
for i in range(num_epoch):
    
    for j,(imgs, labels) in enumerate(dataloader):
        
        trainer.step()
        
    trainer.epoch_show()
        
        
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
# lambda x,y,z: x+y+z
# filter(func, iterable) 


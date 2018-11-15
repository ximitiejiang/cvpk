#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:44:17 2018

@author: ubuntu
"""


from dataset import voc
from utils.image_tensor import tensor2img, imgshow
import matplotlib.pyplot as plt 

'''使用vocbboxdataset类，这个类预先编写好了__getitem__()和__len__()函数
    
    这个类可以作为一个子类，每种数据集创建一个子类。
    
    然后另外用一个统一的datset加载子类，并做相关处理
'''
voc_dir = '/home/ubuntu/MyDatasets/VOCdevkit/VOC2007'
dataset = voc.VOCBboxDataset(voc_dir,split='trainval')
datalabel = voc.VOC_BBOX_LABEL_NAMES

img, bbox, label, difficult = dataset[1316]  # img已转化成pytorch的[CHW]或[1HW]形式

# 由于dataset出来的数据还不完善，需要归一化和(0,1)化
def transformer(img):
    import torch
    from torchvision import transforms
    img = img/255              # (0,1)化
    img = torch.tensor(img)    # 转tensor
                               # 规范化
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(img)
    return img


img = transformer(img)

img = tensor2img(img)

imgshow(img, bbox)

for i in range(len(label)):
    print(datalabel[label[i]]) 
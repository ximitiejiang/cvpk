#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:25:06 2018

@author: ubuntu
"""
from dataset import voc
from utils.image_tensor import tensor2img, imgshow
import matplotlib.pyplot as plt
from model.vgg import vgg16 

voc_dir = '/home/ubuntu/MyDatasets/VOCdevkit/VOC2007'
dataset = voc.VOCBboxDataset(voc_dir,split='trainval')
datalabel = voc.VOC_BBOX_LABEL_NAMES

img, bbox, label, difficult = dataset[1316]  # img已转化成pytorch的[CHW]或[1HW]形式

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

extractor, classifier = vgg16()

'''由于batch_size = 1, 且也不需要shuffle, 这里省略dataloader

接下来
'''
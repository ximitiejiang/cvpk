#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:31:38 2018

@author: ubuntu
"""
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils.bboxtool import RectAngle

def img2tensor(img):
    '''输入：
            img: [H,W]
       输出：
            tensor (已转向量/转置/(0,1)化)
    '''
    ts = img.transforms.ToTensor()(img)
    
    return ts


def tensor2img(ts):
    '''把tensor转为图片
    输入： [c, h, w]
    输出： [w, h]
    '''
    # 逆归一化有点复杂：先去均值方差，再以numpy截断，再转float32
    img = (ts*0.225 + 0.45).numpy().clip(min=0,max=1).astype(np.float32)
    img = transforms.ToPILImage()(torch.from_numpy(img))
    return img


def imgshow(img):
    '''显示图片
    输入：
        图片 [w, h]
    '''
    plt.imshow(img)


def imgbboxshow(img,bbox):
    '''同时显示img和bbox
    输入：
        img: WxH
        bbox: [ymin,xmin,ymax,xmax]
    输出：
        
    '''
    plt.imshow(img)             # 显示图片
    for i in range(len(bbox)): 
        RectAngle(bbox[i])      # 显示bbox


def imgscale(img,min_size=600, max_size=1000):
    '''对图形进行缩放到合适尺寸
    默认缩放到faster rcnn的要求：最小尺寸>600, 最大尺寸<1000
    输入：图片 [C, H, W]
    输出：图片 [C, H, W]
        
    '''
    from skimage import transform as sktsf
    
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)   # 用小比例缩放图片，确保图片缩放到框定的HxW(1000x600)之内
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)  # 具体的缩放步骤
    
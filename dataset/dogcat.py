#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:54:39 2018

@author: ubuntu
"""

from torchvision import transforms
from torch.utils import data
import os
from PIL import Image

class DogCat(data.Dataset):
    '''猫狗数据集
    输入root
    输出

    '''
    def __init__(self, root, transform=None, train=True, test=False):
        self.test = test
        self.train = train
        # 1. 获得每个样本地址
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        if test:  # self.test如果是测试数据集(xxx/xxx/2345.jpg), 对文件排序
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
            
        else:     # self.train如果是训练数据集(xxx/xxx/cat.2345.jpg), 对文件排序
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        img_num = len(imgs)

        # 2. 拆分数据集
        if self.test:        # 如果是测试集则不分
            self.imgs = imgs
        elif self.train:    # 如果是训练集，则取0.7训练
            self.imgs = imgs[:int(0.7*img_num)]
        else: # 如果是验证集, 则取0.3
            self.imgs = imgs[int(0.7*img_num):]
        
        # 3. 定义变换器
        if transform is None: # 默认图片变换方式           
            if self.test or not self.train:  # 如果是测试集或者验证集
                self.transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                                      ])
            else:  # 训练集
                self.transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                                      ])
        
    def __getitem__(self, index):
        '''该内建函数返回一条数据或一个样本，data[index]等效于data.__getitem__(index)
        '''
        img_path = self.imgs[index]
        
        if self.test:  # 如果是测试集，没有标签，返回文件名的数字号
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:         # 如果是训练集，返回标签,1=dog, 0=cat
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        
        data = Image.open(img_path)   # 打开图片
        data = self.transforms(data)  # 对图片做变换transforms
        
        return data, label
    
                            
    def __len__(self):
        '''该内建函数返回样本数量，len(data)等效于data.__len__()
        '''
        return len(self.imgs)
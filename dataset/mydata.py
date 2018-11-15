#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 08:49:35 2018

@author: ubuntu
"""

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

'''创建一个基础版数据集，继承自pytorch的Dataset(不继承是不是也没问题，只要跟dataloader兼容就行)
导入dogcate数据集，原数据集包含train//test两个数据包，其中dog和cat的图片是放在一起的，通过名字区分
典型名字：cat.1.jpg, dog.7391.jpg 但注意他把cat的编号统一放在前边，dog的编号统一放在后边
'''

def transformer():
    pass

class traindata(Dataset):
    '''训练数据集
    '''

    def __init__(self,root,transform=None, train=True, val=False):
        super().__init__()
        self.transform= transformer()  # 一个子函数定义在外部还是内部更好？怎么导入？
        self.root = '/home/ubuntu/MyDatasets/DogsVSCats/train/'
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.n_imgs = len(imgs)
        
        if train and not val:
            imgs = imgs[]
        elif val and not train:
            img
        else:
            raise ValueError('train and val value should be bool type value!)
        
        
    def __getitem__(self,index):
        
        img = Image.open()
        return img, label
    
    
    def __len__(self):
        return self.n_imgs
    

class testdata(Dataset):
    def __init__(self,root,transform=None,):
        super().__init__()
        self.root = '/home/ubuntu/MyDatasets/DogsVSCats/test/'
        
        
    
    def __getitem__(self):
        pass
    
    def __len__(self):
        pass
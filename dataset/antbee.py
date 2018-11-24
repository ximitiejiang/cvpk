#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:53:32 2018

@author: ubuntu
"""

from torchvision import datasets

datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])

class AntBeeDataset():
    """基于ImageNet的一个数据子集，只包含了ant和bee两类
    参考pytorch英文网站中关于迁移学习的实例。
    数据源结构特点：每类图片统一放在一个文件夹，适合用pytorch的ImageFolder工具
    来创建dataset
    
    
    """
    def __init__(self, root, train=True):
        self.train = train
        if self.train:
            datasets.ImageFolder
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass


if __name__ == "__main__":
    root = ''
    trainset = AntBeeDataset(root)
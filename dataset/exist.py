#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:14:26 2018

@author: suliang
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def data_transform(train=True, input_size = 224):
    '''数据变换：如果要使用dataloader，就必须使用transform
    因为dataset为img格式，而dataloader只认tensor，所以在dataloader之前需要定义transform
    把img转换为tensor。这一步一般放在dataset步完成。
    同时还可以在transform中定义一些数据增强。
    
    '''
    
    if train:
        transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])
                                       ])
        
    else:
        transform = transforms.Compose([transforms.Resize(input_size),
                                        transforms.CenterCrop(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])
                                       ])
    return transform


def exist_datasets(name, root, train=True, transform =None, download=False):
    '''用于创建数据对象：具备__getitem__，__len__的类的实例称为数据对象
    输入：
        name: 'MNIST', 'CIFAR10', 'FashionMNIST', 'CocoCaptions', 'CocoDetection'
        root
    输出：
        data
        默认数据没有做transform，如果需要可自定义transform之后再创建数据对象    
    '''
    if train:  # for trainsets
        if name == 'MNIST':
            '''从MNIST源码看，root文件夹需定义到xxx/MNIST
            同时该MNIST文件夹下需要有processed,raw两个子文件夹，源码会拼接成图片地址和提取label
            '''
            datas = datasets.MNIST(root=root, train=train, transform=transform, download=download)
            
        elif name == 'CIFAR10':
            '''从
            '''
            datas = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        elif name == 'CocoCaptions':
            datas = datasets.CocoCaptions(root=root, train=train, transform=transform, download=download)
        elif name == 'CocoCaptions':
            datas = datasets.CocoCaptions(root=root, train=train, transform=transform, download=download)
            
        else:
            raise ValueError('not recognized data source!')
    
    
    
    else:  # for testsets
        pass
    
    return datas


if __name__ == '__main__':
    
    root = '/Users/suliang/MyDatasets/MNIST'
    transform = data_transform(train=True, input_size=32)
    datas = exist_datasets('MNIST', root, train=True, transform=transform)
    img, label = datas[0]
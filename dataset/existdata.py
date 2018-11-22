#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:14:26 2018

@author: suliang
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def data_transform(train=True, input_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    '''数据变换：如果要使用dataloader，就必须使用transform
    因为dataset为img格式，而dataloader只认tensor，所以在dataloader之前需要定义transform
    把img转换为tensor。这一步一般放在dataset步完成。
    同时还可以在transform中定义一些数据增强。
    
    '''
    
    if train:
        transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
                                       ])
        
    else:
        transform = transforms.Compose([transforms.Resize(input_size),
                                        transforms.CenterCrop(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
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
    数据集说明：MNIST为32x32小图，CIFAR10为
    '''
    if train:  # for trainsets
        if name == 'mnist':
            ''' 图片大小： 1x32x32, 单通道黑白图片
                类别数：10，labels采用0-9的数字表示
                训练集大小：60,000张图
            从MNIST源码看，root文件夹需定义到xxx/MNIST
            同时该MNIST文件夹下需要有processed,raw两个子文件夹，源码会拼接成图片地址和提取label
            
            '''
            datas = datasets.MNIST(root=root, train=train, transform=transform, download=download)
            classes = [0,1,2,3,4,5,6,7,8,9]
            
        elif name == 'cifar10':
            ''' 图片大小：3x32x32, 三通道彩色图片
                类别数：10, labels采用0-9的数字表示
                训练集大小: 50,000张图片
            从CIFAR10源码看，本地目录下需要有base_folder = 'cifar-10-batches-py'
            同时压缩文件也不能删除filename = "cifar-10-python.tar.gz"
            且训练集包含5个data_batch, 测试集包含1个test_batch
            '''
            datas = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
            classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
            
        elif name == 'cococaptions':
            datas = datasets.CocoCaptions(root=root, train=train, transform=transform, download=download)
            
        elif name == 'cococaptions':
            datas = datasets.CocoCaptions(root=root, train=train, transform=transform, download=download)
        else:
            raise ValueError('not recognized data source!')
            
    else:   # for testsets
        if name == 'mnist':
            '''从MNIST源码看，root文件夹需定义到xxx/MNIST
            同时该MNIST文件夹下需要有processed,raw两个子文件夹，源码会拼接成图片地址和提取label
            '''
            datas = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        else:
            raise ValueError('not recognized data source!')
    
    return datas



if __name__ == '__main__':
    
#    root = '/home/ubuntu/MyDatasets/MNIST'
    root = '/Users/suliang/MyDatasets/MNIST'
    transform_train = data_transform(train=True, input_size=32)
    transform_test = data_transform(train =False, input_size=32)
    
    # 1. 创建数据对象
    trainset = exist_datasets(name='MNIST', train=True, root=root, transform = transform_train)
    testset = exist_datasets(name='MNIST', train=False, root=root, transform = transform_test)
    
    img, label = trainset[0]         
    print(label)
    

    
    # 2. 做数据加载
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    
    imgs, labels = next(iter(trainloader))  # dataloader的数据格式
    
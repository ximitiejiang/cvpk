#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:08:56 2018

@author: ubuntu
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
    '''
    if train:
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
    
    else:
        if name == 'MNIST':
            '''从MNIST源码看，root文件夹需定义到xxx/MNIST
            同时该MNIST文件夹下需要有processed,raw两个子文件夹，源码会拼接成图片地址和提取label
            '''
            datas = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        else:
            raise ValueError('not recognized data source!')
    return datas

    
        
class Mydatasets(Dataset):
    '''创建自定义数据集, 继承Dataset类，可以？？？
    '''
    def __init__(self):
        super().__init__()
        
    def __getitem__(self):
        pass
    
    def __len__(self):
        pass
        
    
'''Dataloder类的是继承object的类，定义了__iter__(), __len__()方法
没有定义__next__方法，所以是一个可迭代对象，而不是一个迭代器!

可以用for循环而不能直接用next(),但可通过iter()函数转换成迭代器后就可以用next()函数
使用实例：
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
'''

    
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
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    
    imgs, labels = next(iter(trainloader))  # dataloader的数据格式
    
    
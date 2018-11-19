#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:08:56 2018

@author: ubuntu
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

def trainsets(name, root, train=True, transform =None, download=False):
    '''用于创建数据对象：具备__getitem__，__len__的类的实例称为数据对象
    输入：
        name: 'MNIST', 'CIFAR10', 'FashionMNIST', 'CocoCaptions', 'CocoDetection'
        root
    输出：
        data
        默认数据没有做transform，如果需要可自定义transform之后再创建数据对象
        
    '''
    if name == 'MNIST':
        data = datasets.MNIST(root=root, train=train, transform=transform, download=download)
    elif name == 'CIFAR10':
        data = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    elif name == 'CocoCaptions':
        data = datasets.CocoCaptions(root=root, train=train, transform=transform, download=download)
    elif name == 'CocoCaptions':
        data = datasets.CocoCaptions(root=root, train=train, transform=transform, download=download)
        
    else:
        raise ValueError('not recognized data source!')
        
    return data


def testsets(name, root, train=True, transform =None, download=False):
    '''用于创建数据对象：具备__getitem__，__len__的类的实例称为数据对象
    输入：
        name: 'MNIST', 'CIFAR10', 'FashionMNIST', 'CocoCaptions', 'CocoDetection'
        root
    输出：
        data
    '''
    pass
    
        
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
没有定义__next__方法，所以是一个可迭代对象，而不是一个迭代器，可以用for循环而不能直接用next()
但可通过iter()函数转换成迭代器后就可以用next()函数
使用实例：
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
'''

    
if __name__ == '__main__':

    root = '/home/ubuntu/MyDatasets/MNIST'
    trainset = trainsets(name='MNIST', root = root)
    testset = testsets(name='MNIST', root= root)
    
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

    imgs, labels = next(iter(trainloader))

    plt.imshow()
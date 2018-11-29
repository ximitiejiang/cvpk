#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 18:34:39 2018

@author: ubuntu

TODO:  train/val divide

"""
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image

def data_transform(train=True, input_size = 64, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
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

class GestureDataset():
    def __init__(self, root, split = 'train', transform = None):
        """train/test文件夹下各包含sign/sign1/sign2/sign3四个子文件夹，
        训练数据集文件夹从img_0000到img_1079，测试数据集从img_0000到img_0119
        图片总数：1080*4 = 4320张训练集，120*4 = 480张测试集 
        单张图片尺寸： 64x64
        参数： split = 'train'代表70%的数据, 'trainval'代表30%的数据, 
            'test'代表测试集。
        参数： transform = None代表没有指定外部transform则使用内部自带transform
            做变换，包括resize， totensor(含0,1化)，normalization(归一化)
        """
        if split == 'train':  # 0.7* total imgs
            db_dir = os.path.join(root, 'images/train.txt')
            self.imgs = []
            self.labels = []
            with open(db_dir) as f:
                lines = f.readlines()
                for line in lines:
                    self.labels.append(int(line.split(' ')[-1]))
                    self.imgs.append(os.path.join(root,line.split(' ')[0][2::]))
            # 划分train/val数据集: 0.7/0.3
            self.imgs = self.imgs[:int(0.7*len(self.imgs))]
            self.labels = self.labels[:int(0.7*len(self.imgs))]
        
        elif split == 'trainval':  # 0.3* total imgs
            db_dir = os.path.join(root, 'images/train.txt')
            self.imgs = []
            self.labels = []
            with open(db_dir) as f:
                lines = f.readlines()
                for line in lines:
                    self.labels.append(int(line.split(' ')[-1]))
                    self.imgs.append(os.path.join(root,line.split(' ')[0][2::]))
            # 划分train/val数据集: 0.7/0.3
            self.imgs = self.imgs[int(0.7*len(self.imgs)):]
            self.labels = self.labels[int(0.7*len(self.imgs)):]
                    
        elif split == 'test':
            db_dir = os.path.join(root, 'images/test.txt')
            self.imgs = []
            self.labels = []
            with open(db_dir) as f:
                lines = f.readlines()
                for line in lines:
                    self.labels.append(int(line.split(' ')[-1]))
                    self.imgs.append(os.path.join(root,line.split(' ')[0][2::]))
        else:
            raise ValueError('invalide split type: train/trainval/test')
            
        if transform == None:
            if split == 'train' or split == 'trainval':
                self.transform = data_transform(train=True, input_size = 64, 
                                                mean = [0.485, 0.456, 0.406], 
                                                std = [0.229, 0.224, 0.225])
            else:
                self.transform = data_transform(train=False, input_size = 64, 
                                                mean = [0.485, 0.456, 0.406], 
                                                std = [0.229, 0.224, 0.225])
                
    def __getitem__(self, index):
        img_dir = self.imgs[index]
        img = Image.open(img_dir)
        img = self.transform(img)
        label = self.labels[index]
        print(index)
        return img, label
    
    def __len__(self):
        return len(self.imgs)
    
    
if __name__ == '__main__':
    root = '/home/ubuntu/MyDatasets/HandsDataset'
    trainset = GestureDataset(root, split = 'train')
    print(len(trainset))
    
    data, label = trainset[18]
    print(label)
    
    # 统计每种类别的分布情况，发现基本相同:
    # train数据集中0,1,2,3,4,5的数量为[355 355 352 349 353 352]
    cal = [0,0,0,0,0,0]
    for i, (img,label) in enumerate(trainset): 
        label = label    
        cal[label] += 1

#    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#    imgs, labels = next(iter(trainloader))

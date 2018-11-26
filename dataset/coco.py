#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:22:26 2018

@author: ubuntu
"""
import torch.utils.data as data
from PIL import Image
import os
import os.path

class CocoDetection(data.Dataset):
    """`该数据集类来自于pytorch自带的类，这个类可以独立运行。
    但使用该接数据集之前，需要先安装coco api才能使用api中的COCO类和相关方法。
    MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    
    dataloader的输出说明(经过基本的transform后输出)
    imgs: 输出为BxCxHxW, 比如4x3x224x224
        训练集总计118,287张，验证集总计5000张
    labels: 输出为list，里边每个label为一个字典dict，字典包含信息如下
        {'area': float, 比如4457.12
        'bbox': list, 比如[8.29,57,26,75.33,146.9]
        'category_id': int, 比如21
        'id': int, 比如75654
        'image_id': int, 比如184613
        'iscrowd': int, 比如0
        'segmentation': list，比如[[272.72, 74.9,273.51,64.74 ..]]
        }
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
if __name__ == '__main__':
    # 以下调试成功：
    dataDir='/media/ubuntu/4430C54630C53FA2/SuLiang/MyDatasets/coco'
    dataType='train2017'  # 也可定义成'val2017'
    
    root = os.path.join(dataDir, dataType)
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    
    from torchvision import transforms
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()
                                    ])
    trainset = CocoDetection(root=root, annFile=annFile,transform=transform)   
    print('length of traindata:{}'.format(len(trainset)))
    img, label = trainset[2]
    
    from torch.utils.data import DataLoader
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2)
    imgs,labels = next(iter(trainloader))


    

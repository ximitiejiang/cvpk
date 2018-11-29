#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:26:09 2018

@author: ubuntu
"""

from torch import nn

def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3  # 初始通道数为3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):

    def __init__(self, num_classes=2):
        
        super().__init__()
        
        self.model_name = 'vgg16'  # 创建一个名称属性，用于保存model

        cfg ={11 : [64,     'M', 128,      'M', 256, 256,      'M', 512, 512,                'M', 512, 512,           'M'],
              13 : [64, 64, 'M', 128, 128, 'M', 256, 256,      'M', 512, 512,                'M', 512, 512,           'M'],
              16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,           'M', 512, 512, 512,      'M'],
              19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
        
        # 如果要用VGG19,则修改为cfg[19],
        # 默认增加BN层，如果不要BN层，则改为make_layers(cfg[16], batch_norm=False)
        self.features = make_layers(cfg[16])
        
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, num_classes),
                                        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:46:47 2018

@author: ubuntu
"""

from torchvision import models
import torch.nn as nn


def vgg16():
    model = models.vgg16(pretrained=True)
    
    features = model.features[:30]
    classifier = model.classifier
    
    #classifier = list(classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]
    
    #classifier = nn.Sequential(*classifier)
    
    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    
    return nn.Sequential(*features), nn.Sequential(*classifier)  
    # 如果不加*就会显示两层sequential嵌套，加了*就只显示一层sequential

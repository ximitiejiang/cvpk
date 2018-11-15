#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:00:18 2018

@author: ubuntu
"""
import torch
from torch import nn

class Trainer(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()    
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
#        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#        if torch.cuda.device_count() > 1:
#            self.model = nn.DataParallel(self.model)
        self.model.cuda()
            
    def forward(self):
        pass
    
    def train_step(self, imgs, labels):  # 5步循环
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        self.optimizer.zero_grad()
        outputs = self.model(imgs)            
        loss = self.criterion(outputs, labels)     # 这一步有可能很复杂的扩展
        loss.backward()                     
        self.optimizer.step()                    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:45:04 2018

@author: ubuntu
"""
import torch.nn as nn
from utils.anchor_base import generate_anchor_base, enumerate_shifted_anchor

class RegionProposalNetwork(nn.Module):
    '''RPN网络类: 完成从20000个候选框中筛选出2000个左右rois候选框的过程
    初始化：
        生成9个anchor base
        生成20,000个anchor候选框
    输入：
        feature: 从backbone解出来的features map, vgg16的输出为 [1,512,H/16,W/16] 
    输出：
        rois
        
    '''
    def __init__(self, in_channels = 512, mid_channels= 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.loc = nn.Conv2d(mid_channels, 36, 1, 1, 0)
        self.score = nn.Conv2d(mid_channels, 18, 1, 1, 0)

        
    def forward(self,features):
        h = self.conv1(features)
        rpn_locs = self.loc(h)
        rpn_score = self.score(h)
        
        
        
        # 开始分析rois的生成
        ab = generate_anchor_base(base_size=16, 
                                  ratios=[0.5, 1, 2],
                                  anchor_scales=[8, 16, 32])
        eab = enumerate_shifted_anchor(ab)
        rois
        
        return rois rpn_locs, rpn_score
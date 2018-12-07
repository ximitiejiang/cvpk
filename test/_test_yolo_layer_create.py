#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:55:48 2018

@author: ubuntu

以下为评估yolo layer的计算过程

"""


# yolo layer parameters
mask = [6,7,8]
anchors0 = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

"""在create_module阶段做的事情主要是提取了anchor box的参数,三个yolo对应anchor不同
(82)层为小尺度yolo，输入75x13x13，采用mask=6,7,8, anchor为[(116,90),(156,198),(373,326)]
(94)层为中尺度yolo，输入75x26x26，采用mask=3,4,5, anchor为[(30,61),(62,45),(59,119)]
(106)层为大尺度yolo，输入75x52x52，采用mask=0,1,2, anchor为[(10,13),(16,30),(33,23)]
"""
#anchor_idxs = [int(x) for x in mask.split(",")]
anchor_idxs = mask
# Extract anchors
#anchors = [int(x) for x in module_def["anchors"].split(",")]
anchors1 = anchors0  # 提取字符
anchors2 = [(anchors1[i], anchors1[i + 1]) for i in range(0, len(anchors1), 2)] # 分组：9组anchor
anchors3 = [anchors2[i] for i in anchor_idxs]  # 取第6,7,8组
#num_classes = int(module_def["classes"])
num_classes = 80
#img_height = int(hyperparams["height"])
img_height = 416

"""在yolo init阶段做的事情是：
* 初始化变量：anchors, num_anchors, num_classes, bbox_attrs, image_dim, ignore_thres, lambda_coord
* 初始化损失：mse_loss, bce_loss, crossentropy_loss
"""
## Define detection layer
#yolo_layer = YOLOLayer(anchors, num_classes, img_height)
anchors = anchors3
num_anchors = len(anchors)
num_classes = 80
bbox_attrs = 5 + num_classes
image_dim = img_height
ignore_thres = 0.5
lambda_coord = 1

import torch.nn as nn
mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
ce_loss = nn.CrossEntropyLoss()  # Class loss

"""在forward阶段做的事情
"""
import torch
x = torch.randn(16,75,13,13)  # yolo layer的输入

nA = num_anchors # 3个anchor
nB = x.size(0)   # 16张图片
nG = x.size(2)   # 13是图片尺寸
stride = image_dim / nG  # 32倍下采样倍数

# 把原本x size(16,75,13,13)的feature map，
# 变换为size(16, 3, 85, 13, 13)再转换为size(16, 3, 13, 13, 85)
prediction = x.view(nB, nA, bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()












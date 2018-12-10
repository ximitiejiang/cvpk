#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:55:48 2018

@author: ubuntu

以下为评估yolo layer的计算过程

"""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

#from utils.parse_config import *

from collections import defaultdict
#
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
    """这是yolov3算法用来生成anchor box的子程序
    """
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            
            """子程序bbox_iou()用于
            """
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls

"""-----------------------------------------------------------------
yolo层定义时的参数输入
"""
# yolo layer parameters
mask = [6,7,8]
anchors0 = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
classes=20
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

"""---------------yolo init---------------------------------------------------
* 初始化变量：anchors, num_anchors, num_classes, bbox_attrs, image_dim, ignore_thres, lambda_coord
* 初始化损失：mse_loss, bce_loss, crossentropy_loss
"""
## Define detection layer
#yolo_layer = YOLOLayer(anchors, num_classes, img_height)
anchors = anchors3  
num_anchors = len(anchors)
num_classes = 20
bbox_attrs = 5 + num_classes
image_dim = img_height
ignore_thres = 0.5
lambda_coord = 1

import torch.nn as nn
mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
ce_loss = nn.CrossEntropyLoss()  # Class loss

"""-------------yolo forward---------------------------------

"""
import torch
x = torch.randn(16,75,13,13)  # yolo layer的输入
targets = torch.zeros((16,20,5))

nA = num_anchors  # num of anchors(3)
nB = x.size(0)         # num of batch(16张图)
nG = x.size(2)         # num of grid points(???13的宽高)
stride = image_dim / nG  # 下采样缩放比例(32)

# Tensors for cuda support
FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
# 参考https://github.com/ultralytics/yolov3/blob/master/models.py
# (batch_size, num_anchor, map_size, map_size) -> ()
# (16, 75, 13, 13) -> (16, 3, 25, 13, 13) -> (16,3,13,13,25)
prediction = x.view(nB, nA, bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

# Get output
# 分类问题需要卷积+激活 ？？？
x = torch.sigmoid(prediction[..., 0])  # Center x
y = torch.sigmoid(prediction[..., 1])  # Center y

w = prediction[..., 2]  # Width
h = prediction[..., 3]  # Height
pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

# Calculate offsets for each grid
grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

# Add offset and scale with anchors
pred_boxes = FloatTensor(prediction[..., :4].shape)
pred_boxes[..., 0] = x.data + grid_x
pred_boxes[..., 1] = y.data + grid_y
pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

# Training
if targets is not None:

    if x.is_cuda:
        mse_loss = mse_loss.cuda()
        bce_loss = bce_loss.cuda()
        ce_loss = ce_loss.cuda()

    nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
        pred_boxes=pred_boxes.cpu().data,
        pred_conf=pred_conf.cpu().data,
        pred_cls=pred_cls.cpu().data,
        target=targets.cpu().data,
        anchors=scaled_anchors.cpu().data,
        num_anchors=nA,
        num_classes=num_classes,
        grid_size=nG,
        ignore_thres=ignore_thres,
        img_dim=image_dim,
    )

    nProposals = int((pred_conf > 0.5).sum().item())
    recall = float(nCorrect / nGT) if nGT else 1
    precision = float(nCorrect / nProposals)

    # Handle masks
    mask = Variable(mask.type(ByteTensor))
    conf_mask = Variable(conf_mask.type(ByteTensor))

    # Handle target variables
    tx = Variable(tx.type(FloatTensor), requires_grad=False)
    ty = Variable(ty.type(FloatTensor), requires_grad=False)
    tw = Variable(tw.type(FloatTensor), requires_grad=False)
    th = Variable(th.type(FloatTensor), requires_grad=False)
    tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
    tcls = Variable(tcls.type(LongTensor), requires_grad=False)

    # Get conf mask where gt and where there is no gt
    conf_mask_true = mask
    conf_mask_false = conf_mask - mask

    # Mask outputs to ignore non-existing objects
    loss_x = mse_loss(x[mask], tx[mask])
    loss_y = mse_loss(y[mask], ty[mask])
    loss_w = mse_loss(w[mask], tw[mask])
    loss_h = mse_loss(h[mask], th[mask])
    loss_conf = bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + bce_loss(
        pred_conf[conf_mask_true], tconf[conf_mask_true]
    )
    loss_cls = (1 / nB) * ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    output = (
        loss,
        loss_x.item(),
        loss_y.item(),
        loss_w.item(),
        loss_h.item(),
        loss_conf.item(),
        loss_cls.item(),
        recall,
        precision,
    )

else:
    # If not in training phase return predictions
    output = torch.cat(
        (
            pred_boxes.view(nB, -1, 4) * stride,
            pred_conf.view(nB, -1, 1),
            pred_cls.view(nB, -1, self.num_classes),
        ),
        -1,
    )













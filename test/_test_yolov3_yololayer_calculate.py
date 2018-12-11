#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:55:48 2018

@author: ubuntu

以下为评估yolo layer的计算过程

"""
from __future__ import division

import ipdb
import torch
import torch.nn as nn
import numpy as np
#from utils.parse_config import *

def class_weights():  # frequency of each class in coco train2014
    """计算出的每个类的样本数
    """
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def build_targets(pred_boxes, pred_conf, pred_cls, target, anchor_wh, nA, nC, nG, batch_report):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch
    # nT 代表每张图片包含多少bbox
    nT = [len(x) for x in target]  # torch.argmin(target[:, :, 4], 1)  # targets per image
    #  (16, 3, 13, 13)
    tx = torch.zeros(nB, nA, nG, nG)  # batch size (4), number of anchors (3), number of grid points (13)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    # (16,3,13,13)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    # (16,3,13,13,20)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes
    # (16,20)  20代表其中一张图最多bbox的个数
    TP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FN = torch.ByteTensor(nB, max(nT)).fill_(0)
    TC = torch.ShortTensor(nB, max(nT)).fill_(-1)  # target category

    for b in range(nB):  # 循环每张图
        nTb = nT[b]  # number of targets
        if nTb == 0:
            continue
        t = target[b]  # 取出该图的target(16,20,5) -> t (20,5)
        if batch_report:
            FN[b, :nTb] = 1  # 第b张图的nTb个bbox，都置1

        # Convert to position relative to box
        # 
        TC[b, :nTb], gx, gy, gw, gh = t[:, 0].long(), t[:, 1] * nG, t[:, 2] * nG, t[:, 3] * nG, t[:, 4] * nG
        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        gi = torch.clamp(gx.long(), min=0, max=nG - 1)
        gj = torch.clamp(gy.long(), min=0, max=nG - 1)

        # iou of targets-anchors (using wh only)
        box1 = t[:, 3:5] * nG
        # box2 = anchor_grid_wh[:, gj, gi]
        box2 = anchor_wh.unsqueeze(1).repeat(1, nTb, 1)
        inter_area = torch.min(box1, box2).prod(2)
        iou_anch = inter_area / (gw * gh + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_anch_best, a = iou_anch.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            iou_order = np.argsort(-iou_anch_best)  # best to worst

            # Unique anchor selection (slower but retains original order)
            u = torch.cat((gi, gj, a), 0).view(3, -1).numpy()
            _, first_unique = np.unique(u[:, iou_order], axis=1, return_index=True)  # first unique indices

            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_anch_best[i] > 0.10]
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_anch_best < 0.10:
                continue
            i = 0

        tc, gx, gy, gw, gh = t[:, 0].long(), t[:, 1] * nG, t[:, 2] * nG, t[:, 3] * nG, t[:, 4] * nG

        # Coordinates
        tx[b, a, gj, gi] = gx - gi.float()
        ty[b, a, gj, gi] = gy - gj.float()

        # Width and height (yolo method)
        tw[b, a, gj, gi] = torch.log(gw / anchor_wh[a, 0])
        th[b, a, gj, gi] = torch.log(gh / anchor_wh[a, 1])

        # Width and height (power method)
        # tw[b, a, gj, gi] = torch.sqrt(gw / anchor_wh[a, 0]) / 2
        # th[b, a, gj, gi] = torch.sqrt(gh / anchor_wh[a, 1]) / 2

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1

        if batch_report:
            # predicted classes and confidence
            tb = torch.cat((gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2)).view(4, -1).t()  # target boxes
            pcls = torch.argmax(pred_cls[b, a, gj, gi], 1).cpu()
            pconf = torch.sigmoid(pred_conf[b, a, gj, gi]).cpu()
            iou_pred = bbox_iou(tb, pred_boxes[b, a, gj, gi].cpu())

            TP[b, i] = (pconf > 0.5) & (iou_pred > 0.5) & (pcls == tc)
            FP[b, i] = (pconf > 0.5) & (TP[b, i] == 0)  # coordinates or class are wrong
            FN[b, i] = pconf <= 0.5  # confidence score is too low (set to zero)

    return tx, ty, tw, th, tconf, tcls, TP, FP, FN, TC

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
anchor_idxs = mask  # 3个anchors
# Extract anchors
#anchors = [int(x) for x in module_def["anchors"].split(",")]
anchors1 = anchors0  # 提取字符
anchors2 = [(anchors1[i], anchors1[i + 1]) for i in range(0, len(anchors1), 2)] # 分组：9组anchor
anchors3 = [anchors2[i] for i in anchor_idxs]  # 取第6,7,8组
#num_classes = int(module_def["classes"])
num_classes = 20
#img_height = int(hyperparams["height"])
img_height = 416

"""---------------yolo init---------------------------------------------------
* 初始化变量：anchors, num_anchors, num_classes, bbox_attrs, image_dim, ignore_thres, lambda_coord
* 初始化损失：mse_loss, bce_loss, crossentropy_loss
"""
## Define detection layer
#yolo_layer = YOLOLayer(anchors, num_classes, img_height)
anchors = [(a_w, a_h) for a_w, a_h in anchors3]  # (pixels)
nA = len(anchors)

nA = nA  # number of anchors (3)
nC = num_classes  # number of classes (20)
bbox_attrs = 5 + nC
img_dim = img_height  # from hyperparams in cfg file, NOT from parser (416)

if anchor_idxs[0] == (nA * 2):  # 6
    stride = 32
elif anchor_idxs[0] == nA:  # 3
    stride = 16
else:
    stride = 8

# Build anchor grids
nG = int(img_dim / stride)  # number grid points
# (13,13) -> (1,1,13,13)
grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
# 缩放原始anchors
scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
# 缩放后的w 进行升维 (3,1) -> (1,3,1,1)
anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
weights = class_weights()  # 返回每个类的数量百分比

loss_means = torch.ones(6)
tx, ty, tw, th = [], [], [], []

"""-------------yolo forward---------------------------------

"""
import torch
from torch.utils.data import DataLoader
from slcv.dataset.voc_yolov3 import VOCDataset, voc_classes
from slcv.cfg.config import Config
cfg = Config().fromfile('../slcv/cfg/cfg_yolov3_voc.py')  # 需要写相对路径
trainset = VOCDataset(base_path=cfg.data_root, dataset_name='VOC2007',
                      dataset_type='train', classes=voc_classes, 
                      img_size=416, img_ext=".jpg")
trainloader = DataLoader(trainset, batch_size=16, 
                         shuffle=False, num_workers=1)
_, targets = next(iter(trainloader)) # 把targets拿出来 (16,50,5) 16张图，50个bbox x 5个坐标(x,y,w,h)
prediction = torch.randn(16,75,13,13)  # 把prediction造出来
#targets = torch.zeros((16,20,5))
batch_report =True

FT = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
# p为卷积输出，假定(16,75,13,13), 用最小的feature map配最大的anchors
bs = prediction.shape[0]  # batch size
nG = prediction.shape[2]  # number of grid points
stride = img_dim / nG

if prediction.is_cuda and not grid_x.is_cuda:
    grid_x, grid_y = grid_x.cuda(), grid_y.cuda()
    anchor_w, anchor_h = anchor_w.cuda(), anchor_h.cuda()
    weights, loss_means = weights.cuda(), loss_means.cuda()

# p.view(16, 75, 13, 13) -> (16, 3, 25, 13, 13) -> (16,3,13,13,25) 把25分离出来放在最后一维便于分离25个变量  
# (bs, anchors, grid, grid, classes + xywh)
prediction = prediction.view(bs, nA, bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction
"""-------------预测后处理 1. 取出x, y, w, h , confidence, cls-----------"""
# Get outputs
x = torch.sigmoid(prediction[..., 0])  # Center x  (16,3,13,13) 表示的是16张图，3个anchor的x坐标预测
y = torch.sigmoid(prediction[..., 1])  # Center y

# Width and height (yolo method)
w = prediction[..., 2]  # Width
h = prediction[..., 3]  # Height
pred_conf = prediction[..., 4]  # Conf
pred_cls = prediction[..., 5:]  # Class (voc = 20)

"""-------------预测后处理 2. 转换预测w,h为实际w,h-----------
w,h 先exp, 再乘以pw?
x, y 先sigmoid，再加上网格偏移，然后转化为x1,y1,x2,y2
"""
# 把预测的w,h转换为实际w,h
# (16,3,13) * (1,3,1,1) -> (16,3,13)  按位置相乘，原理是？？？
width = torch.exp(w.data) * anchor_w
height = torch.exp(h.data) * anchor_h

# Width and height (power method)
# w = torch.sigmoid(p[..., 2])  # Width
# h = torch.sigmoid(p[..., 3])  # Height
# width = ((w.data * 2) ** 2) * self.anchor_w
# height = ((h.data * 2) ** 2) * self.anchor_h

# Add offset and scale with anchors (in grid space, i.e. 0-13)
pred_boxes = FT(bs, nA, nG, nG, 4) # 预创建 (16,3,13,13,4) 4代表x,y,w,h,confidence共5个变量


# Training
if targets is not None:
    MSELoss = nn.MSELoss()
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    CrossEntropyLoss = nn.CrossEntropyLoss()
    
    ipdb.set_trace()
    
    if batch_report:  # ??
        gx = grid_x[:, :, :nG, :nG]  # (1,1,13,13)
        gy = grid_y[:, :, :nG, :nG]
        
        pred_boxes[..., 0] = x.data + gx - width / 2
        pred_boxes[..., 1] = y.data + gy - height / 2
        pred_boxes[..., 2] = x.data + gx + width / 2
        pred_boxes[..., 3] = y.data + gy + height / 2
    
    tx, ty, tw, th, mask, tcls, TP, FP, FN, TC = \
        build_targets(pred_boxes, pred_conf, pred_cls, 
                      targets, scaled_anchors, 
                      nA, nC, nG,
                      batch_report)
    tcls = tcls[mask]
    if x.is_cuda:
        tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

    # Compute losses
    nT = sum([len(x) for x in targets])  # number of targets
    nM = mask.sum().float()  # number of anchors (assigned to targets)
    nB = len(targets)  # batch size
    k = nM / nB
    if nM > 0:
        lx = k * MSELoss(x[mask], tx[mask])
        ly = k * MSELoss(y[mask], ty[mask])
        lw = k * MSELoss(w[mask], tw[mask])
        lh = k * MSELoss(h[mask], th[mask])

        # self.tx.extend(tx[mask].data.numpy())
        # self.ty.extend(ty[mask].data.numpy())
        # self.tw.extend(tw[mask].data.numpy())
        # self.th.extend(th[mask].data.numpy())
        # print([np.mean(self.tx), np.std(self.tx)],[np.mean(self.ty), np.std(self.ty)],[np.mean(self.tw), np.std(self.tw)],[np.mean(self.th), np.std(self.th)])
        # [0.5040668, 0.2885492] [0.51384246, 0.28328574] [-0.4754091, 0.57951087] [-0.25998235, 0.44858757]
        # [0.50184494, 0.2858976] [0.51747805, 0.2896323] [0.12962963, 0.6263085] [-0.2722081, 0.61574113]
        # [0.5032071, 0.28825334] [0.5063132, 0.2808862] [0.21124361, 0.44760725] [0.35445485, 0.6427766]
        # import matplotlib.pyplot as plt
        # plt.hist(self.x)

        # lconf = k * BCEWithLogitsLoss(pred_conf[mask], mask[mask].float())

        lcls = (k / 4) * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
        # lcls = (k * 10) * BCEWithLogitsLoss(pred_cls[mask], tcls.float())
    else:
        lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

    # lconf += k * BCEWithLogitsLoss(pred_conf[~mask], mask[~mask].float())
    lconf = (k * 64) * BCEWithLogitsLoss(pred_conf, mask.float())

    # Sum loss components
    balance_losses_flag = False
    if balance_losses_flag:
        k = 1 / loss_means.clone()
        loss = (lx * k[0] + ly * k[1] + lw * k[2] + lh * k[3] + lconf * k[4] + lcls * k[5]) / k.mean()

        loss_means = loss_means * 0.99 + \
                          FT([lx.data, ly.data, lw.data, lh.data, lconf.data, lcls.data]) * 0.01
    else:
        loss = lx + ly + lw + lh + lconf + lcls

    # Sum False Positives from unassigned anchors
    FPe = torch.zeros(nC)
    if batch_report:
        i = torch.sigmoid(pred_conf[~mask]) > 0.5
        if i.sum() > 0:
            FP_classes = torch.argmax(pred_cls[~mask][i], 1)
            FPe = torch.bincount(FP_classes, minlength=nC).float().cpu()  # extra FPs

    output = (loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
           nT, TP, FP, FPe, FN, TC)

else:
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = width
    pred_boxes[..., 3] = height

    # If not in training phase return predictions
    output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
                        torch.sigmoid(pred_conf.view(bs, -1, 1)), pred_cls.view(bs, -1, nC)), -1)









"""
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
grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)  # (13,13)->(1,1,13,13)
grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor) # (13,13)->(1,1,13,13)
# anchors缩小到feature map等级
scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
# 缩小后的anchor_w，并升维成(1,3,1,1)
anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))  # (3,1)->(1,3,1,1)
anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

# Add offset and scale with anchors
pred_boxes = FloatTensor(prediction[..., :4].shape)  # (16,3,13,13,4) 包含了x,y,w,h,confi
pred_boxes[..., 0] = x.data + grid_x   # (16,3,13,13) + (13,13) -> (16,3,13,13)
pred_boxes[..., 1] = y.data + grid_y
pred_boxes[..., 2] = torch.exp(w.data) * anchor_w # (16,3,13,13)*(1,3,1,1) -> (16,3,13,13)
pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

# Training
if targets is not None:

    if x.is_cuda:
        mse_loss = mse_loss.cuda()
        bce_loss = bce_loss.cuda()
        ce_loss = ce_loss.cuda()

    nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
        pred_boxes=pred_boxes.cpu().data,  # 经sigmoid/offset grid以后的x,y,w,h
        pred_conf=pred_conf.cpu().data,    # 经sigmoid以后的confidence
        pred_cls=pred_cls.cpu().data,      # 经sigmoid以后的cls
        target=targets.cpu().data,         # labels
        anchors=scaled_anchors.cpu().data, # 缩小后的anchors
        num_anchors=nA,                     
        num_classes=num_classes,
        grid_size=nG,
        ignore_thres=ignore_thres,      # 忽略限值
        img_dim=image_dim,              # 
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
            pred_cls.view(nB, -1, num_classes),
        ),
        -1,
    )
"""










#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:50:00 2018

@author: ubuntu
"""

"""另一个yolov3 model，源模型基于coco数据集
参考：https://github.com/ultralytics/yolov3/blob/master/models.py
"""
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np

#from utils.parse_config import *
#from utils.utils import *

def class_weights():  # frequency of each class in coco train2014
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
    nT = [len(x) for x in target]  # torch.argmin(target[:, :, 4], 1)  # targets per image
    tx = torch.zeros(nB, nA, nG, nG)  # batch size (4), number of anchors (3), number of grid points (13)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes
    TP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FN = torch.ByteTensor(nB, max(nT)).fill_(0)
    TC = torch.ShortTensor(nB, max(nT)).fill_(-1)  # target category

    for b in range(nB):  # 取每张图
        nTb = nT[b]  # number of targets
        if nTb == 0: # 如果该图没有bbox则进入下一轮循环
            continue
        t = target[b]  # target (16,50,5) -> t (50,5)
        if batch_report:
            FN[b, :nTb] = 1

        # Convert to position relative to box
        TC[b, :nTb], gx, gy, gw, gh = t[:, 0].long(), \
                                      t[:, 1] * nG, \    # label x * nG
                                      t[:, 2] * nG, \    # label y * nG
                                      t[:, 3] * nG, \    # label w * nG
                                      t[:, 4] * nG       # label h * nG
        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        gi = torch.clamp(gx.long(), min=0, max=nG - 1)
        gj = torch.clamp(gy.long(), min=0, max=nG - 1)

        # iou of targets-anchors (using wh only)
        box1 = t[:, 3:5] * nG   # 取出该张图
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


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs)
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, img_dim, anchor_idxs):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (20)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        nG = int(self.img_dim / stride)  # number grid points
        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        self.weights = class_weights()

        self.loss_means = torch.ones(6)
        self.tx, self.ty, self.tw, self.th = [], [], [], []

    def forward(self, p, targets=None, batch_report=False, var=None):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        # p为卷积输出，假定(16,75,13,13), 用最小的feature map配
        bs = p.shape[0]  # batch size
        nG = p.shape[2]  # number of grid points
        stride = self.img_dim / nG

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
            self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

        # p.view(12, 255, 13, 13) -- > (12, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Get outputs
        x = torch.sigmoid(p[..., 0])  # Center x
        y = torch.sigmoid(p[..., 1])  # Center y

        # Width and height (yolo method)
        w = p[..., 2]  # Width
        h = p[..., 3]  # Height
        width = torch.exp(w.data) * self.anchor_w
        height = torch.exp(h.data) * self.anchor_h

        # Width and height (power method)
        # w = torch.sigmoid(p[..., 2])  # Width
        # h = torch.sigmoid(p[..., 3])  # Height
        # width = ((w.data * 2) ** 2) * self.anchor_w
        # height = ((h.data * 2) ** 2) * self.anchor_h

        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FT(bs, self.nA, nG, nG, 4)
        pred_conf = p[..., 4]  # Conf
        pred_cls = p[..., 5:]  # Class

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            if batch_report:
                gx = self.grid_x[:, :, :nG, :nG]
                gy = self.grid_y[:, :, :nG, :nG]
                pred_boxes[..., 0] = x.data + gx - width / 2    # xc -w/2 = xmin
                pred_boxes[..., 1] = y.data + gy - height / 2   # yc -h/2 = ymin
                pred_boxes[..., 2] = x.data + gx + width / 2    # ??        = xmax
                pred_boxes[..., 3] = y.data + gy + height / 2   # ??        = ymax

            tx, ty, tw, th, mask, tcls, TP, FP, FN, TC = \
                build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
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
                k = 1 / self.loss_means.clone()
                loss = (lx * k[0] + ly * k[1] + lw * k[2] + lh * k[3] + lconf * k[4] + lcls * k[5]) / k.mean()

                self.loss_means = self.loss_means * 0.99 + \
                                  FT([lx.data, ly.data, lw.data, lh.data, lconf.data, lcls.data]) * 0.01
            else:
                loss = lx + ly + lw + lh + lconf + lcls

            # Sum False Positives from unassigned anchors
            FPe = torch.zeros(self.nC)
            if batch_report:
                i = torch.sigmoid(pred_conf[~mask]) > 0.5
                if i.sum() > 0:
                    FP_classes = torch.argmax(pred_cls[~mask][i], 1)
                    FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()  # extra FPs

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   nT, TP, FP, FPe, FN, TC

        else:
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
                                torch.sigmoid(pred_conf.view(bs, -1, 1)), pred_cls.view(bs, -1, self.nC)), -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(cfg_path)
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nT', 'TP', 'FP', 'FPe', 'FN', 'TC']

    def forward(self, x, targets=None, batch_report=False, var=0):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets, batch_report, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            if batch_report:
                self.losses['TC'] /= 3  # target category
                metrics = torch.zeros(3, len(self.losses['FPe']))  # TP, FP, FN

                ui = np.unique(self.losses['TC'])[1:]
                for i in ui:
                    j = self.losses['TC'] == float(i)
                    metrics[0, i] = (self.losses['TP'][j] > 0).sum().float()  # TP
                    metrics[1, i] = (self.losses['FP'][j] > 0).sum().float()  # FP
                    metrics[2, i] = (self.losses['FN'][j] == 3).sum().float()  # FN
                metrics[1] += self.losses['FPe']

                self.losses['TP'] = metrics[0].sum()
                self.losses['FP'] = metrics[1].sum()
                self.losses['FN'] = metrics[2].sum()
                self.losses['metrics'] = metrics
            else:
                self.losses['TP'] = 0
                self.losses['FP'] = 0
                self.losses['FN'] = 0

            self.losses['nT'] /= 3
            self.losses['TC'] = 0

        return sum(output) if is_training else torch.cat(output, 1)


def load_weights(self, weights_path, cutoff=-1):
    # Parses and loads the weights stored in 'weights_path'
    # @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)

    if weights_path.endswith('darknet53.conv.74'):
        cutoff = 75

    # Open the weights file
    fp = open(weights_path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
    
    
if __name__ == '__main__':
    config_path = 'yolov3.cfg'
    model = Darknet(config_path)
    print(model)
    
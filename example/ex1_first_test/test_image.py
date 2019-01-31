#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:52:49 2018

@author: ubuntu

这是第一个演示程序，采用的是inference_detector()函数作为演示入口

"""

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
# 读取配置文件
cfg = mmcv.Config.fromfile('../configs/faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# 创建模型
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# 加载模型预训练参数
_ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

# test a single image
img = mmcv.imread('test2.jpg')
result = inference_detector(model, img, cfg)
show_result(img, result)

## test a list of images
#imgs = ['test1.jpg', 'test2.jpg']
#for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#    print(i, imgs[i])
#    show_result(imgs[i], result)
    




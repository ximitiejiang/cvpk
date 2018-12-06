#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:58:50 2018

@author: ubuntu

基于slcv新框架runner搭建yolov3

源码参考：https://github.com/eriklindernoren/PyTorch-YOLOv3
源码基于coco_2014数据集，本部分基于voc_s2007进行训练

"""
import torch
from torch.utils.data import DataLoader
from slcv.runner.runner import Runner

from slcv.model.yolov3 import Darknet
from slcv.utils.init import weights_init_normal
from slcv.dataset.voc_yolov3 import VOCDataset, voc_classes
from slcv.cfg.config import Config
import os

# ----------------------0. 固定设置-----------------------------------
cfg = Config().fromfile('../slcv/cfg/cfg_yolov3_voc.py')  # 需要写相对路径

# ----------------------1. 数据---------------------------------------
trainset = VOCDataset(base_path=cfg.data_root, dataset_name='VOC2007',
                      dataset_type='train', classes=voc_classes, 
                      img_size=416, img_ext=".jpg")
trainloader = DataLoader(trainset, batch_size=cfg.batch_size, 
                         shuffle=False, num_workers=1)

# ----------------------2. 模型---------------------------------------
model_cfg_path = os.path.abspath(cfg.model_cfg)
#os.path.dirname(__file__)  + model_dir
model = Darknet(model_cfg_path, img_size=416)
model.apply(weights_init_normal)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# ----------------------3. 训练---------------------------------------
runner = Runner(trainloader, model, optimizer, cfg) # cfg对象也先传进去，想挂参数应该是需要的
runner.register_hooks(cfg.optimizer_config, cfg.log_config, cfg.text_config)
runner.train()



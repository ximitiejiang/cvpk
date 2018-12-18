#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:58:50 2018

@author: ubuntu

基于slcv新框架runner搭建yolov3

初始源码参考：https://github.com/eriklindernoren/PyTorch-YOLOv3
源码基于coco_2014数据集，但该源码在内部逻辑有重大问题，build_target()函数中x,y,w,h跟x1,y1,x2,y2混淆

进一步源码参考：https://github.com/ultralytics/yolov3
源码也基于coco数据集

"""
import torch
from torch.utils.data import DataLoader
from slcv.runner.runner_yolo import RunnerYolo

from slcv.model.yolov3_refer import Darknet  # 换yolo模型
from slcv.utils.init import weights_init_normal
from slcv.dataset.voc_yolov3 import VOCDataset, voc_classes
from slcv.cfg.config import Config
import os

def main():
    # 0. 固定设置
    cfg = Config().fromfile('../slcv/cfg/cfg_yolov3_voc.py')  # 需要写相对路径
    
    # 1. 数据
    trainset = VOCDataset(
        base_path=cfg.data_root, 
        dataset_name='VOC2007',
        dataset_type='train', 
        classes=voc_classes, 
        img_size=416, 
        img_ext=".jpg")
    trainloader = DataLoader(
        trainset, 
        batch_size=cfg.batch_size, 
        shuffle=False, num_workers=1)
    
    # 2. 模型
    model_cfg_path = os.path.abspath(cfg.model_cfg)
    model = Darknet(model_cfg_path, img_size=416)
    model.apply(weights_init_normal)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=cfg.lr0, 
        momentum=.9)
    
    # 3. 训练
    runner = RunnerYolo(trainloader, model, optimizer, cfg) 
    runner.register_hooks(
            cfg.optimizer_config,
            cfg.checkpoint_config,
            cfg.logger_config)
    # 恢复训练
    if cfg.resume_from is not None:
        runner.resume(cfg.resume_from, resume_optimizer=True, map_location='default')  # 确保map_location与cfg的定义一致
    # 加载模型做inference
    elif cfg.load_from is not None:
        runner.load_checkpoint(cfg.load_from)
        
    runner.train()

if __name__=='__main__':
    main()

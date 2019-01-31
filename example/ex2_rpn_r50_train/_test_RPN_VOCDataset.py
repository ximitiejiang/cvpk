#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:54:32 2018

@author: ubuntu
"""
from mmcv import Config
from mmdet.datasets import get_dataset
from mmdet.datasets import build_dataloader

def main():
    # step1 导入cfg
    cfg = Config.fromfile('./cfg_rpn_r50_fpn_1x.py') # 自定义基于voc的cfg
    
    # step2 加载dataset
    train_dataset = get_dataset(cfg.data.train)
    
    # step3 创建data_loader
    data_loaders = [
        build_dataloader(
            train_dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)]
    for j, batch_data in enumerate(data_loaders):
        if j == 0:
            a,b,c,d,e,f,g,h = batch_data
            return j, batch_data


if __name__ == '__main__':
    j, batch_data = main()

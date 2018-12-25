#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:54:32 2018

@author: ubuntu
"""
from mmcv import Config
from mmdet.datasets import get_dataset

def main():

    cfg = Config.fromfile('./cfg_rpn_r50_fpn_1x.py') # 自定义基于voc的cfg

    train_dataset = get_dataset(cfg.data.train)



if __name__ == '__main__':
    main()
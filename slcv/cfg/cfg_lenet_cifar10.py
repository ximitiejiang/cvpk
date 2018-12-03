#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:40:50 2018

@author: ubuntu

这个config文件模板参考mmcv

"""

# -------------------model----------------------------------
model = 'resnet18'

# -------------------datasets----------------------------------
data_root = '/home/ubuntu/MyDatasets/CIFAR10'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size = 64

# -------------------optimizer----------------------------------
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=2)

# -------------------runtime----------------------------------
work_dir = './demo'   # 在examples目录下自添加一个demo文件夹
gpus = range(2)
dist_params = dict(backend='nccl')
data_workers = 2  # data workers per gpu
checkpoint_config = dict(interval=1)  # save checkpoint at every epoch
workflow = [('train', 1), ('val', 1)]
total_epochs = 6
resume_from = None
load_from = None

# -------------------logging----------------------------------
log_level = 'INFO'
log_config = dict(interval=50,  # log at every 50 iterations
                  hooks=[
                  dict(type='TextLoggerHook'),
                  # dict(type='TensorboardLoggerHook'),
                  ])
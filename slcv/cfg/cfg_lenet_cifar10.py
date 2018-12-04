#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:40:50 2018

@author: ubuntu

这个config文件模板参考mmcv
"""



"""--------------------model-------------------------------
用于定义模型相关参数，如下是相关必选参数，请不要修改变量名
model
"""
model = 'resnet18'

"""--------------------datasets-------------------------------
用于定义数据集相关参数，如下是相关必选参数，请不要修改变量名
data_root
mean
std
batch_size
"""
data_root = '/home/ubuntu/MyDatasets/CIFAR10'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
num_classes = 10
input_layers = 3
batch_size = 64

"""--------------------optimizer-------------------------------
用于优化器相关参数，如下是相关必选参数，请不要修改变量名
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.SGD()


"""
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=2)

"""--------------------runtime-------------------------------
运行时相关参数，如下是相关必选参数，请不要修改变量名
total_epochs
"""
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
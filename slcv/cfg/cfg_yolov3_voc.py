#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:40:50 2018

@author: ubuntu

这个config文件模板参考mmcv
"""

import sys

"""--------------------model-------------------------------
用于定义模型相关参数，如下是相关必选参数，不可修改变量名
model
"""
model_name = 'darknet'
model_cfg = '../slcv/model/yolov3.cfg'

"""--------------------datasets-------------------------------
用于定义数据集相关参数，如下是相关必选参数，不可修改变量名
dataset_name/data_root/mean/std/num_classes/input_size/input_layers/batch_size
"""
dataset_name = 'VOC2007'       # 如果是exist_datasets则需要输入跟exist_datasets要求的名字

data_root = '/home/ubuntu/MyDatasets/voc/VOCdevkit'     # for ubuntu 

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
#num_classes = 10
#input_size = 32   # 图片传入网络的size
input_layers = 3  # 图片传入网络的层数
batch_size = 8   # yolo源码16

"""--------------------optimizer-------------------------------
用于优化器相关参数，如下是相关必选参数，不可修改变量名
optimizer
可定义为字典optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
也可定义为对象optimizer = torch.optim.SGD()
"""
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=5e-4)

"""--------------------runtime-------------------------------
运行时相关参数，如下是相关必选参数，不可修改变量名
epoch_num/
"""
epoch_num = 1   #源码为30
checkpoints_dir = '../checkpoints'
working_dir = '../checkpoints/working'
data_workers = 2  # data workers per gpu
workflow = [('train', 1), ('val', 1)]
resume_from = None
load_from = None

"""--------------------hooks config-------------------------------
用于所有hooks的config，如下是相关比选hook config，不可修改变量名
optimizer_config/log_config/
"""
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=2)  #TODO
text_config = dict(interval = 500)     # 输出iter数的间隔，0表示不输出
checkpoint_config = dict(interval=2)  # save checkpoint at every epoch
log_config = dict(interval=20)        # 每隔n个interval显示，如果interval=0则每个epoch显示

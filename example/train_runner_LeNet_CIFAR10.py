#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:49:59 2018

@author: ubuntu
"""

from slcv.dataset.existdata import exist_datasets, data_transform
from torch.utils.data import DataLoader
from slcv.runner.runner import Runner
from slcv.model.lenet import LeNet5
from slcv.cfg.config import Config
import torch

def main():
    # 0. 固定设置
    cfg = Config().fromfile('../slcv/cfg/cfg_lenet_cifar10.py')  # 需要写相对路径
    
    # 1. 数据
    transform = data_transform(
        train=True, 
        input_size = cfg.input_size, 
        mean = cfg.mean, 
        std = cfg.std)
    trainset = exist_datasets(
        cfg.dataset_name, 
        root=cfg.data_root, 
        train=True, 
        transform =transform, 
        download=False)
    trainloader = DataLoader(
        trainset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=2)
    
    # 2. 模型
    model = LeNet5(num_classes=cfg.num_classes, input_layers=cfg.input_layers)
    if torch.cuda.device_count() > 0 and len(cfg.gpus) == 1:
        model = model.cuda()
    elif torch.cuda.device_count() > 1 and len(cfg.gpus) > 1:  # 数据并行模型
        model = torch.nn.DataParallel(model, device_ids=cfg.gpus).cuda()
    optimizer = cfg.optimizer
    
    # 3. 训练
    runner = Runner(trainloader, model, optimizer, cfg) # cfg对象也先传进去，想挂参数应该是需要的
    runner.register_hooks(
            cfg.optimizer_config, 
            cfg.log_config, 
            cfg.text_config,
            cfg.checkpoint_config)
    # 恢复训练
    if cfg.resume_from is not None:
        runner.resume(cfg.resume_from, resume_optimizer=True, map_location='default')  # 确保map_location与cfg的定义一致
    # 加载模型做inference
    elif cfg.load_from is not None:
        runner.load_checkpoint(cfg.load_from)
    
    runner.train()

if __name__=='__main__':
    main()

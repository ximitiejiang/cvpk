#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:36:46 2018

@author: ubuntu
"""
import torch
import sys
from slcv.hook.hook import Hook

def obj_from_dict(info, parrent=None, default_args=None):
    """基于字典初始化一个对象.
    Args:
        info (dict): dict类型的对象参数,必须包含一个type字段，该type可以等于如下
        的优化器名字('Adam', 'SGD', 'RMSprop'...等价于pytorch优化器描述)
        module (:class:`module`): 目标object类型
        default_args (dict, optional): 初始化object的默认参数
    Returns:
        任何类型
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type') # 获得优化器的名称
    if isinstance(obj_type, str):
        if parrent is not None:
            obj_type = getattr(parrent, obj_type)  # 获得优化器的对应子类
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)

from slcv.model.pretrained_models import pretrained_models
model = pretrained_models('vgg16',10)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
obj_from_dict(optimizer, torch.optim, dict(params=model.parameters()))

class Runner():
    """定义一个runner主类，中间包含train,val,register_hook,call_hook等核心操作函数
    保持柔性可扩展：hooks可以增加，train/val可以合并成run()函数
    """
    def __init__(self, dataloader, model, optimizer, cfg):
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.cfg = cfg
        self._hooks = []
    
    def model_name(self):
        return self.model.__class__.__name__
    
    def init_optimizer(self, optimizer):
        """可以传入一个optimizer对象，也可以传入一个dict参数字典
        输入：optimizer 为module对象或者dict对象
        输出：optimizer 为module对象
        """
        if isinstance(optimizer, dict):
            # 传入优化器参数，优化器
            optimizer = obj_from_dict(optimizer, torch.optim, dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                    'optimizer must be either an Optimizer object or a dict, '
                    'but got {}'.format(type(optimizer)))
        return optimizer
    
    @property
    def hooks(self):
        return self._hooks
    
    def register_hooks(self):
        """注册hooks, 默认hooks包括
        
        """
        pass
    
    def call_hook(self, fn_name):
        """批量调用Hook类中所有hook实例的对应方法"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
    
    def save_checkpoint(self):
        pass
    
    def train(self):
        self.model.train()
        
        self.call_hook('before_run')
        for i in range(num_epoch):
            self.call_hook('before_train_epoch')

            for j, (imgs, labels) in self.dataloader:
                 self.call_hook('before_train_iter')
                 
                 
                 self.call_hook('after_train_iter')
                 
            self.call_hook('after_train_epoch')
        self.call_hook('after_run')
        
    def val(self):
        pass
    
    def test(self):
        pass
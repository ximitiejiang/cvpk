#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:36:46 2018

@author: ubuntu
"""
import torch
import torch.nn.functional as F
import sys
from slcv.hook.hook import Hook
from slcv.hook.optimizer_hook import OptimizerHook

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad(): # with内的tensor均不更新梯度requires_grad =False
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def obj_from_dict(info, parrent=None, default_args=None):
    """基于字典初始化优化器对象.
    Args:
        info (dict): dict类型的对象参数,必须包含一个type字段，该type可以等于如下
        的优化器名字('Adam', 'SGD', 'RMSprop'...等价于pytorch优化器描述)
        module (:class:`module`): 目标object类型
        default_args (dict, optional): 初始化object的默认参数
    Returns:
        目标对象
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


class Runner():
    """定义一个runner主类，中间包含train,val,register_hook,call_hook等核心操作函数
    保持柔性可扩展：hooks可以增加，train/val可以合并成run()函数
    输入：dataloader, 数据加载器
        model，模型对象
        criterion，损失函数
        optimizer，优化器，可以是dict优化器参数也可以是torch.optim.Optimizer对象
        cfg，配置对象
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
    
    def init_criterion(self, criterion):
        """定义损失函数:当前只接收标准pytorch的criterion(loss)
        """
        return criterion
    
    @property
    def hooks(self):
        return self._hooks
    
    def register(self, args, sub_hook_class=None):
        """基于hook config创建hook对象，并加载入_hook变量
        输入：args, 为hook创建参数
            sub_hook_class为hook的子类
        可以是hook对象，此时sub_hook_class就无需输入
        如果是dict，就需要再输入sub_hook_class
        """
        if isinstance(args, Hook):
            hook = args
        elif isinstance(args, dict):
            hook = sub_hook_class(args)
        self._hooks.insert(0, hook)
    
    def register_hooks(self, optimizer_config):
        """注册hooks, 默认hooks包括
        lr_hook, 
        optimizer_hook, 
        checkpoint_hook, 
        iter_time_hook, 
        logger_hook
        """
        if optimizer_config is None:
            optimizer_config = {}
        self.register(optimizer_config, OptimizerHook)
    
    def call_hook(self, fn_name):
        """批量调用Hook类中所有hook实例的对应方法"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
    
    def save_checkpoint(self):
        pass
    
    def train(self):
        self.model.train()
        
        self.call_hook('before_run')
        for i in range(self.cfg.total_epochs):
            self.call_hook('before_train_epoch')

            for j, (imgs, labels) in self.dataloader:
                 self.call_hook('before_train_iter')
                 if torch.cuda.is_available():
                     imgs = imgs.cuda()
                     labels = labels.cuda()
                 
                 pred = self.model(imgs)
                 acc_top1,acc_top5 = accuracy(pred,labels, topk=(1,5))
                 # 复杂loss则通过loss function导入计算            
                 loss = F.cross_entropy(pred, labels, reduction='none')
                 
                 self.outputs = dict(loss = loss.item(), acc_top1=acc_top1.item(), acc_top5 = acc_top5.item()) 
                 self.call_hook('after_train_iter')
                 
            self.call_hook('after_train_epoch')
        self.call_hook('after_run')
        
    def val(self):
        pass
    
    def test(self):
        pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:36:46 2018

@author: ubuntu
"""
import torch
import sys, os, time
from collections import OrderedDict
from slcv.hook.hook import Hook
from slcv.hook.optimizer_hook import OptimizerHook
from slcv.hook.visdom_logger_hook import VisdomLoggerHook
from slcv.hook.timer_hook import TimerHook
from slcv.hook.text_hook import TextHook
from ..hook.log_buffer import LogBuffer
from slcv.utils.checkpoint import load_checkpoint

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
        self.dataloader = dataloader
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.cfg = cfg
        self._hooks = []
        self.log_buffer = LogBuffer()
        self._iter = 0
        self._epoch = 0
    
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
    
    def register(self, sub_hook_class, args=None):
        """基于hook config创建hook对象，并加载入_hook变量
        输入：args, 为hook创建参数
            sub_hook_class为hook的子类
        可以是hook对象，此时sub_hook_class就无需输入
        如果是dict，就需要再输入sub_hook_class
        """
        if isinstance(args, Hook):
            hook = args                  # 创建现成hook
        elif isinstance(args, dict):
            hook = sub_hook_class(args)  # 创建带参hook对象
        elif args is None:
            hook = sub_hook_class()      # 创建不带参hook
        else:
            raise TypeError('args should be hook obj or dict type')
        self._hooks.insert(0, hook)      # 加入_hooks数组,最后一个hook放最前面，也最先执行
    
    def register_hooks(self, 
                       optimizer_config,
                       log_config=None,
                       text_config=None):
        """注册hooks, 默认hooks包括
        OptimizerHook(带配置文件), 
        TimerHook
        可选：
        checkpoint_hook, 
        iter_time_hook(带配置文件), 
        logger_hook(带配置文件)
        """
        self.register(OptimizerHook, optimizer_config)
        self.register(TimerHook)
        
        if log_config is not None:
            self.register(VisdomLoggerHook, log_config)
        if text_config is not None:
            self.register(TextHook, text_config)
        
    
    def call_hook(self, fn_name):
        """批量调用Hook类中所有hook实例的对应方法"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
    
    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        """加载checkpoint
        输入：filename, map_location(跟torch.load函数一样)，strict(是否允许不同参数)
        返回dict
        """
        if not os.path.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        # 加载checkpoint
        checkpoint = torch.load(filename, map_location = map_location)
        
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(filename))
            
        return load_checkpoint(self.model, filename, map_location, strict)
    
    def save_checkpoint(self, out_dir, filename, save_optimizer=True, meta=None):
        """保存checkpoint到文件
        输入：meta 保存version和time, dict, 默认是{'epoch':epoch, 'iter':iter}
              model
        输出：OrderedDict
        """
        if meta is None:
            meta = dict(epoch=self._epoch +1, iter = self._iter)
        else:
            meta.update(epoch=self._epoch +1, 
                        iter = self._iter,
                        time = time.time())
        # 判断是否保存optimizer
        filepath = os.path.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        # 从GPU拷贝state_dict到cpu
        state_dict_cpu = OrderedDict()
        for key, val in self.model.state_dict.items():
            state_dict_cpu[key] = val.cpu()
        # 生成checkpoint    
        checkpoint ={'meta': meta,
                     'state_dict': state_dict_cpu}
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        # 保存checkpoint
        torch.save(checkpoint, filepath)
        
    
    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)
#        if torch.cuda.is_available():
#            self.model.cuda()
            
        self.model.train()
        
        self.call_hook('before_run')
        for i in range(self.cfg.epoch_num):
            self.call_hook('before_train_epoch')
            
            for j, (imgs, labels) in enumerate(self.dataloader):
                 self.call_hook('before_train_iter')
                 if torch.cuda.is_available():
                     imgs = imgs.to(device)
                     labels = labels.to(device)
                 
                 pred = self.model(imgs)
                 # 复杂loss则通过loss function导入计算            
                 loss = torch.nn.CrossEntropyLoss()(pred,labels)
                 
                 # outputs作为汇总变量，传入hooks
                 acc_top1,acc_top5 = accuracy(pred,labels, topk=(1,5))
                 log_vars = OrderedDict()
                 log_vars['loss'] = loss.item()
                 log_vars['acc_top1'] = acc_top1.item()
                 log_vars['acc_top5'] = acc_top5.item()
                 
                 self.outputs = dict(loss=loss, log_vars=log_vars, num_samples=imgs.size(0))
                 self.log_buffer.update(self.outputs['log_vars'])
                 self.call_hook('after_train_iter')
                 
                 self._iter += 1
            self.call_hook('after_train_epoch')
            self._epoch += 1
        self.call_hook('after_run')
        
    def val(self):
        pass
    
    def resume(self):
        pass
    
    def test(self):
        pass
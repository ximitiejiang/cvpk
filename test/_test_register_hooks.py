#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:33:28 2018

@author: ubuntu
"""
from slcv.cfg.config import Config
from slcv.hook.hook import Hook
from slcv.hook.optimizer_hook import OptimizerHook
from slcv.hook.visdom_logger_hook import VisdomLoggerHook

class Runner():
    def __init__(self):
        self._hooks = []
    
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
            hook = sub_hook_class(args)  # 创建hook对象
        self._hooks.insert(0, hook)      # 加入_hooks数组
    
    def register_hooks(self, 
                       optimizer_config,
                       log_config):
        """注册hooks, 默认hooks包括
        lr_hook, 
        optimizer_hook, 
        checkpoint_hook, 
        iter_time_hook, 
        logger_hook
        """
        self.register(optimizer_config, OptimizerHook)
        self.register(log_config, VisdomLoggerHook)
    
    def call_hook(self, fn_name):
        """批量调用Hook类中所有hook实例的对应方法"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)


def _test_register_hooks():
    cfg = Config().fromfile('../slcv/cfg/cfg_lenet_cifar10.py')
    runner = Runner()
    runner.register_hooks(cfg.optimizer_config, cfg.log_config)  # 初始化hooks
    runner.call_hook('test_hook')   # 调用hook方法(第一个子类重写该方法，第二个子类调用父类方法)
    
    
if __name__ == '__main__':
    _test_register_hooks()  # 调试通过: 可以初始化hooks,可以调用hooks方法
    
    
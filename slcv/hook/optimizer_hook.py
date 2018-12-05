#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:17:10 2018

@author: ubuntu
"""

from torch.nn.utils import clip_grad
from .hook import Hook

class OptimizerHook(Hook):
    """优化器hook
    """
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        """每个循环结束：优化器清零，基于损失计算梯度(loss.backward)，基于梯度用优化器更新参数(optimizer step)，"""
        runner.optimizer.zero_grad()            # 清零上一个iter的梯度
        runner.outputs['loss'].backward()       # 计算梯度
#        if self.grad_clip is not None:
#            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()                 # 更新参数

if __name__ == '__main__':
    h = OptimizerHook()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:17:10 2018

@author: ubuntu
"""

from torch.nn.utils import clip_grad
from hook import Hook

class OptimizerHook(Hook):
    """优化器hook，默认用于loss.backward()
    输入：grad_clip, 为真则对梯度进行求均值
    """
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        """每个循环结束：优化器清零，"""
        runner.optimizer.zero_grad()
        runner.loss.backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()

if __name__ == '__main__':
    h = OptimizerHook()
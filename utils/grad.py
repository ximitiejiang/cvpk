#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:16:42 2018

@author: ubuntu
"""

def zero_grad(self):
    '''每一次batch_size的循环，optimizer的梯度清零算法如下：
    optimizer类会建立一个param_group属性，self.param_group用于存放模型参数和优化器参数
    param_group是一个只有一个元素的list，这个元素是一个dict
    查看方式：
        optimizer.param_group[0].keys()
        包含的参数包括：
            - 'params'  代表model.parameters
            - 'lr'
            - 'momentum'
            - 'dampening'
            - 'weight_decay'
            - 'nesterov'
            ...
    '''
    r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
    
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

if __name__=='__main__':
    pass
    
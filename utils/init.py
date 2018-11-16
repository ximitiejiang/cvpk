#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 09:27:12 2018

@author: suliang
"""
import math
def fc_reset_parameters(weight, bias):
    '''该代码是在linear class层中的子函数，对该模型的参数w,b进行初始化
    为了便于调试，去掉了self.
    输入：weight，bias 均为tensor
    输出：直接修改weight，bias的值
    
    公式：
    
    '''
    stdv = 1. / math.sqrt(weight.size(1))
    weight.data.uniform_(-stdv, stdv)
    if bias is not None:
        bias.data.uniform_(-stdv, stdv)
        
        
def conv_reset_parameters(in_channels, kernel_size, weight, bias):
    '''该代码是在_ConvNd class中的子函数，对该模型的参数w,b进行初始化
    为了便于调试，去掉了self.
    输入：weight，bias 均为tensor
    输出：直接修改weight，bias的值
    
    公式：
    
    '''
    n = in_channels
    for k in kernel_size:
        n *= k
    stdv = 1. / math.sqrt(n)
    weight.data.uniform_(-stdv, stdv)
    if bias is not None:
        bias.data.uniform_(-stdv, stdv)


def other_init_way():
    from torch import nn
    nn.init.uniform()


if __name__ == '__main__':
    import torch
    weight= torch.zeros(3,3)
    bias = torch.zeros(1,1)
    
    fc_reset_parameters(weight, bias)
    
    
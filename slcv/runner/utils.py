#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:19:15 2018

@author: ubuntu
"""

import torch
import sys

def accuracy(output, target, topk=(1, )):
    """计算输出与标签相比的预测准确率
    输入：output   网络输出, 比如[64,10]代表64张图片，10类中每类概率
          target   标签，比如[64]代表每张图的一个标签(0到9的一个数)
    输出：res      预测结果[top1, top5]，top1为取最大概率值作为预测值得到的精度
                   top5为取最大概率的前5个值只要有一个正确的精度。
    Computes the precision@k for the specified values of k
    """
    with torch.no_grad(): # with内的tensor均不更新梯度requires_grad =False
        maxk = max(topk)  # 5
        batch_size = target.size(0)  # 64
        # output [64, 10] 中取最有可能(最大)的5个概率 -> [64, 5]
        # tensor.topk(k,dim,largest=True,sorted=True), 沿dim方向，K个最大值，排序
        _, pred = output.topk(maxk, 1, True, True)  # dim=1为列方向5个最大值，pred=[64, 5]
        pred = pred.t()   # [64,5] -> [5,64]
        # target从[64] -> [1,64] -> [5,64]
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # correct [5,64]相等为1,不等为0

        res = []
        for k in topk:
            # correct从[1,64] -> [64]
            # correct从[5,64] -> [320] -> float 求和，为topk类预测的正确个数(每个batch)
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # 除以batch张数，就是平均每张图预测正确率
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def obj_from_dict(info, parrent=None, default_args=None):
    """基于字典,父类, 参数，来生成一个对象
    Args:
        %info (dict)    dict类型的对象参数,需要type字段定义子类，该type可以等于如下
                        的优化器名字('Adam', 'SGD', 'RMSprop'...等价于pytorch优化器描述)
        % parrent       目标object父类
        % default_args  初始化object的默认参数
    Returns:
        目标对象
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type') # 获得子类的名称
    if isinstance(obj_type, str):
        if parrent is not None:
            obj_type = getattr(parrent, obj_type)  # 基于父类和子类，创建子类对象
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)
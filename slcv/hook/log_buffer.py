#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:18:36 2018

@author: ubuntu
"""
from collections import OrderedDict
import numpy as np

class LogBuffer():
    """log_buffer类用来存储在计算过程中临时需要累积求平均值的数据，比如loss
    每个iter得到的log_vars = {'loss': float, 'acc_top1':float, 'acc_top5':float}
    会存放在这里
    """
    def __init__(self):
        """value_history {'loss':[], 'acc_top1':[], 'acc_top5':[], 
                            'iter_time':[], 'epoch_time':[]}
           count_history {'loss':[], 'acc_top1':[], 'acc_top5':[], 
                            'iter_time':[], 'epoch_time':[]}
           内置变量可以有不同的list长度。
        """
        self.value_history = OrderedDict()   # 用于存储数据
        self.count_history = OrderedDict()   # 用于存储数据次数
        self.average_output = OrderedDict()  # 用于存储阶段性输出值
        
        self.ready = False
        
    def clear(self):
        self.value_history.clear()
        self.count_history.clear()
        self.ready = False
    
    def update(self, data):
        """把字典数据存入buffer,存入形式为{'loss':[], 'acc_top1':[], 'acc_top5':[]}
        dict与list组合存储数据是最常用的形式。
        """
        assert isinstance(data, dict)
        for key, var in data.items():
            if key not in self.value_history:
                self.value_history[key] = []
                self.count_history[key] = []
            self.value_history[key].append(var)
            self.count_history[key].append(1)
        self.ready = False
    
    def average(self, interval):
        """计算平均值： 如果要interver=n进行一次平均，则可增加n参数
        log_buffer.average_buffer是一个平均值字典{'loss':float, 'acc_top1':float, 'acc_top5':float}
        """
        interval = int(interval)
        for key in self.value_history:
            val = np.array(self.value_history[key][-interval:])
            num = np.array(self.count_history[key][-interval:])
            avg = np.sum(val * num) / np.sum(num)  # array的*代表逐个相乘
            self.average_output[key] = avg
        self.ready = True

if __name__ == '__main__':
    lb = LogBuffer()
    data1 = OrderedDict(loss=0.3, acc_top1=0.5, acc_top5=0.8)
    data2 = OrderedDict(loss=0.1, acc_top1=0.9, acc_top5=0.1)
    lb.update(data1)
    lb.update(data2)
    
    lb.average()
    
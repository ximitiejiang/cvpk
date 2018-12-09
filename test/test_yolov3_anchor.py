#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:16:39 2018

@author: ubuntu
"""

"""yolov3的模型总共3个yolo layer
每个yolo layer的anchor参数是：
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
每2个数据为一个anchor，每3个anchor为一组。
每个anchor的数据含义[10,13]代表anchor的w, h
"""
import numpy as np

anchor1 = [(10,13),(16,30),(33,23)]
anchor2 = [(30,61),(62,45),(59,119)]
anchor3 = [(116,90),(156,198),(373,326)]


def _rectangle(ab,color = 'r'):  # (ymin, xmin, ymax, xmax)
    '''绘制一个矩形: 需要在该程序运行之前创建一个figure (plt.figure())
    input: ab = [ymin, xmin, ymax, xmax]
    '''
    ab = np.array(ab)
    ab = ab.reshape(1,4)
    xmin = np.asscalar(ab[0,1])
    xmax = np.asscalar(ab[0,3])
    ymin = np.asscalar(ab[0,0])
    ymax = np.asscalar(ab[0,2])

    x = [xmin, xmin]
    y = [ymin, ymax]
    plt.plot(x,y, color)
    
    x = [xmin, xmax]
    y = [ymax, ymax]
    plt.plot(x,y, color)
    
    x = [xmin, xmax]
    y = [ymin, ymin]
    plt.plot(x,y,color)
    
    x = [xmax, xmax]
    y = [ymin, ymax]
    plt.plot(x,y, color)

import matplotlib.pyplot as plt
f = plt.figure()

"""分别绘制3组anchor,由小到达

"""
ab=[]
for i, anchor in enumerate(anchor1):
    ab.append([-anchor[1]/2, -anchor[0]/2, anchor[1]/2, anchor[0]/2])
    _rectangle(ab[i], color='r')
ab=[]    
for i, anchor in enumerate(anchor2):
    ab.append([-anchor[1]/2, -anchor[0]/2, anchor[1]/2, anchor[0]/2])
    _rectangle(ab[i], color='g')
ab=[]    
for i, anchor in enumerate(anchor3):
    ab.append([-anchor[1]/2, -anchor[0]/2, anchor[1]/2, anchor[0]/2])
    _rectangle(ab[i], color='b')
    




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:20:03 2018

@author: ubuntu
"""

import six
from six import __init__
import numpy as xp
import numpy as np
import matplotlib.pyplot as plt

def RectAngle(ab,color = 'r'):
    '''绘制一个矩形: 需要在该程序运行之前创建一个figure (plt.figure())
    input: ab = [ymin, xmin, ymax, xmax]
    
    '''
    ab = np.array(ab).reshape(1,4)
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


def loc2bbox(src_bbox, loc):
    '''loc2bbox的功能是： 基于候选anchor的坐标和RPN坐标回归网络输出的rpn_loc参数[dy,dx,dh,dw]，
    对候选的anchor进行tweaking微调，得到更接近实际的anchor
    最终输出调整后的dst_bbox (destination bbox)
    
    2组可用的变换前后参数
    src_bbox_1 = np.array([-37.254833, -82.50967 ,  53.254833,  98.50967 ])
    dst_bbox_1 = np.array([-37.02167 , -84.59462 ,  51.731297, 105.54578 ])      
    
    src_bbox_2 = np.array([ -82.50967, -173.01933,   98.50967,  189.01933])
    dst_bbox_2 = np.array([ -75.93692 , -173.06662 ,  109.633286,  191.19025 ])
    
    '''
    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    # anchor输入是(ymin,xmin,ymax,xmax),这里转化为中心点y,x,h,w
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width
    # 这是在炫技吗，loc[:,0],loc[:,1],loc[:,2],loc[:,3]
    # 获得位置预测的y,x,h,w
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]
    # 
    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]
    
    # 分别得到(ymin,xmin,ymax,xmax)
    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    '''bbox2loc的功能是： 基于输入的候选anchor，以及对应目标anchor，
    计算相应的回归参数放入loc变量
    最终输出参数loc
    '''
    # src代表source bbox, dst代表destination bbox
    # 先求源bbox的h,w,centrl_x, centrol_y
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    # 再求目标bbox的h,w,centrl_x, centrol_y
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 
    eps = xp.finfo(height.dtype).eps
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)
    
    # [1,4] - > [4,1]
    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc


def _test_loc2bbox():
    '''基于2组从代码中获得的微调前的bbox和微调后的bbox，查看回归参数对bbox的影响
    '''
    src_bbox_1 = np.array([-37.254833, -82.50967 ,  53.254833,  98.50967 ])
    dst_bbox_1 = np.array([-37.02167 , -84.59462 ,  51.731297, 105.54578 ])      
    
    src_bbox_2 = np.array([ -82.50967, -173.01933,   98.50967,  189.01933])
    dst_bbox_2 = np.array([ -75.93692 , -173.06662 ,  109.633286,  191.19025 ])
    
    plt.figure()
    RectAngle(src_bbox_1,'r')
    RectAngle(dst_bbox_1, 'g')
    
    plt.figure()
    RectAngle(src_bbox_2,'r')
    RectAngle(dst_bbox_2, 'g')

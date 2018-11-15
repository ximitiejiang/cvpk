#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:59:21 2018

@author: ubuntu
"""
import numpy as np
import numpy as xp

def bbox_iou(bbox_a, bbox_b):
    '''用于计算两个bbox之间的IOU值，方法很巧妙
       利用np.maxmum(a,b)函数，能够求得两个数据按位比较大小的结果
       然后
    input:
        bbox_a: [ymin,xmin,ymax,xmax]
        bbox_b: [ymin,xmin,ymax,xmax]
    '''

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


if __name__=='__main__':
    src_bbox_1 = np.array([-37.254833, -82.50967 ,  53.254833,  98.50967 ]).reshape(1,-1)
    dst_bbox_1 = np.array([-37.02167 , -84.59462 ,  51.731297, 105.54578 ]).reshape(1,-1) 
    
    iou = bbox_iou(src_bbox_1, dst_bbox_1)
    print(iou)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:16:55 2018

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt

import six
from six import __init__
import torch

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    '''生成9个anchor base
    '''
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def enumerate_shifted_anchor(anchor_base, feat_stride=16, height=40, width=30):
    '''基于9个anchor base，stride，以及送进来的features的高宽(应该是input img的1/16)
    来生成可用的20000个左右的anchor
    '''
    import numpy as xp
    # 先对x, y 轴方向的原图进行间隔划分，一个800x600的图就可以划分成50x37个值，间隔16
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    # 然后meshgrid后把划分扩展到整张图每行每列，所以x有37x50行，y有37x50
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    # 然后把x,y堆跌，每一行代表一个点坐标，不过有2个x，2个y（y,x,y,x），一共1850行，代表1850个像素点
    # 这个(y,x,y,x)后续平移后就得到(y+ymin,x+xmin, y+ymax,x+xmax)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    # 最后用base anchor [1,9,4]进行平移，即跟[1850,1,4]这样的数组进行相加平移。
    # 利用广播原则，相当与1850个anchor，每个anchor的4个坐标分别跟anchor base相加
    # 最终得到每个像素点位置的9个anchor坐标：(y+ymin,x+xmin, y+ymax,x+xmax)
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


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



def _draw_anchor_base():
    '''绘制anchor base = 9个anchor
       绘制enumerate shift anchor = hh x ww x 9 个
       （为便于观察，只画出了每组最中间那个正方形）
    '''
    ab = generate_anchor_base()  # 生成9个anchor base, (ymin,xmin, ymax,xmax)
    plt.figure()
    plt.title('Anchor base(num=9)')
    for i in range(len(ab)):
        _rectangle(ab[i])



def _draw_enumerated_anchors():
    '''基于anchor base作为输入绘制枚举生成的所有anchors
    '''
    ab = generate_anchor_base()
    sa = enumerate_shifted_anchor(ab, 16, 37, 50)    
    print('total anchors are: {}'.format(len(sa)))
    plt.figure()
    plt.title('Enumerated anchor(num=20,000)')
    ii=1

    for i in range(len(sa)):
        if i%9 == 3:
            _rectangle(sa[i].reshape(1,-1))         # 画出每个位置的中间第3个anchor   
            ii+=1

def _plot_img_and_bbox():
    '''绘制图片和所属的所有bbox
    '''
    from utils.config import opt
    from data.dataset import Dataset
    from torch.utils import data
    
    train = Dataset(opt)
    dataloader = data.DataLoader(train,
                            batch_size=1,
                            num_workers=8,
                            shuffle=True
                            )
    (img,bbox,label,scale) = next(iter(dataloader))

    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch
    # tensor to pil
    img1 = (img.numpy()*0.225+0.45).clip(min=0,max=1)
    img1 = img1.squeeze(0)
    img1 = transforms.ToPILImage()(torch.from_numpy(img1.astype(np.float32)))  
    plt.imshow(img1)
 
    bbox = bbox.numpy()[0]
    for i in range(len(bbox)):
        _rectangle(bbox[i])   # [[x,y,x,y]]
        
    return img, bbox, label, scale



if __name__=='__main__':
    
    _draw_anchor_base()
    
    _draw_enumerated_anchors()
    
    _plot_img_and_bbox()
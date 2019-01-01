#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:40:48 2018

@author: ubuntu

测试图片预处理的几个步骤

#   img的处理: 0.read, 1.scale/resize, 2.normalize, 3.pad, 4.flip, 5.transpose, 6.to tensor
#   gtbbox处理：0.load, 1.scale, 2.flip, 3.clip

"""

# 以下探讨一个最简实现

import numpy as np
import mmcv
import os.path as osp
import cv2

# --------------------------0. read--------------------------
# 关键1：区别读入的差别，cv2.imread()读入的是(H,W,C),而PIL的PIL.imread()读入的是(C,H,W)
#
def imread(img_or_path, flag='color'):
    """Read an image.

    Args:
        img_or_path (ndarray or str): Either a numpy array or image path.
            If it is a numpy array (loaded image), then it will be returned
            as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        flag = imread_flags[flag] if is_str(flag) else flag
        check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
        return cv2.imread(img_or_path, flag)
    else:
        raise TypeError('"img" must be a numpy array or a filename')

#img = mmcv.imread(osp.join(img_prefix, img_info['filename']))
#img_infos = mmcv.load(ann_file)

# --------------------------1. scale--------------------------
# 实例：输入img_scale=(1000, 600)
# 先通过img_scale = random_scale(img_scales)得到一个随机缩放值，然后再缩放
def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)  # 这里都输入的是[(1000, 600)]
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]  # 等于(1000, 600)
        
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':  # 范围缩放模式，则分别返回长边和短边的缩放范围(long_edge, short_edge)
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':  # 单值缩放模式，则随机从所给范围中取一个值作为缩放值
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale

def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple): w, h.
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}
def imresize(img, size, return_scale=False, interpolation='bilinear'):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple): Target (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float or tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]  # 假设h,w = 375,500
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(
                'Invalid scale {}, must be positive.'.format(scale))
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale) # 1000
        max_short_edge = min(scale) # 600
        # 这句是定义缩放比例的关键：定义的(1000,600)是图片的最大尺寸，
        # min(1000/500, 600/375)=min(2,1.6)
        # 基本对象检测算法都是这种scale方式，包括faster rcnn
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            'Scale must be a number or tuple of int, but got {}'.format(
                type(scale)))
    new_size = _scale_size((w, h), scale_factor)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img

# --------------------------1.2. resize--------------------------



# --------------------------2. normalize--------------------------
def convert_color_factory(src, dst):

    code = getattr(cv2, 'COLOR_{}2{}'.format(src.upper(), dst.upper()))

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img
    convert_color.__doc__ = """Convert a {0} image to {1} image.
    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted {1} image.
    """.format(src.upper(), dst.upper())
    return convert_color

bgr2rgb = convert_color_factory('bgr', 'rgb')

def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb:
        img = bgr2rgb(img)
    return (img - mean) / std

# --------------------------3.pad_to_multiple --------------------------
# 实例：输入img, divisor=32
def impad(img, shape, pad_val=0):
    """Pad an image to a certain shape.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple): Expected padding shape.
        pad_val (number or sequence): Values to be filled in padding areas.

    Returns:
        ndarray: The padded image.
    """
    # 
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img.shape[-1]
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1], )
    assert len(shape) == len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    pad = np.empty(shape, dtype=img.dtype)
    pad[...] = pad_val
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad

def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (number or sequence): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    # 为了得到divisor的倍数，先除以倍数得到的商上取整再乘以倍数即可得到需要的h,w
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, (pad_h, pad_w), pad_val)



# --------------------------4. flip--------------------------
def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or "vertical".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    else:
        return np.flip(img, axis=0)


# --------------------------5. transpose--------------------------
#img = img.transpose(2, 0, 1)


# --------------------------6. to tensor--------------------------
from mmcv import Config
#from mmdet.datasets import get_dataset
from PIL import Image
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
if __name__ == '__main__':
    
    id = 3
    
    if id == 1: # 验证数据集出来的数据是不是处理的到位了(所有transform都加进去了)
        cfg = Config.fromfile('./cfg_rpn_r50_fpn_1x.py') # 自定义基于voc的cfg
        train_dataset = get_dataset(cfg.data.train)
        data = train_dataset[1]
        print(data['img'].data.shape)
    
    if id == 2: #处理一张照片: 用opencv + python算法
        path = 'test.jpg'
        img = cv2.imread(path)    # cv2.imread()读出来1的是(h, w, c)=(350,500,3)
        img_scales=(1000, 600)
        img_scales = img_scales if isinstance(img_scales, list) else [img_scales]  # 转化为list[(1000, 600)]
        # 缩放
        img_scale = random_scale(img_scales)
        img = imrescale(img, img_scale) # 转换成()
        # normalize
        
        #

    if id==3:  #处理一张照片，用pytorch
        path = 'test.jpg'
        img = Image.open(path)     # PIL.Image.open()读出来的是(w,h)
        plt.imshow(img)
        transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                        ])
        img = transform(img)
    

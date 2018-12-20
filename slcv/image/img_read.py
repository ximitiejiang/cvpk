#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:17:11 2018

@author: ubuntu
"""


import cv2
import numpy as np

from mmcv.utils import is_str, check_file_exist
from mmcv.opencv_info import USE_OPENCV2

if not USE_OPENCV2:
    from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
else:
    from cv2 import CV_LOAD_IMAGE_COLOR as IMREAD_COLOR
    from cv2 import CV_LOAD_IMAGE_GRAYSCALE as IMREAD_GRAYSCALE
    from cv2 import CV_LOAD_IMAGE_UNCHANGED as IMREAD_UNCHANGED

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}


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
        
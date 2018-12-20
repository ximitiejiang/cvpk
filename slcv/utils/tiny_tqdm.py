#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 12:16:27 2018

@author: ubuntu

该tqdm参考自pytorch model_zoo.可用于自定义一个tqdm显示进度
"""
import sys

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # defined below
    
if tqdm is None:
    # fake tqdm if it's not installed
    class Tqdm(object):

        def __init__(self, total, disable=False):
            self.total = total
            self.disable = disable
            self.n = 0

        def update(self, n):
            if self.disable:
                return

            self.n += n
            sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.disable:
                return

            sys.stderr.write('\n')

if __name__ =='__main__':
    tqdm = Tqdm(100)
    

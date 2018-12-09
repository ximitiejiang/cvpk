# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

1. parse_argue模块跟config文件本质一样，都是创建了对象，把配置放入对象cfg的属性里边
区别在于，parse_argue是通过parser.add_argument()实现添加对象属性；
而config文件是通过Config类定义了__getitem__()实现，本质上其实跟parse_argue一样

2. cfg这个变量引用内部参数的方式：cfg.root = ....这种引用模式是对象的引用


"""

import tensorflow as tf
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs')  # 创建一个writer对象，并在根目录创建了‘runs’文件夹

writer.add_scalar('Train/Loss', loss.data[0], niter)
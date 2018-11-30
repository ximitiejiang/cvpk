#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:52:42 2018

@author: ubuntu
"""
#import sys
#sys.path.insert(0, '/home/ubuntu/suliang_git/slcv')
"""如果导入模块报错说找不到，是因为系统调用模块分3步，
第一步搜索sys path, 第二步搜索，第三部在__main__函数入口的下层目录搜索

这种情况，__main__函数入口下层肯定搜索不到，所以必须把模块路径加到sys path
例如：添加'/home/ubuntu/suliang_git/slcv', 之后就会在这个slcv路径之下按顺序
分别搜索slcv, utils, anchor_base三个模块，所以只要确保在大的slcv文件夹下面
包含的了对应3个分层模块就能完成导入，其中slcv/utils两个都是添加__init__把包
转换成模块了。

"""

from slcv.utils.anchor_base import generate_anchor_base   # 这个导入命令，在进入内层slcv文件夹后失效，但在外层slcv下可行

z = generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32])


"""
os模块基本命令
import os
os.path.abspath(__file__)  # 返回当前运行文件的绝对路径：包含文件本身名字
os.path.dirname(path)   # 返回文件所在路径：不包含文件本身名字。相当与往上一级，可多次往上。
os.path.basename(path)  # 返回文件名: dirname + basename就等于abspath
os.path.join(path1, path2,path3...)  # 组合路径名
os.path.isdir(path)   # 判断是否为路径
os.path.isfile(path)   # 判断是否为文件


方向1： 
import sys
print(sys.path)
可以查看相关已经添加的系统路径.
尝试把slcv的路径添加进去，这样相当与运行了slcv的init
sys.path.insert(0,'/home/ubuntu/suliang_git/slcv')

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

方向2：
在python3.7目录中增加pth文件
/home/ubuntu/anaconda3/lib/python3.7/site-packages/torchvision-0.2.1-py3.7.egg
"""
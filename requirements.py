#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:58:58 2018

@author: ubuntu
"""

"""-----step1. create symblic link to datasets-----------
$cd slcv
$mkdir data
$ln -s /home/ubuntu/MyDatasets/coco
$ln -s /home/ubuntu/MyDatasets/voc/VOCdevkit

go to position slcv and create 'data' dir, to build structure in it as below:
slcv
|——checkpoints
|——example
|——slcv
|——test
|——data
    |——coco
        |——annotations
        |——train2017
        |——val2017
        |——test2017
    |——VOCdevkit
        |——VOC2007
        |——VOC2012
    |——...
"""

"""-----step2. start visdom server------------ ----------
python3 -m visdom.server
"""

"""-----step3. add slcv to sys path---------- -----------
step3的永久加入PYTHONPATH的方法是：
$ sudo gedit /etc/profile
export PYTHONPATH=
"""
import sys, os
rootpath = os.path.dirname(__file__)
sys.path.insert(0, rootpath)

print(sys.path)

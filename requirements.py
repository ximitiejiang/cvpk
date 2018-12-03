#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:58:58 2018

@author: ubuntu
"""
"""
#-----step1-----------
python3 -m visdom.server
"""

#-----step2-----------
import sys, os
rootpath = os.path.dirname(__file__)
sys.path.insert(0, rootpath)

print(sys.path)

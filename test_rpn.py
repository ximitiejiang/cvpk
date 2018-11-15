#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:11:38 2018

@author: ubuntu
"""

from model.RPN import RegionProposalNetwork
from model.vgg import vgg16

extractor, classifier = vgg16()
rpn = RegionProposalNetwork()  # 初始化




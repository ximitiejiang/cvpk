#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:11:38 2018

@author: ubuntu

本文件只用于测试rpn model
    - generate anchor base

"""

from model.rpn import RegionProposalNetwork, generate_anchor_base
from model.vgg_divide import vgg16


anchors = generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32])





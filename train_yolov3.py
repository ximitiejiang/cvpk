#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:10:30 2018

@author: ubuntu
"""
from model.yolov3 import Darknet




# --------3. model----------------
# Initiate model
model_config_path = 'yolov3.cfg'
model = Darknet(model_config_path, img_size=416)

# model.load_weights(opt.weights_path)
model.apply(weights_init_normal)
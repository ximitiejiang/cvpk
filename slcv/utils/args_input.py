#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:48:57 2018

@author: ubuntu
"""
from argparse import ArgumentParser   # 导入类

def input_args_parse():
  """用于输入参数的子程序
  解析器能够获得.py文件运行时的输入参数，存入解析器，然后解析到参数变量arg中
  1. parser = ArgumentParser()           # 创建解析器对象
  2. parser.add_argument(...)               # 解析器添加参数
  3. args = parser.parse_args()          # 解析其解析参数
  """
    parser = ArgumentParser()    
      
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    args = parser.parse_args()
  
    return args


if __name__ == '__main__':

  args = input_args_parse()

  print(args)
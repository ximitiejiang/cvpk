#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:48:57 2018

@author: ubuntu
"""
import argparse

def parse_args():
  """用于输入参数的子程序
  解析器能够获得.py文件运行时的输入参数，存入解析器，然后解析到参数变量arg中
  1. parser = argparse.ArgumentParser()  # 创建解析器
  2. parser.add_argument()               # 解析器添加参数
  3. args = parser.parse_args()          # 解析其解析参数
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  
  args = parser.parse_args()
  return args


if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
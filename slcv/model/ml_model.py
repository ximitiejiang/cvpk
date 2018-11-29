#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:16:38 2018

@author: suliang

这个包主要存储重新写的machine learning models
现已包含：
    - logistic regression
    - softmax regression

"""
from utils.dataload import exist_data 


def logistic_regression(data, labels):
    '''用pytorch的autograd实现逻辑回归，进行分类
    '''
    pass


def softmax_regression():
    '''用pytorch的autograd实现softmax回归，进行分类
    '''
    pass



if __name__ == '__main__':
    
    root = '/Users/suliang/MyDatasets/MNIST'
    trainset = exist_datasets('MNIST', root, train=True, transform =None)
    
    logistic_regression()
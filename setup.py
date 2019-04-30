#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 07:17:30 2019

@author: ubuntu

创建发行包注意事项
1. 安装setuptools
2. 在根目录创建setup.py安装文件
3. 在根目录文件下创建同名文件夹，并在该同名文件夹下创建__init__文件作为包文件，把相关库代码都拷贝到这个同名文件夹下
4. 检查setup.py文件语法: python3 setup.py check
    + 该步要用python3，因为我安装了多个版本的python，通常采用python3(对应python3.7，而python对应3.5)，我的setuptools也是安装在3.7里边
5. 打包: 创建可分发的egg，也就是build distribute egg的意思，python3 setup.py bdist_egg
    + 该步会产生build文件夹/cvpk.egg-info文件夹/dist文件夹
    + 在egg文件中会包含相关安装所需的文件
6. 把egg文件上传或者拷贝到别的机器，解压缩：unzip -l cvpk-0.1-py3.6.egg
6. 安装: 属于从源码安装，先从github下载源码，然后进入该目录运行代码，
    + 开发方式安装: python3 setup.py install develop，在有频繁变更情况下，每次安装都要先卸载原来版本比较麻烦，
      而用develop方式安装，代码不会真的被拷贝到本地python环境site-packages目录，而是在site-package目录创建一个
      指向当前位置的链接，如果当前位置有改动，会马上反映到site-package里边
    + 用户方式安装: python3 setup.py install (or pip3 install .) 该方式会在如下目录产生一个cvpk-0.1-py3.7.egg的解压缩文件夹
      即在python的site-package目录下/home/ubuntu/anaconda3/lib/python3.7/site-packages/cvpk-0.1-py3.7.egg
7. 使用: from cvpk.dataset.class_names import voc_classes
    + 该步只要导入了该lib，就能使用该lib里边的module
8. 查看setup.py的选项： python3 setup.py --help-commands
    + python3 setup.py install          #表示从目录安装
    + python3 setup.py install develop  #表示在开发模式安装
    + python3 setup.py bdist_egg        #表示创建一个egg
    + python3 setup.py easy_install     #表示会寻找相关依赖然后一起安装
    + python3 setup.py check            #表示对安装文件进行语法检查
    + python3 setup.py install --record record.txt   # 
9. 卸载: 如果需要卸载干净，需要在安装时产生记录文件，然后基于记录文件进行删除
    + python3 setup.py install --record record.txt  # 获得安装记录文件
    + sudo rm $(cat record.txt)
    + 其实查看产生的record.txt文件会发现所有文件都在site-package的那个安装目录下，所以手动删除这个文件夹也是可以的
10. 重新安装：当源代码修改过后，就需要重新安装
    + 需要先重新生成egg: python3 setup.py bdist_egg
    + 再重新安装egg: python3 setup.py install --record record.txt

"""

from setuptools import find_packages, setup

setup(
    name = 'cvpk',
    version = '0.1',
    packages = find_packages(),  # 这里通过自动搜索，也可以用packages = ['cvpk']
    description = 'computer vision tools library',
    author='ximi',
    author_email='ximitiejiang@163.com',
    url='https://github.com/ximitiejiang/cvpk')
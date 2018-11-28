#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:06:53 2018

@author: ubuntu
"""
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def read_xml(dir):
    """ xml解析，首先获得root
    0. 以voc的xml为例:有一个主tag为<anotation>, 下面并列8个子tag:
        <folder><filename><source><owner><size><segmented><object><object>
    1. root.tag可以获得根tag,root.attrib获得根attrib，也可获得child的长度len(root)
    2. 然后通过for循环获得child.tag
    """
    root = ET.parse(dir).getroot()   # parse()获得的是解析树，getroot()是获得根结点
    return root

def read_txt():
    pass
    
def read_csv():
    pass


if __name__=='__main__':
    dir = '/home/ubuntu/MyDatasets/voc/VOCdevkit/VOC2007/Annotations/000017.xml'
    root = read_xml(dir)
    for child in root:
        print(child.tag, child.attrib.values())
#        for children in child:
#            print(children.tag, children.attrib)
    
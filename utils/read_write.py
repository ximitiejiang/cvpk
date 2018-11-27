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
    # 首先获得root,里边包含data和item: root.tag, root.attrib
    # for child in root:
    #   print(child.tag, child.attrib)
    
    # for item in root.iter():
    #   print(item)
    root = ET.parse(dir).getroot()
    img = np.array(Image.open(self._imgpath % img_id).convert('RGB'))
    pass

def read_txt():
    pass
    
def read_csv():
    pass
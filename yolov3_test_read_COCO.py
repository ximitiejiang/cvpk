#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:55:42 2018

@author: ubuntu

COCO2017官网登录不上去，但下载是可以通过他的子链接进行：
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip
http://images.cocodataset.org/annotations/image_info_test2017.zip

coco api的安装：git clone ssh://.. 然后进入目录运行：
python setup.py install


"""

# =============================================================================
# 解压缩coco数据集
# =============================================================================
for ann_name in annDir:
    z = zipfile.ZipFile(dataDir + '/' + ann_name)
    # 全部解压
    z.extractall(dataDir)
    
    
# =============================================================================
# 开始使用coco API
# =============================================================================
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

dataDir='/media/ubuntu/4430C54630C53FA2/SuLiang/MyDatasets/coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# 初始化COCO api 针对实例的标注
coco=COCO(annFile)

# 显示coco的类(80类)和超类(12类)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# 显示一张随机照片
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()




# 随机数的实验
import random

random.seed(1)

print(random.randn(2,2))

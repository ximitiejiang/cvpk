#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 08:41:47 2018

@author: ubuntu

基于voc的标注数据，通过kmean算法聚类所有IOU，找到9个聚类点用于先验anchor输入到yolov3
参考：https://github.com/lars76/kmeans-anchor-boxes

"""

import glob
import xml.etree.ElementTree as ET
from unittest import TestCase

import numpy as np
from slcv.utils.anchor_base import generate_anchor_base

#from slcv.utils.kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "/home/ubuntu/MyDatasets/voc/VOCdevkit/VOC2007/Annotations"


def iou(box, clusters):
    """
    得到一个bbox跟n个clusters的IOU [0.3,0.5,...]
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    基于得到cluster聚类n=9或n=5个anchors计算跟数据集的所有bbox的平均IOU
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    # iou()得到一个bbox跟9个clusters的IOU list, 取max就是会被选定的anchor，最终对所有会被选定的anchor求平均
    # np.mean([0.4,0.3,0.5,0.8.....])
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]  # bbox的个数

    distances = np.empty((rows, k))  
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # 随机出9个anchors作为一组clusters

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)  # 用所有bbox跟该组clusters求1-IOU作为判定标准

        nearest_clusters = np.argmin(distances, axis=1)     # 取1-IOU最小的，也就是IOU最大的clusters

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


class TestVoc2007(TestCase):
    def __load_dataset(self):
        dataset = []
        for xml_file in glob.glob("{}/*xml".format(ANNOTATIONS_PATH)):
            tree = ET.parse(xml_file)

            height = int(tree.findtext("./size/height"))
            width = int(tree.findtext("./size/width"))

            for obj in tree.iter("object"):  # 基于图片w,h进行坐标归一化
                xmin = int(obj.findtext("bndbox/xmin")) / width
                ymin = int(obj.findtext("bndbox/ymin")) / height
                xmax = int(obj.findtext("bndbox/xmax")) / width
                ymax = int(obj.findtext("bndbox/ymax")) / height

                dataset.append([xmax - xmin, ymax - ymin])  # 归一化之后的bbox的w,h

        return np.array(dataset)  # 返回[[w1,h1], [w2,h2],...]

    def test_kmeans_5(self):
        dataset = self.__load_dataset()

        out = kmeans(dataset, 5)
        percentage = avg_iou(dataset, out)

        np.testing.assert_almost_equal(percentage, 0.61, decimal=2)

    def test_kmeans_9(self):
        dataset = self.__load_dataset()  # 返回voc2007所有图片的bbox数据

        out = kmeans(dataset, 9)
        percentage = avg_iou(dataset, out)

        np.testing.assert_almost_equal(percentage, 0.672, decimal=2)
        
        return out, percentage
    
    def test_yolov3_mIOU(self):
        dataset = self.__load_dataset()
        # 这是darknet中定义的voc anchors
        anchor1 = [[10,13],[16,30],[33,23]]
        anchor2 = [[30,61],[62,45],[59,119]]
        anchor3 = [[116,90],[156,198],[373,326]]
        ab = anchor1
        ab.extend(anchor2)
        ab.extend(anchor3)
        for i in range(len(ab)):
            ab[i][0] = ab[i][0]/416
            ab[i][1] = ab[i][1]/416
        perc = avg_iou(dataset, ab)
        return perc
    
    def test_faster_rcnn_mIOU(self):
        dataset = self.__load_dataset()
        ab = generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32])
        ab = translate_boxes(ab)
        perc = avg_iou(dataset, ab)
        return perc
    

        
if __name__=='__main__':
    
    testvoc = TestVoc2007()
    
    # -----------1.计算聚类anchors
    # (可认为anchors是一个相对原图的比例anchor，所以在实际416方图上anchors的大小就是乘以416)
    out, perct = testvoc.test_kmeans_9()
    print('kmean anchors: {}'.format(out))
    print('average IOU for kmean anchors: {}'.format(perct))
    out_original = np.array([[out[i][0]*416, out[i][1]*416] for i in range(len(out))]) # [w, h]
    from slcv.utils.anchor_base import _rectangle
    ab=[]
    import matplotlib.pyplot as plt
    f1 = plt.figure()
    for i, anchor in enumerate(out_original):
        ab.append([-anchor[1]/2, -anchor[0]/2, anchor[1]/2, anchor[0]/2])  # 转换为[ymin,xmin,ymax,xmax]
        _rectangle(ab[i], color='r')
    
    # -----------2. 验证darknet上里边给出的9个anchors在voc上的mIOU
    f2 = plt.figure()
    perc = testvoc.test_yolov3_mIOU()
    print('darknet anchors mIOU: {}'.format(perc))
    
    # -----------3. 验证faster rcnn的anchor base在voc数据集上的mIOU
    

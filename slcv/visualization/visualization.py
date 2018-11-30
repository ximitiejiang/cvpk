#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:19:16 2018

@author: suliang
"""

import visdom
import torch
import numpy as np

class Visualization():
    '''这个可视化模块基于visdom进行显示，相关loss和accuracy的计算直接用源码计算
    先要在命令行运行python3 -m visdom.server
    '''
    def __init__(self, env = 'default'):
        self.vis = visdom.Visdom(env=env)
        self.total_loss=0
        self.total_correct=0
        self.total_imgs=0
    
    def batch_update(self, loss, outputs, labels):
        '''用于更新每个batch的loss数据
        '''
        _, predicts = torch.max(outputs, 1) # 返回每行最大概率值tensor，和概率值位置tensor
        
        self.total_loss += loss.item() * labels.size(0)
        self.total_correct += torch.sum(predicts == labels.data).item()
        self.total_imgs += labels.size(0)
        
    def epoch_update(self,epoch_num):
        average_loss = self.total_loss / self.total_imgs
        average_acc = float(self.total_correct) / self.total_imgs
        self.vis.line(X=[epoch_num],
                      Y=[average_loss],
                      opts=dict(markers=False,xlabel='epoch',ylabel='loss'),
                      win='loss',
                      name='loss',
                      update='append')
        
        self.vis.line(X=[epoch_num],
                      Y=[average_acc],
                      opts=dict(markers=False,xlabel='epoch',ylabel='accuracy'),
                      win='accuracy',
                      name='accuracy',
                      update='append')
        '''
        self.vis.heatmap(X=np.outer(np.arange(1, 6), np.arange(1, 11)),
                        opts=dict(columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                                  rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
                        colormap='Electric'))
        '''
        self.total_loss = 0
        self.total_correct = 0
        self.total_imgs = 0
        

if __name__ == '__main__':
    vis = Visualization(env='test1')

 
    
""" 
from torchnet import meter
from torchnet.logger import VisdomPlotLogger, VisdomLogger
class Visulization_tn():
    '''创建可视化类：该可视化模块基于torchnet的meter和logger模块计算，并利用visdom显示
    由于文档太简单，使用的人太少，当前先放弃进一步开发。用Visulization class替代
    
    '''
    def __init__(self,num_classes):
        ''' torchnet是用来更便捷绘图的工具: 可以省去对loss平均值的计算，
        也省去confusion的绘图操作，而只需要把loss, outputs, labels传入meter即可
        参考：
        
        '''
        self.vis_env = visdom.Visdom(env='mypytorch') 
        # 数据提取器
        self.loss_meter = meter.AverageValueMeter()     # 创建平均值量表
        self.confusion_meter = meter.ConfusionMeter(num_classes)  # 创建混淆量表
        # 数据显示器
        self.loss_logger = VisdomPlotLogger('line', win='loss',   # 创建平均值记录仪
                                            opts={'title':'Train Loss'}, 
                                            port=8097, server='localhost')
        self.confusion_logger = VisdomLogger('heatmap', win='conf',  # 创建混淆矩阵记录仪
                                             opts={'title': 'Confusion matrix','columnnames': list(range(num_classes)),'rownames': list(range(num_classes))}, 
                                             port=8097, server='localhost')

    def reset(self):
        '''每个epoch开始之前，都重置2个meter，确保每次计算的loss，accuracy(confusion)
        都是针对一个epoch
        '''
        self.loss_meter.reset()       # 每个epoch重置average loss
        self.confusion_meter.reset() # 每个epoch重置confusion matrix

    
    def add_batch_data(self,loss, outputs, labels):
        '''在每个batch的最后
        输入：该batch的loss，outputs，labels
        
        * loss_meter.add(loss.item()) 输入的loss会通过AverageValueMeter计算该batch的平均loss
        * confusion_meter.add(outputs, labels) 输入的outputs和labels会通过ConfusionMeter统计每个分类的正确数量
        '''
        self.loss_meter.add(loss.item())  # loss_meter是对每个batch的平均loss累加
        self.confusion_meter.add(outputs.detach(), labels.detach()) # accurary的另一种表达为混淆矩阵
    
    def log_epoch_data(self,epoch):
        self.loss_logger.log(epoch, self.loss_meter.value()[0])
        self.confusion_logger.log(self.confusion_meter.value())
        
        
#        accuracy = 100. * (self.confusion_meter.value()[0][0] + self.confusion_meter.value()[1][1]) / (self.confusion_meter.value().sum())
#        self.vis_env.line(X=np.array(epoch).reshape(1), Y=np.array(accuracy).reshape(1), win='acc',opts={'title':'Train accuracy'}, update='append')
    
    def vis_show_image():
        '''采用visdom显示图片
        输入：可以是图片形式HxW, 也可以向量形式CxHxW
        '''
        pass
    def vis_show_loss():
        '''采用visdom显示损失函数曲线
        '''
        pass
    
    def vis_show_confusion():
        '''采用visdom显示精度matrix
        '''
        pass
    def vis_show_accuracy():
        '''采用visdom显示精度曲线
        '''
        pass
"""    
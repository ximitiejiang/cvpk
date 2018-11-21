#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:00:18 2018

@author: ubuntu
"""
import torch
from torch import nn, optim
import os
import time
import numpy as np

def define_criterion(name):
    '''定义loss损失函数
    输入：损失函数简称（均用小写）
    输出：损失函数模型
    '''
    if name =='mse':
        cr = nn.MSELoss()
    elif name == 'cross_entropy':
        cr = nn.CrossEntropyLoss()  # 交叉熵损失 = logsoftmax + nll
    elif name == 'smooth_l1':
        cr = nn.SmoothL1Loss()  # 用于faster-rcnn的RPN中做回归损失函数
    elif name =='nll':
        cr = nn.NLLLoss()  # negtive log likelihood loss负对数似然损失
    else:
        raise ValueError('not recognized criterion!')
    return cr


def define_optimizer(name, params):
    '''定义参数优化器
    '''
    if name == 'adam':
        op = optim.Adam(params,
                        lr = 0.001,
                        weight_decay = 0)
    elif name == 'sgd':
        op = optim.SGD(params, 
                       lr = 0.001, 
                       momentum=0.9)
    else:
        raise ValueError('not recognized criterion!')
    return op


def write_txt(results, file_name='test', type='a+'):
    '''用于写入一个txt文件，默认路径在checkpoints文件夹
    写入模式：a+ 自由读写，扩展模式，文件没有就创建
    写入模式：w+ 自由读写，覆盖模式
    '''
    directory = '/Users/suliang/slcv/checkpoints/'
    file_path = os.path.join(directory, file_name)
        
    with open(file_path, type) as f:  # 以结尾写入的方式打开，只有'a'和'at'两种模式的指针是在文件末尾
        print(results, file = f)


class Visdom_kit():
    def create():
        pass
    
    def add():
        pass

    def log():
        pass
    
    
from torchnet import meter
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import visdom

class Visulization():
    '''创建可视化类：  先要在命令行运行python -m visdom.server
    '''
    def __init__(self,num_classes):
        ''' torchnet是用来更便捷绘图的工具
        
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
        
        # 增加一个accuracy曲线显示
#        accuracy = 100. * (confusion_meter.value()[0][0] + confusion_meter.value()[1][1]) / (confusion_meter.value().sum())
#        self.vis_env.line(X=np.array(epoch).reshape(1), Y=np.array(accuracy).reshape(1), win='cur',opts={'title':'Train accuracy'}, update='append')
    
    def show_image():
        '''用于显示图片
        输入：可以是图片形式HxW, 也可以向量形式CxHxW
        '''
        pass
    
    
class Trainer(nn.Module):
    ''' 用于进行网络训练的类：    
    '''
    
    def __init__(self, model,num_classes):
        super().__init__()
        self.model = model
        self.criterion = define_criterion('cross_entropy')
        self.optimizer = define_optimizer('sgd', self.model.parameters())
#        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#        if torch.cuda.device_count() > 1:
#            self.model = nn.DataParallel(self.model)
        
        self.vis = Visulization(num_classes)
        self.vis.reset()
        
#        if self.model.cuda:
#            self.model.cuda()
    
    def _grads_output(self,j):
        '''用于在5步核心step中提取grad，并写入文件中
        '''
        # grad通过backward()步已更新, 显示前10个batch的梯度
#        if j<10
        if j%1000==0:
            out=[]
            for k,(name, p) in enumerate(self.model.named_parameters()):
                out.append((name, p.grad.data.std()))

            write_txt(out, file_name='grad',type='a+')
            
    
    def _lr_output(self, j):
        '''用于在循环步中提取learning rate，并写入文件
        '''
        pass

            
    
    def step(self, j, imgs, labels):  # 5步核心循环   
        '''每个batch_size循环运行
        '''
#        if self.model.cuda:
#            imgs = imgs.cuda()
#            labels = labels.cuda()
        
        self.optimizer.zero_grad()               # (1)
        outputs = self.model(imgs)               # (2)
        loss = self.criterion(outputs, labels)   # (3)  
        loss.backward()                          # (4)
        
        self._grads_output(j)                  
        
        self.optimizer.step()                    # (5)
                        
        self.vis.add_batch_data(loss,outputs,labels)
        
    
    def epoch_show(self, epoch):
        '''每个epoch循环的最后运行
        '''
        self.vis.log_epoch_data(epoch)
        self.vis.reset()
    
    
    def validate(self):
        '''在training结束后进行validation
        可输出validation的相关loss，accuracy，confusion等
        
        '''
        self.model.eval()
        
        
        self.model.train()
        

if __name__ == '__main__':
    test_id =0
    
    if test_id == 0:
        vis = Visulization(num_classes=2)
        
        loss = 1
        outputs = 1
        labels = 1
        vis.add_batch_data(loss, outputs, labels)
        
        vis.log_epoch_data(0)
    
    elif test_id == 1:
        # 以下验证已通过
        from torchvision import models
        model = models.alexnet(pretrained=True)
        trainer = Trainer(model,num_classes=10)
    
        imgs = torch.randn(1,3,32,32)
        labels = torch.tensor([1])
    
        trainer.step(imgs, labels)



        
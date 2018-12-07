#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:00:18 2018

@author: ubuntu

debuge items: lr, batch_size, opt_type, bn, 

"""
# TODO: gpu training
# TODO: accuracy plot

import torch
from torch import nn, optim
import os
import time
from slcv.utils.visualization import Visualization

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
    import sys
    if sys.platform == 'linux':
        directory = '/home/ubuntu/suliang_git/slcv/checkpoints'   # for ubuntu 
    else:                     
        directory = '/Users/suliang/slcv/checkpoints/'            # for mac os
    
    file_path = os.path.join(directory, file_name)
        
    with open(file_path, type) as f:  # 以结尾写入的方式打开，只有'a'和'at'两种模式的指针是在文件末尾
        print(results, file = f)



    
class Trainer(nn.Module):
    ''' 用于进行网络训练的类：    
    '''
    
    def __init__(self, model,num_classes):
        super().__init__()
        self.model = model
        self.criterion = define_criterion('cross_entropy')
        self.optimizer = define_optimizer('adam', self.model.parameters())
        
#        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#        if torch.cuda.device_count() > 1:
#            self.model = nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        self.vis = Visualization(env='mynewvis')
    
    
    def _grads_output(self,j):
        '''用于在5步核心step中提取grad，并写入文件中
        '''
        # grad通过backward()步已更新, 显示前10个batch的梯度
#        if j<10
        if j%500==0 and j!=0:
            out=[]
            for k,(name, p) in enumerate(self.model.named_parameters()):
                out.append((name, p.grad.data.std()))

            write_txt(out, file_name='grad',type='a+')
            
    def _loss_output(self,j,loss):
        '''用于在循环步中导出loss
        '''
        if j%500 ==0 and j!=0:
            write_txt((j,loss.item()), file_name='loss',type='a+')
            
    def _parameter_output(self):
        """用于输出模型参数的均值和方差
        """
        out = []
        for k,(name, p) in enumerate(self.model.named_parameters()):
            out.append((name, p.data.mean(),p.data.std()))
        write_txt(out, file_name='parameter',type='a+')       
     
        
    def _lr_output(self, j):
        '''用于在循环步中提取learning rate，并写入文件
        '''
        pass

   
    def step(self, j, imgs, labels):  # 5步核心循环   
        '''每个batch_size循环运行
        输入：为内层for循环的3个循环变量
            - j 
            - imgs 
            - labels
        '''
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        self.optimizer.zero_grad()               # (1)
        outputs = self.model(imgs)               # (2)  outputs: n_batch_size x n_classes
                                                 #      labels: 1 x n_batch_size
        loss = self.criterion(outputs, labels)   # (3)  1 x n_classes
        
        self._loss_output(j,loss)
        
        loss.backward()                          # (4)
        
        self._grads_output(j)                  
        
        self.optimizer.step()                    # (5)
                        
#        self.vis.add_batch_data(loss,outputs,labels)
        self.vis.batch_update(loss, outputs, labels)
        
    
    def epoch_show(self, epoch_num):
        '''每个epoch循环的最后运行
        输入：为外层for循环的1个循环变量
            - epoch 为
        输出：
            - 数据送入visdom
            - 量表清零
        '''
        self.vis.epoch_update(epoch_num)
    
    def model_save(self, name='model'):
        '''用于模型的保存，参考陈云的书P170
        '''
        if name is 'model':
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.path')
            torch.save(self.state_dict(),name)
        return name
    
    
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



        
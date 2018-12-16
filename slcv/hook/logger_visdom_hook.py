#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:13:15 2018

@author: ubuntu
"""
import visdom
from .hook import Hook

class LoggerVisdomHook(Hook):
    
    def __init__(self, log_config):
        """输入为log_config dict: {}"""
        self.vis = visdom.Visdom(env='slcv')
        self.interval = log_config['interval']   # log per n iters
        self.total_loss=0
        self.total_correct=0
    
#    def log_lr(self, runner):
#        self.vis.line(X=[num_x],
#              Y=
#              opts=dict(markers=False,xlabel=xlabel,ylabel='learning rate'),
#              win='lr',
#              name='learning rate',
#              update='append')
    
    def log(self,runner):
        """数据显示到终端"""
        if self.interval > 0:
            num_x = runner._iter
            xlabel = 'iter'
        else:
            num_x = runner._epoch
            xlabel = 'epoch'
        
        self.vis.line(X=[num_x],
              Y=[runner.log_buffer.average_output['loss']],
              opts=dict(markers=False,xlabel=xlabel,ylabel='loss'),
              win='loss',
              name='loss',
              update='append')
        
        self.vis.line(X=[num_x],
                      Y=[runner.log_buffer.average_output['acc_top5']],
                      opts=dict(markers=False,xlabel=xlabel,ylabel='acc'),
                      win='accuracy',
                      name='acc_top5',
                      update='append')
        
        self.vis.line(X=[num_x],
                      Y=[runner.log_buffer.average_output['acc_top1']],
                      opts=dict(markers=False,xlabel=xlabel,ylabel='acc'),
                      win='accuracy',
                      name='acc_top1',
                      update='append')
        

        ''' TODO: heatmap
        self.vis.heatmap(X=np.outer(np.arange(1, 6), np.arange(1, 11)),
                        opts=dict(columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                                  rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
                        colormap='Electric'))
        '''
    
    def before_train_epoch(self, runner):
        """核心的3组hooks: before_run"""
        runner.log_buffer.clear()  # clear logs of last epoch
    
    def after_train_iter(self, runner):
        """核心的3组hooks: before_run"""
        if (self.interval > 0) and (runner._iter%self.interval == 0): # interval > 0则执行logger per n iter
            runner.log_buffer.average(self.interval)  # 该句之前在主代码中需要事先log_buffer.update(xx) 
            if runner.log_buffer.ready: # 计算完成
                self.log(runner)
        
    def after_train_epoch(self, runner):
        """核心的3组hooks: before_run"""
        if not self.interval > 0:  # interval = 0 则执行logger per epoch
            runner.log_buffer.average(self.interval)
            if runner.log_buffer.ready:
                self.log(runner)
    
    
    def test_hook(self, runner):
        print('this is visdom_logger_hook test_hook')  # only for test


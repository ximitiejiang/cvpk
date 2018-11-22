#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:44:47 2018

@author: suliang
"""

import visdom
import matplotlib.pyplot as plt
import numpy as np

class Visdomkit():
    def __init__(self):
        self.vis = visdom.Visdom(env='123456') 
        
        
    
    def add(self):
        pass

    def log(self):
        pass
    
    def show_points(self):
        self.vis.scatter(X=np.random.rand(255, 2),
                         Y=(np.random.rand(255) + 1.5).astype(int),
                         opts=dict(markersize=10,markercolor=np.random.randint(0, 255, (2, 3,)))
                         )
        
#    def show_line(self):  # ok, 不过尺寸优点大，且不能缩放，类似于把matplot的图片拷贝过去的
#        x = [1,2,3,4,5]
#        y = [2,5,1,7,9]
#        plt.plot(x,y)
#        plt.xlabel('items')
#        plt.ylabel('ages')
#        self.vis.matplot(plt)
#        
    def show_lines(self): 
        Y = np.linspace(-5, 5, 100)
        self.vis.line(Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
                      X=np.column_stack((Y, Y)),
                      opts=dict(markers=False))
    def show_line(self):

        self.vis.line(X=[1,2,3,4,5],
                      Y=[10,20,30,40,50],win = 'line',
                      opts=dict(markers=False,xlabel='x_item',ylabel='y_age'),
                      name = '1')
        
    def update_lines(self):  # ok，没相当visdom的line命令有点反直觉：一定要大写X/Y，一定要指定win/name/updae三个参数
        self.vis.line(X=[6,7,8,9,10],
                      Y=[50,40,30,20,10],
                      win='line',         # 决定了是否绘制在同一窗口
                      name = '2',         # 决定了是否连在一起
                      update='append')    # 决定了是覆盖还是扩展
    
    def show_img(self): # ok
        self.vis.image(np.random.rand(3, 512, 256),
                       opts=dict(title='Random!', caption='How random.'))
        
    def show_imgs(self): # ok
        self.vis.images(np.random.randn(20, 3, 64, 64),
                        opts=dict(title='Random images', caption='How random.')
                        )
        
    def show_heatmap(self): # ok
        '''注意x作为行，对应rownames，但显示在heatmap的竖轴上。
        np.outer(x1,x2),其中x1为每行的倍数，x2为每行的基数，每行=x2*x1[i]
        '''
        self.vis.heatmap(X=np.outer(np.arange(1, 6), np.arange(1, 11)),
                        opts=dict(columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                                  rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
                        colormap='Electric')
                        )
    
vis = Visdomkit()
vis.show_line()
vis.update_lines()

#vis.show_imgs()
#vis.show_points()
#vis.show_heatmap()

#vis.show_line()
#vis.update_lines()

'''
vis2 = visdom.Visdom(env='hello') 



win = vis2.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
)
    
vis2.line(
    X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
    win=win,
    update='append'
)
    
vis2.line(
X=np.arange(21, 30),
Y=np.arange(1, 10),
win=win,
name='2',
update='append')

vis2.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,              # 决定是否在同一窗口绘制
    name='delete this',   # 如果名称不同，也不会连在一起
    update='append'       # 如果append，默认是用同一颜色，连在一起
    )

vis2.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='4',
    update='insert'
    )

vis2.line(X=None, Y=None, win=win, name='delete this', update='remove')

vis2.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,              # 决定是否在同一窗口绘制
    name='2',   # 如果名称不同，也不会连在一起
    update='append'       # 如果append，默认是用同一颜色，连在一起
    )

vis2.line(X=[1,2,3,4,5],
          Y=[10,20,30,40,50],
          win='99')
'''
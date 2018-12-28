#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:58:50 2018

@author: ubuntu

分布式训练教程：参考https://pytorch.org/tutorials/intermediate/dist_tuto.html

1. 原有的DataParalle属于multiprocessin多进程，但这个多进程被限制在单机多卡，无法多机进行。
而DistributedDataParalle函数，

2. 有三种可选的backend后端：gloo, mpi, tcp。其中gloo支持GPU


"""

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run_1(rank, size):
    """这个run的模式是block模式，因为send/recv都是同步通信模式
    所以在通信完成之前是blocking模式，在循环过程中，rank0发送不会立即打印，因为
    rank1还没收到，所以rank0 block等待，一直等到rank1收到了，rank1打印输出了，
    rank0才会继续执行打印
    """
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)  # 如果是进程0，则发送tensor
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)  # 如果是进程1,则接收tensor
    print('Rank ', rank, ' has data ', tensor[0])

def run_2(rank, size):
    """这个run的模式是no-blocking模式，因为isend/irecv都是异步通信模式
    rank0发送，等到发送完成就可以打印rank0状态了，而不需要等到rank1接收该信息
    然后rank1接收该信息，等到接收完成就可以打印rank1状态了
    """
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)  # 如果是进程0，则发送tensor
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)  # 如果是进程1,则接收tensor
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

def run_3(rank, size):
    """"""
    group = dist.new_group([0,1])  # 创建一个group,包含2个进程
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
    

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'    # 初始化public ip
    os.environ['MASTER_PORT'] = '29500'        # 初始化端口
    # 初始化后端
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # 创建进程：每个进程都进行初始化，定义rank, 程序
        p = Process(target=init_processes, args=(rank, size, run_3))
        p.start()  # 进程开始
        processes.append(p)  

    for p in processes:
        p.join()
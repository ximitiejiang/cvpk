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

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
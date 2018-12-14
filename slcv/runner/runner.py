#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:36:46 2018

@author: ubuntu
"""
import torch
import sys, os, time
from collections import OrderedDict
from slcv.hook.hook import Hook
from slcv.hook.optimizer_hook import OptimizerHook
from slcv.hook.visdom_logger_hook import VisdomLoggerHook
from slcv.hook.timer_hook import TimerHook
from slcv.hook.text_hook import TextHook
from ..hook.log_buffer import LogBuffer

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad(): # with内的tensor均不更新梯度requires_grad =False
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def obj_from_dict(info, parrent=None, default_args=None):
    """基于字典初始化优化器对象.
    Args:
        info (dict): dict类型的对象参数,必须包含一个type字段，该type可以等于如下
        的优化器名字('Adam', 'SGD', 'RMSprop'...等价于pytorch优化器描述)
        module (:class:`module`): 目标object类型
        default_args (dict, optional): 初始化object的默认参数
    Returns:
        目标对象
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type') # 获得优化器的名称
    if isinstance(obj_type, str):
        if parrent is not None:
            obj_type = getattr(parrent, obj_type)  # 获得优化器的对应子类
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


class Runner():
    """定义一个runner主类，中间包含train,val,register_hook,call_hook等核心操作函数
    保持柔性可扩展：hooks可以增加，train/val可以合并成run()函数
    输入：dataloader, 数据加载器
        model，模型对象
        criterion，损失函数
        optimizer，优化器，可以是dict优化器参数也可以是torch.optim.Optimizer对象
        cfg，配置对象
    """
    def __init__(self, dataloader, model, optimizer, cfg):
        self.dataloader = dataloader
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.cfg = cfg
        self._hooks = []
        self.log_buffer = LogBuffer()
        self._iter = 0
        self._epoch = 0
    
    def model_name(self):
        return self.model.__class__.__name__ 
    
    
    def init_optimizer(self, optimizer):
        """可以传入一个optimizer对象，也可以传入一个dict参数字典
        输入：optimizer 为module对象或者dict对象
        输出：optimizer 为module对象
        """
        if isinstance(optimizer, dict):
            # 传入优化器参数，优化器
            optimizer = obj_from_dict(optimizer, torch.optim, dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                    'optimizer must be either an Optimizer object or a dict, '
                    'but got {}'.format(type(optimizer)))
        return optimizer
    
    
    @property
    def hooks(self):
        return self._hooks
    
    def register(self, sub_hook_class, args=None):
        """基于hook config创建hook对象，并加载入_hook变量
        输入：args, 为hook创建参数
            sub_hook_class为hook的子类
        可以是hook对象，此时sub_hook_class就无需输入
        如果是dict，就需要再输入sub_hook_class
        """
        if isinstance(args, Hook):
            hook = args                  # 创建现成hook
        elif isinstance(args, dict):
            hook = sub_hook_class(args)  # 创建带参hook对象
        elif args is None:
            hook = sub_hook_class()      # 创建不带参hook
        else:
            raise TypeError('args should be hook obj or dict type')
        self._hooks.insert(0, hook)      # 加入_hooks数组,最后一个hook放最前面，也最先执行
    
    def register_hooks(self, 
                       optimizer_config,
                       log_config=None,
                       text_config=None):
        """注册hooks, 默认hooks包括
        OptimizerHook(带配置文件), 
        TimerHook
        可选：
        checkpoint_hook, 
        iter_time_hook(带配置文件), 
        logger_hook(带配置文件)
        """
        self.register(OptimizerHook, optimizer_config)
        self.register(TimerHook)
        
        if log_config is not None:
            self.register(VisdomLoggerHook, log_config)
        if text_config is not None:
            self.register(TextHook, text_config)
        
    
    def call_hook(self, fn_name):
        """批量调用Hook类中所有hook实例的对应方法"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
            
    def load_state_dict(self, module, state_dict, strict=False):
        """Load state_dict to a module.
        """
        unexpected_keys = [] # 存放额外多出来的key
        
        own_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
    
            try:
                own_state[name].copy_(param)  # 拷贝checkpoint的参数到model中
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(),
                                           param.size()))
        # 存放少了的key
        missing_keys = set(own_state.keys()) - set(state_dict.keys())
    
        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}\n'.format(
                ', '.join(unexpected_keys)))
        if missing_keys:
            err_msg.append('missing keys in source state_dict: {}\n'.format(
                ', '.join(missing_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            else:
                print(err_msg)
            
    def load_checkpoint(self, filename, map_location=None, strict=False):
        """加载checkpoint，把state_dict传递给model，并返回checkpoint字典(可用于提取checkpoint中其他信息)
        输入：filename, 
        map_location: 可以选择''
        strict(是否允许不同参数)
        返回dict
        torch.load()参考：https://pytorch.org/docs/stable/torch.html
        to same cpu or GPU: torch.load('gen_500000.pkl')
        to->cpu: torch.load('gen.pkl', map_location=lambda storage, loc: storage)
        cpu->GPU(1): torch.load('gen.pkl', map_location=lambda storage, loc: storage.cuda(1))
        GPU0->GPU1: torch.load('gen.pkl', map_location={'cuda:0':'cuda:1'})
        """
        if not os.path.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        # 加载checkpoint
        print('loading checkpoint file: {}'.format(filename))
        checkpoint = torch.load(filename, map_location = map_location)
        # ----------------从checkpoint获得state_dict-------------------
        if isinstance(checkpoint, OrderedDict): # 如果直接存的是OrderedDict则直接读取
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # 如果存的是dict则读取state_dict
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(filename))
        # 如果是data paralle model，则去掉module关键字(即从第7个字符开始)后得到state_dict
        # 参考：https://www.ptorch.com/news/74.html
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            
        # --------------把state_dict加载到模型中-------------------------
        if hasattr(self.model, 'module'):  # 如果是data paralle model，则需要提取module再加载state_dict
            self.load_state_dict(self.model.module, state_dict)
        else:  # 如果是普通模型，则直接用model加载state_dict
            self.load_state_dict(self.model, state_dict)           
        return checkpoint
    
    def save_checkpoint(self, out_dir, filename, save_optimizer=True, meta=None):
        """保存checkpoint到文件
        输入：meta 保存version和time, dict, 默认是{'epoch':epoch, 'iter':iter}
              out_dir保存地址
              filename保存文件名
              save_optimizer是否保存优化器
        输出：OrderedDict {'meta':dict, 'state_dict':OrderedDict, 'optimizer':dict}
        """
        if meta is None:
            meta = dict(epoch=self._epoch +1, iter = self._iter)
        else:
            meta.update(epoch=self._epoch +1, 
                        iter = self._iter,
                        time = time.time())
        # 判断是否保存optimizer
        appendname = '_epoch_{}.pth'
        filepath = os.path.join(out_dir, filename + appendname.format(self._epoch+1))
        optimizer = self.optimizer if save_optimizer else None
        # 从GPU拷贝state_dict到cpu
        state_dict_cpu = OrderedDict()
        for key, val in self.model.state_dict().items():
            state_dict_cpu[key] = val.cpu()
#            state_dict_cpu[key] = val
        # 生成checkpoint    
        checkpoint ={'meta': meta,
                     'state_dict': state_dict_cpu}
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        # 保存checkpoint
        torch.save(checkpoint, filepath)
        
    
    def train(self):
#        # 定义设备
#        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cuda模式或cpu模式
#        if len(self.cfg.gpus) == 0:    # cpu模式
#            device = torch.device("cpu")
#        self.model.to(device)
        
        self.model.train()        
        self.call_hook('before_run')
#        for i in range(self.cfg.epoch_num):
        while self._epoch < self.cfg.epoch_num:  # 不用for循环是为了resume时兼容
            self.call_hook('before_train_epoch')
            
            for j, (imgs, labels) in enumerate(self.dataloader):
                self.call_hook('before_train_iter')

                imgs = imgs.float().to(device)  # 这里to(device)需要保证imgs为torch.float32类型
                labels = labels.to(device)      # label为torch.int64
                 
                pred = self.model(imgs)
                # 复杂loss则通过loss function导入计算            
                loss = torch.nn.CrossEntropyLoss()(pred,labels)
                 
                # outputs作为汇总变量，传入hooks
                acc_top1,acc_top5 = accuracy(pred,labels, topk=(1,5))
                log_vars = OrderedDict()
                log_vars['loss'] = loss.item()
                log_vars['acc_top1'] = acc_top1.item()
                log_vars['acc_top5'] = acc_top5.item()
                
                # 更新2个主参数容器： 
                self.outputs = dict(loss=loss, log_vars=log_vars, num_samples=imgs.size(0))
                self.log_buffer.update(self.outputs['log_vars'])
                self.call_hook('after_train_iter')
                 
                self._iter += 1
            
            # test save_checkpoint()
            self.save_checkpoint(self.cfg.checkpoints_dir, 'yolov3', save_optimizer=True, meta=None)
            self.call_hook('after_train_epoch')
            self._epoch += 1
        self.call_hook('after_run')

    
    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        """恢复某个checkpoint：state_dict给model
        输入：map_location，默认加载到GPU(device 0)，也可加载到cpu
        在resume时需要确保cfg中指定的cpu/gpu方式跟resume()中map_location定义是一致的
        """
        if map_location == 'default':  # 默认是第0个GPU
            device_id = torch.cuda.current_device()  # 默认加载到第0个GPU
            checkpoint = self.load_checkpoint(checkpoint, 
                                              map_location = lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint, 
                                              map_location=map_location)
        
        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
                        
    def val(self):
        """验证模块：待调试
        """
        self.model.eval()
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
    
    def test(self):
        """测试模块
        """
        pass
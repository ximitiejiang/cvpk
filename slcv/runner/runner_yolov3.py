#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:36:46 2018

@author: ubuntu
"""
import torch
import sys, os, time
from collections import OrderedDict
from collections import defaultdict
from slcv.hook.hook import Hook
from slcv.hook.optimizer_hook import OptimizerHook
from slcv.hook.visdom_logger_hook import VisdomLoggerHook
from slcv.hook.timer_hook import TimerHook
from slcv.hook.text_hook import TextHook
from ..hook.log_buffer import LogBuffer
from slcv.utils.checkpoint import load_checkpoint


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
    
    def save_checkpoint(self):
        pass
    
    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        return load_checkpoint(self.model, filename, map_location, strict)
    
    def resume(self, checkpoint, resume_optimizer=True, map_location = 'default'):
        pass
        

    
    def train(self):
        torch.cuda.empty_cache()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检查是cpu还是gpu训练
        if torch.cuda.device_count() > 1 and len(self.cfg.gpus) > 1:  # 自定义是否多GPU并行训练
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)
#        if torch.cuda.is_available():
#            self.model.cuda()
            
        self.model.train()
        
        # 针对yolov3
        freeze_backbone = True
        t0,t1 = time.time(), time.time()
        mean_recall, mean_precision = 0, 0
        best_loss = float('inf')
        
        self.call_hook('before_run')
        for i in range(self.cfg.epoch_num):
            self.call_hook('before_train_epoch')
            
            # 针对yolo的调整: 可以考虑放进lr_hook中在‘before_train_epoch’
            rloss = defaultdict(float)  # runningloss
            metrics = torch.zeros(3, self.cfg.num_classes)  # (3,20)
            ui = -1
            if self._epoch > 50:
                lr = self.cfg.lr0/10
            else:
                lr = self.cfg.lr0
            for g in self.optimizer.param_groups:
                g['lr'] = lr
            if freeze_backbone is not False:
                if self._epoch == 0:
                    for i, (name, p) in enumerate(self.model.named_parameters()):
                        if int(name.split('.')[1]) <75:
                            p.requires_grad = False
                elif self._epoch == 1:
                    for i, (name, p) in enumerate(self.model.named_parameters()):
                        if int(name.split('.')[1]) <75:
                            p.requires_grad = True
            
            for j, (imgs, labels) in enumerate(self.dataloader):
                self.call_hook('before_train_iter')
                 
                 # 针对yolo的lr调整
                if (self._epoch == 0) & (i <=1000):
                    lr = self.cfg.lr0 * (j/1000)**4
                    for g in self.optimizer.param_groups:
                        g['lr'] = lr
                 
                if torch.cuda.is_available():
                    imgs = imgs.float().to(device)
                    labels = labels.float().to(device)
                 
#                 pred = self.model(imgs)
#                 loss = torch.nn.CrossEntropyLoss()(pred,labels)
                 
                # 基于yolov3模型修改runner主结构: 此处loss为yolo layer的output后求sum()
                loss = self.model(imgs, labels, batch_report= self.cfg.report, var=self.cfg.var)
                ui += 1
                for key, val in self.model.losses.items():
                    rloss[key] = (rloss[key] * ui + val) / (ui +1)
                if self.cfg.report:
                    TP, FP, FN = metrics
                    metrics += self.model.losses['metrics']
                     
                    precision = TP/(TP + FP)
                    k = (TP + FP) > 0
                    if k.sum() > 0:
                        mean_precision = precision[k].mean()
                     
                    recall = TP / (TP + FN)
                    k = (TP + FN) > 0
                    if k.sum() > 0:
                        mean_recall = recall[k].mean()
                         
                s = ('%8s%12s' + '%10.3g' * 14) % (
                    '%g/%g' % (self._epoch, self._epochs - 1), '%g/%g' % (j, len(self.dataloader) - 1), rloss['x'],
                    rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                    rloss['loss'], mean_precision, mean_recall, self.model.losses['nT'], self.model.losses['TP'],
                    self.model.losses['FP'], self.model.losses['FN'], time.time() - t1)
                t1 = time.time()
                print(s)                     
                 
#                 print("[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]" 
#                       % (self._epoch,
#                          self.cfg.epoch_num,
#                          self._iter,
#                          len(self.dataloader),
#                          self.model.losses["x"],
#                          self.model.losses["y"],
#                          self.model.losses["w"],
#                          self.model.losses["h"],
#                          self.model.losses["conf"],
#                          self.model.losses["cls"],
#                          loss.item(),
#                          self.model.losses["recall"],
#                          self.model.losses["precision"],
#                          )
#                       )
                 
                 
#                 acc_top1,acc_top5 = accuracy(pred,labels, topk=(1,5))
#                 log_vars = OrderedDict()
#                 log_vars['loss'] = loss.item()
#                 log_vars['acc_top1'] = acc_top1.item()
#                 log_vars['acc_top5'] = acc_top5.item()
                 
#                 self.outputs = dict(loss=loss, log_vars=log_vars, num_samples=imgs.size(0))
                self.outputs = dict(loss=loss) # yolov3修改，目的是loss传进optimizer_hook做backward计算(也避免修改其他文件)
                 
#                 self.log_buffer.update(self.outputs['log_vars'])
                self.call_hook('after_train_iter')
                 
                self._iter += 1
            #针对yolo
            loss_per_target = rloss['loss'] / rloss['nT']
            if loss_per_target < best_loss:
                best_loss = loss_per_target
            checkpoint ={'epoch': self._epoch,
                         'best_loss': best_loss,
                         'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict()}
            torch.save(checkpoint, latest_weights_file)
            
            # Save best checkpoint
            if best_loss == loss_per_target:
                os.system('cp {} {}'.format(
                    latest_weights_file,
                    best_weights_file,
                ))
    
            # Save backup weights every 5 epochs
            if (epoch > 0) & (epoch % 5 == 0):
                backup_file_name = 'backup{}.pt'.format(epoch)
                backup_file_path = os.path.join(weights_path, backup_file_name)
                os.system('cp {} {}'.format(
                    latest_weights_file,
                    backup_file_path,
                ))
    
            # Calculate mAP
            mAP, R, P = test.test(
                net_config_path,
                data_config_path,
                latest_weights_file,
                batch_size=batch_size,
                img_size=img_size,
            )
    
            # Write epoch results
            with open('results.txt', 'a') as file:
                file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')
            
            
            self.call_hook('after_train_epoch')
            self._epoch += 1
        self.call_hook('after_run')
        
        dt = time.time() - t0
        print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))

        
    def val(self):
        pass
    
    def test(self):
        pass
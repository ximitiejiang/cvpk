from __future__ import division

from collections import OrderedDict
from mmdet import __version__
import torch
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from mmdet.apis import get_root_logger
from mmdet.models import build_detector
from mmdet.datasets import get_dataset
from mmcv import Config


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
        else:
            if cfg.data.val.type == 'CocoDataset':
                runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))
            else:
                runner.register_hook(DistEvalmAPHook(cfg.data.val))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    
    # test一下输出????---------
    
    
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
#    model = model.cuda()  # 看看非并行计算的GPU模型是不是可以用: 不行，好像model不接收datacontainer
    
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)



def main():
#    args = parse_args()
    cfg = Config.fromfile('./cfg_rpn_r50_fpn_1x.py') # 自定义基于voc的cfg
    # update configs according to CLI args
#    if args.work_dir is not None:
#        cfg.work_dir = args.work_dir
#    if args.resume_from is not None:
#        cfg.resume_from = args.resume_from
#    cfg.gpus = args.gpus
    
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
#    if args.launcher == 'none':
#        distributed = False
#    else:
#        distributed = True
#        init_dist(args.launcher, **cfg.dist_params)
    distributed =False

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
#    if args.seed is not None:
#        logger.info('Set random seed to {}'.format(args.seed))
#        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = get_dataset(cfg.data.train)
#    train_detector(
#        model,
#        train_dataset,
#        cfg,
#        distributed=distributed,
#        validate=args.validate,
#        logger=logger)
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,   # 分布式训练或者不是
        validate=False,         # train模式或者validate模式
        logger=logger)


if __name__ == '__main__':
    main()

"""针对个人训练已做修改

"""
# model settings
model = dict(
    type='RPN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),  # FPN选择了5个输出，其中3个是叠加输出，1个是非叠加输出，1个是把
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,  # 可计算得到期望的pos = num *pos_fraction
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,   # 0表示不允许bbox超出图片边界。用来在计算anchor_target时评估xmin,ymin >= - ab, xmax,ymax<ab+w,h
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0))
# dataset settings
#dataset_type = 'CocoDataset'
#data_root = 'data/coco/'
dataset_type = 'VOCDataset'
data_root = '../data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,     # 是不是要改
    workers_per_gpu=2,  # 是不是要改

#    train=dict(
#        type=dataset_type,
#        ann_file=data_root + 'annotations/instances_train2017.json',
#        img_prefix=data_root + 'train2017/',
#        img_scale=(1333, 800),
#        img_norm_cfg=img_norm_cfg,
#        size_divisor=32,
#        flip_ratio=0.5,
#        with_mask=False,
#        with_crowd=False,
#        with_label=False),
#    val=dict(
#        type=dataset_type,
#        ann_file=data_root + 'annotations/instances_val2017.json',
#        img_prefix=data_root + 'val2017/',
#        img_scale=(1333, 800),
#        img_norm_cfg=img_norm_cfg,
#        size_divisor=32,
#        flip_ratio=0,
#        with_mask=False,
#        with_crowd=False,
#        with_label=False),
#    test=dict(
#        type=dataset_type,
#        ann_file=data_root + 'annotations/instances_val2017.json',
#        img_prefix=data_root + 'val2017/',
#        img_scale=(1333, 800),
#        img_norm_cfg=img_norm_cfg,
#        size_divisor=32,
#        flip_ratio=0,
#        with_mask=False,
#        with_label=False,
#        test_mode=True))
    
    train=dict(
        type='RepeatDataset',  # to avoid reloading datasets frequently
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            img_scale=(1000, 600),   # 长边小于1000, 短边小于600
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,         # 增加pad,使图片尺寸w,h为32的倍数
            flip_ratio=0.5,          # 水平翻转几率0.5
            with_mask=False,
            with_crowd=False,       # 这里不是完全从faster rcnn的voc版本来，改成False了，因为RPN源码是False
            with_label=False)),     # 这里不是完全从faster rcnn的voc版本来，改成False了，因为RPN源码是False
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(1000, 600),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,       
        with_crowd=False,           # 这里不是完全从faster rcnn的voc版本来，改成False了，因为RPN源码是False
        with_label=False),          # 这里不是完全从faster rcnn的voc版本来，改成False了，因为RPN源码是False
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(1000, 600),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))          # ?
    
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) #学习率修改，原来0.02(8 GPUs)
# runner configs
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
gpus = 1   # 这行是我自己新增的，原config没有这句，但build_dataloader需要这个参数输入
#total_epochs = 12
total_epochs =1  # 暂时先跑1个epoch，方便看效果
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/rpn_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]

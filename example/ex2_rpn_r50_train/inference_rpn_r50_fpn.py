import mmcv
import numpy as np
import torch

from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.core import get_classes

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale=cfg.data.test.img_scale)
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, cfg, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)


def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device)


def show_result(img, result, dataset='coco', score_thr=0.3):  # 修改数据集名，对应class不同
    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr)


def main():
    cfg = mmcv.Config.fromfile('../configs/faster_rcnn_r50_fpn_1x.py')  # 修改config文件路径
    cfg.model.pretrained = None
    # 创建模型
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    # 加载模型预训练参数
    _ = load_checkpoint(                               # 修改预训练detector模型路径
        model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
    
    # test a single image
    img = mmcv.imread('test2.jpg')                    # 修改测试图片名称，放置在同一文件夹
    result = inference_detector(model, img, cfg)
    show_result(img, result)
    
if __name__ == '__main__':
    main()
    
    
    
    
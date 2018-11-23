"""
2018/11/22 新增transform在voc dataset中，统一由dataset内部进行transform
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

def data_transform(train=True, input_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    '''数据变换：如果要使用dataloader，就必须使用transform
    因为dataset为img格式，而dataloader只认tensor，所以在dataloader之前需要定义transform
    把img转换为tensor。这一步一般放在dataset步完成。
    同时还可以在transform中定义一些数据增强。
    
    '''
    
    if train:
        transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
                                       ])
        
    else:
        transform = transforms.Compose([transforms.Resize(input_size),
                                        transforms.CenterCrop(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
                                       ])
    return transform
    

def read_image(path, dtype=np.float32, color=True):
    """ 读取图片文件：color如果是True则返回3通道图片，color如果False则返回灰度图
    """
    f = Image.open(path)        # (W, H)
    # step1: 字节形式[w,h]转向量形式[c,h,w]    
    # step2: 转置
    
    try:
        if color:                       # f.mode可以看到本来就是RGB模式，所以convert没有影响
            img = f.convert('RGB')      # f(W, H) -> img(W,H) 
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)  # (W,H) - >(H,W,C)
    finally:
        if hasattr(f, 'close'):
            f.close()
            
""" img to array: np.asarray(img)
    array to img: Image.fromarray(np.uint8(img))
    考虑修改read_imge()，只做打开，把数据转换单独写成transform
"""            

    if img.ndim == 2:
        img = img[np.newaxis]  # reshape (H, W) -> (1, H, W)
    else:
        img = img.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)


class VOCBboxDataset:
    '''voc2007数据类: 训练集只有5011张图片，测试集有4952张图片，共计20个分类
    分类中person的图片最多有2008张，最少的sheep有96张，在如此少样本下获得分类效果很难
    图片尺寸：500x375或者375x500
    输入：
        data_dir： 数据的root地址, 比如/home/ubuntu/MyDatasets/VOCdevkit/VOC2007
        split：可选择'train', 'val', 'trainval', 'test'，其中test只对2007voc可用
             train则取train的数据集，val则取validate数据集，trainval则取所有
        use_difficult：默认false
        return_difficult：默认false
        去掉了voc年份 '2007,'2012'的选择，只能用在2007版
    输出：
        img：图片数据
        bbox：bbox坐标[ymin,xmin,ymax,xmax]
        label：标签数据
        difficult：略
    中间的数据处理：
        在read_image()做了一点：包括字节转向量和转置，使用之前还需要做(0,1)化和规范化
        
    '''


    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False):

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """ 返回第i张图片的：图片，bbox，label，difficult
        输入：
            函数绑定—__getitem__()作为切片调用函数，输入index号
        输出：
            img：C，H，W格式
            bbox：几个bbox就有几行，每行4个坐标，比如[ymin,xmin,ymax,xmax]
            label：几个bbox就有几个标签，比如[2,6,7]
            difficult = None
        """
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        
        img = Image.open(img_file)   # img为二进制图片
        data = data_transform(train=True, input_size = 224, mean)(img)
#        img = read_image(img_file, color=True)  # ??? 读出来类型就变成NoneType了
        
        # TODO: add transform here
        
        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return data, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

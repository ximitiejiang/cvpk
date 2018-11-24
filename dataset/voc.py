
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from utils import img_bbox
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
"""VOC数据源核心接口类为VOCTrainDataset和VOCTestDataset
VOCBboxDataset为数据源类。
"""



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

        data = img_bbox.read_image(img_file, color=True)  # 
        
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



def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)   # 用小比例缩放图片，确保图片缩放到框定的HxW(1000x600)之内
    img = img / 255.              # 对象素值做0-1化处理
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)  # 具体的缩放步骤
    # both the longer and shorter should be less than
    # max_size and min_size

    normalize = pytorch_normalze
    return normalize(img)         # 对像素值做规范化处理


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = img_bbox.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = img_bbox.random_flip(
            img, x_random=True, return_param=True)
        bbox = img_bbox.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class VOCTrainDataset:
    """trainset包含transform
    """
    def __init__(self, root, split='trainval', use_difficult=False):
        self.db = VOCBboxDataset(root)
        self.tsf = Transform(min_size=600, max_size=1000)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class VOCTestDataset:
    """testset不包含transform
    """
    def __init__(self, root, split='test', use_difficult=True):
        self.db = VOCBboxDataset(root, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)

if __name__ == '__main__':
    data_root = '/home/ubuntu/MyDatasets/VOCdevkit/VOC2007'
    trainset = VOCTrainDataset(root=data_root)
    
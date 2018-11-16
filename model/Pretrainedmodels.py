#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:49:27 2018

@author: ubuntu
"""


import torch.nn as nn
from torchvision import models

def pretrained_models(model_name, num_classes=2):
    ''' pytorch的预训练模型，初次运行会自行下载到.torch/models文件夹中
        已修改最后一层fc，可自定义分类class数量作为最后一层fc的out_channels
    初始化：
        model_name
        num_classes
    输入：
        no
        
    输出：
    model_ft = models.vgg11_bn(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224
    '''
    
    # vgg16_pretrained
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters(): # 取出每一个参数tensor
            param.requires_grad = False  # 原始模型的梯度锁定
            
        in_fc = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_fc, num_classes) # 替换最后一层fc
        parameters_to_update = model.classifier[6].parameters() 
    
    # resnet18_pretrained
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        
        for param in model.parameters(): # 取出每一个参数tensor
            param.requires_grad = False  # 原始模型的梯度锁定
            
        in_fc = model.fc.in_features
        model.fc = nn.Linear(in_fc, num_classes) # 替换最后一层fc
        parameters_to_update = model.fc.parameters() 

    elif model_name == 'alexnet':
        model = models.AlexNet(pretrained=True)

          
    return model        

'''
# =============================================================================
#  是不是能把pretrained model写成一个类，方便的使用
#  这个new model怎么跟自定义model形成映射关系？
# =============================================================================
class Pretrained_Models(nn.Module):
    
    """ 预训练模型类：
        对象model
        1. outputs = model(inputs)，调用了__call__()函数
        2. model
    """
    def __init__(self, model_name, num_classes=2):
        self.model_name = model_name
        self.num_classes = num_classes
        # vgg16_pretrained
        if self.model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            
            for param in self.model.parameters(): # 取出每一个参数tensor
                param.requires_grad = False  # 原始模型的梯度锁定
                
            in_fc = self.model.fc.in_features
            self.model.fc = nn.Linear(in_fc, num_classes) # 替换最后一层fc
            self.parameters_to_update = self.model.fc.parameters() 
        
        # resnet18_pretrained
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            
            for param in model.parameters(): # 取出每一个参数tensor
                param.requires_grad = False  # 原始模型的梯度锁定
                
            in_fc = self.model.fc.in_features
            self.model.fc = nn.Linear(in_fc, num_classes) # 替换最后一层fc
            self.parameters_to_update = self.model.fc.parameters() 
    
    def __call__(self, inputs):   # 等价于写forward()函数
        return self.model(inputs)
    
    def parameters(self):
        return self.parameters_to_update





if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11":
        """ VGG11_bn
        需要比对以下VGG11_bn是否可以用VGG16的模型代替：主要是最后的结构fc是否一致
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
'''


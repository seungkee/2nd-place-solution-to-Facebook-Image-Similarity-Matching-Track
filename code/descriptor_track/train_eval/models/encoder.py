import torch
from torch import nn
import models
from collections import OrderedDict
from argparse import Namespace
import yaml
import os
import timm

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad=False

class EncodeProject(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if hparams.arch in timm.list_models(pretrained=True):
            self.convnet = timm.create_model(hparams.arch,pretrained=True,num_classes=0)
            self.encoder_dim = self.convnet.num_features 
        elif hparams.arch == 'ResNet50':
            cifar_head = (hparams.data == 'cifar')
            self.convnet = models.resnet.ResNet50(cifar_head=cifar_head, hparams=hparams)
            self.encoder_dim = 2048
        elif hparams.arch == 'resnet18':
            self.convnet = models.resnet.ResNet18(cifar_head=(hparams.data == 'cifar'))
            self.encoder_dim = 512
        elif hparams.arch == 'vit_small':
            self.convnet = models.vision_transformer.__dict__['vit_small'](patch_size= 16)
            ckpt = torch.load('dino_deitsmall16_pretrain.pth')
            self.convnet.load_state_dict(ckpt,strict=True)
            self.encoder_dim = 384
        elif hparams.arch=='vit_base':
            self.convnet = models.vision_transformer.__dict__['vit_base'](patch_size=8)
            ckpt = torch.load('dino_vitbase8_pretrain.pth')
            self.convnet.load_state_dict(ckpt,strict=True)
            self.encoder_dim=768
        else:
            raise NotImplementedError
        """
        elif hparams.arch == 'swin_tiny':
            self.convnet = models.swin_transformer.__dict__['swin_tiny']()
            ckpt = torch.load('swin_t_w7.pth')
            msg=self.convnet.load_state_dict(ckpt,strict=False)
            print(msg)
            self.encoder_dim = 96
        """
        num_params = sum(p.numel() for p in self.convnet.parameters() if p.requires_grad)
        print(f'======> Encoder: output dim {self.encoder_dim} | {num_params/1e6:.3f}M parameters')
        """
        self.proj_dim = 256
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, self.proj_dim, bias=False)),
            ('bn2', BatchNorm1dNoBias(self.proj_dim)),
        ]
        self.projection = nn.Sequential(OrderedDict(projection_layers))
        """
        self.projection=nn.Linear(self.encoder_dim,256,bias=False)

    def forward(self, x, out='z'):
        h=self.convnet(x)
        if self.hparams.arch in ['vit_small', 'vit_base']:
            return h
        #print(h.shape)
        #if out == 'h':
        #    return h
        #return self.projection(h)
        return self.projection(h)

from argparse import Namespace, ArgumentParser
import os
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms
from utils import datautils
import models
from utils import utils
import numpy as np
import PIL
from tqdm import tqdm
import sklearn
from utils.lars_optimizer import LARS
import scipy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy
import glob
import random
import math
import imgaug.augmenters as iaa
import albumentations as A
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import string
from utils.datautils import GaussianBlur
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self,basic_transform,transform,hparams):
        self.basic_transform = basic_transform
        self.transform = transform

    def __call__(self, x, batch_img_indexes):
        return [self.basic_transform(x), self.transform(x, batch_img_indexes)]
        #return [self.basic_transform(x), self.transform(x)]

class OverlayImageOnRandomBackground(torch.nn.Module):
    def __init__(self,backgroundDir='/facebook/data/images/train1M/train',size=(224,224),scale=(0.08,0.5),ratio=(0.5,2.0),opacity=(0.4,1.0),p_scale=(0.25,1.0),p_ratio=(0.5,2.0), bgblack=0):
        super().__init__()
        #self.backgroundFiles=glob.glob(backgroundDir)#os.listdir(backgroundDir)
        #self.backgroundFiles = [os.path.join(backgroundDir, x, x+'.jpg') for x in os.listdir(backgroundDir)]
        self.dirname = '/facebook/data/images/train1M/train/'
        self.samples = list(np.load('/facebook/data/images/train_imlist.npy'))
        self.size=size
        self.scale=scale
        self.ratio=ratio
        self.opacity = opacity
        self.rotate90 = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5)
            ])
        self.partialcrop = transforms.RandomResizedCrop(
                            size[0],
                            scale=p_scale,
                            ratio=p_ratio,
                            interpolation=PIL.Image.BICUBIC,)
        self.bgblack = bgblack
    @staticmethod
    def get_params(img,scale,ratio):
        width,height=img.size
        area=height*width
        for _ in range(10):
            target_area=area*torch.empty(1).uniform_(scale[0],scale[1]).item()
            log_ratio=torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0],log_ratio[1])
            ).item()
            w = int(round(math.sqrt(target_area*aspect_ratio)))
            h = int(round(math.sqrt(target_area/aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i,j,h,w,height,width
        # Fallback to central crop
        in_ratio = float(width)/float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2
        return i,j,h,w,height,width

    def forward(self,img,batch_index_list):
        img = img.convert('RGBA')
        if self.bgblack == 0:
            while 1:
                t=random.randint(0, len(self.samples)-1)
                if t not in batch_index_list:
                    break
            background_path=os.path.join(self.dirname,self.samples[t])#self.backgroundFiles[t]#random.choice(self.backgroundFiles)
            with open(background_path,'rb') as f:
                backImage = Image.open(f)
                backImage = backImage.convert('RGBA')
                backImage = backImage.resize(self.size)
        else:
            backImage = Image.fromarray(np.uint8(np.zeros((self.size[0],self.size[0],4)))) #When opacity is low, Black can be too dark when additaional brightness is applied.
        i,j,h,w,height,width=self.get_params(backImage, self.scale, self.ratio)
        if random.random()>0.5:
            img = self.partialcrop(img)
        img = img.resize((w,h))

        opacity = random.uniform(self.opacity[0], self.opacity[1])
        mask = img.getchannel('A')
        mask = Image.fromarray((np.array(mask)*opacity).astype(np.uint8))

        #overlay=overlay.resize((w,h))
        backImage.paste(im=img,box=(j,i), mask = mask)
        return backImage.convert('RGB'), (j+w//2, i+h//2)

class AttachEmoji(torch.nn.Module):
    def __init__(self,emojiDir='/facebook2/noto-emoji/png/512',scale=(0.08,1.0),ratio=(0.5,2.0),opacity=(0.1,0.9), emoji_ver=0):
        super().__init__()
        #self.backgroundFiles=glob.glob(backgroundDir)#os.listdir(backgroundDir)
        self.emojiFiles = [os.path.join(emojiDir, x) for x in os.listdir(emojiDir)]
        self.scale=scale
        self.ratio=ratio
        self.opacity=opacity
        self.emoji_ver=emoji_ver
        if self.emoji_ver == 1:
            self.scale = (0.08,0.5)
            self.opacity = (0.2,1.0)
            self.emojiCropTransform = transforms.RandomResizedCrop(512, scale=(0.08,0.5), ratio=(0.5,2.0), interpolation=PIL.Image.BICUBIC)
        elif self.emoji_ver == 2:
            self.scale = (0.08,0.5)
            self.opacity = (0.2,0.9)
            self.emojiCropTransform = transforms.RandomResizedCrop(512, scale=(0.08,0.5), ratio=(0.5,2.0), interpolation=PIL.Image.BICUBIC)
        elif self.emoji_ver == 3:
            self.scale = (0.08,0.5)
            self.opacity = (1.0,1.0)
            self.emojiCropTransform = transforms.RandomResizedCrop(512, scale=(0.08,0.5), ratio=(0.5,2.0), interpolation=PIL.Image.BICUBIC)

    @staticmethod
    def get_params(img,scale,ratio):
        width,height=img.size
        area=height*width
        for _ in range(10):
            target_area=area*torch.empty(1).uniform_(scale[0],scale[1]).item()
            log_ratio=torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0],log_ratio[1])
            ).item()
            w = int(round(math.sqrt(target_area*aspect_ratio)))
            h = int(round(math.sqrt(target_area/aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i,j,h,w,height,width
        # Fallback to central crop
        in_ratio = float(width)/float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i,j,h,w,height,width

    def forward(self,img, overlayImageCenter=None):
        """ 
        #RGB version
        emoji_path=random.choice(self.emojiFiles)
        with open(emoji_path, 'rb') as f:
            emojiImage=Image.open(f)
            emojiImage=emojiImage.convert('RGB')
            emojiImage=emojiImage.resize(self.size)
        #Cutout is possible using this.
        #backImage = Image.fromarray(np.uint8(np.zeros((self.size[0],self.size[0],3))))
        img = img.convert('RGBA')
        i,j,h,w,height,width=self.get_params(img,self.scale,self.ratio)
        emojiImage=emojiImage.resize((w,h))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        
        mask = emojiImage.convert('RGBA').getchannel('A')
        mask = Image.fromarray((np.array(mask)*opacity).astype(np.uint8))   
        img.paste(im=emojiImage,box=(j,i),mask=mask)
        return img.convert('RGB')
        """
        emoji_path=random.choice(self.emojiFiles)
        with open(emoji_path,'rb') as f:
            emojiImage=Image.open(f)
            emojiImage=emojiImage.convert('RGBA')
            if self.emoji_ver == 0:
                emojiImage = transforms.RandomRotation([0,360])(emojiImage)
            elif self.emoji_ver in [1,2,3]:
                emojiImage = self.emojiCropTransform(emojiImage)
                a= np.ones([512,512,4])
                a[:,:,:3] *= (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                a[:,:,3] *= (np.array(emojiImage.getchannel('A'))==0)*255
                _a = Image.fromarray(a.astype('uint8'))
                emojiImage.paste(_a, mask=_a.getchannel('A'))
        #emojiImage=emojiImage.resize(self.size)
        #Cutout is possible using this.
        #backImage = Image.fromarray(np.uint8(np.zeros((self.size[0],self.size[0],3))))
        img=img.convert('RGBA')
        while 1:
            i,j,h,w,height,width=self.get_params(img,self.scale,self.ratio)
            if overlayImageCenter is None:
                break
            if overlayImageCenter[0]>=j and overlayImageCenter[0]<=w+j and overlayImageCenter[1]>=i and overlayImageCenter[1]<=i+h:
                continue
            else:
                break
        emojiImage=emojiImage.resize((w,h))
        opacity=random.uniform(self.opacity[0],self.opacity[1])
        mask = emojiImage.getchannel('A')
        mask = Image.fromarray((np.array(mask)*opacity).astype(np.uint8))
        img.paste(im=emojiImage,box=(j,i),mask=mask)
        return img.convert('RGB')


def has_glyph(font, glyph):
     for table in font['cmap'].tables:
         if ord(glyph) in table.cmap.keys():
             return True
     return False

class OverlayText(torch.nn.Module):
    def __init__(self, font_dir='/facebook2/AugLy/augly/assets/fonts/*.ttf', text_len=(5,30), x_pos=(-0.5,0.7),y_pos=(0,0.7),font_size=(0.1,0.3)):
        super().__init__()
        font_files=glob.glob(font_dir)
        good_path = []
        for font_file in font_files:
            font = TTFont(font_file)
            if has_glyph(font, 'a') and has_glyph(font,'A') and has_glyph(font,'1'):
                good_path.append(font_file)
        bad_fonts = ['Outgunned.ttf','TypeMyMusic_1.1.ttf', 'WC_Sold_Out_C_Bta.ttf', 'SirucaPictograms1.1_.ttf', 'modernpics.ttf', 'OstrichSans-Light.ttf',
                    'Entypo.ttf', 'heydings_controls.ttf', 'heydings_icons.ttf', 'WCSoldOutABta.ttf', 'DavysDingbats.ttf', 'WC_Rhesus_B_Bta.ttf', 'WC_Rhesus_A_Bta.ttf',
                    'WC_Sold_Out_B_Bta.ttf', 'Windsong.ttf', 'Kalocsai_Flowers.ttf', 'OstrichSansDashed-Medium.ttf']
        very_good_path = []
        for i in good_path:
            flag=0
            for j in bad_fonts:
                if j in i :
                    flag=1
                    break
            if flag==0:
                very_good_path.append(i)
        print(f'very good font number : {len(very_good_path)}')
        self.font_files = very_good_path
        self.font_size = font_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.text_len = text_len
    def forward(self,img):
        width,height=img.size
        img=img.convert('RGBA')
        font_size = random.uniform(self.font_size[0], self.font_size[1])
        font_size = int(min(width,height)*font_size)
        font = ImageFont.truetype(random.choice(self.font_files),font_size)
        draw = ImageDraw.Draw(img)
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        text_len = random.randint(self.text_len[0], self.text_len[1])
        _text = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits+'    ') for _ in range(text_len))
        _R=random.randint(0,255)
        _G=random.randint(0,255)
        _B=random.randint(0,255)
        _O=random.randint(128,255)
        draw.text(xy=(x_pos*width,y_pos*height),text=(_text),fill=(_R,_G,_B,_O),font=font)
        #blank_img = Image.fromarray(np.zeros_like(np.array(img.convert('RGBA'))))
        #draw=ImageDraw.Draw(blank_img)
        #draw.text(xy=(x_pos*width,y_pos*height), text=(_text), fill=(_R,_G,_B,_O), font=font)
        return img.convert('RGB')

class CartoonAug(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cartoon_filter=iaa.Cartoon()
    def forward(self, img):
        return Image.fromarray(cartoon_filter(images=np.expand_dims(np.array(img),0))[0])

class PathDataset(torch.utils.data.Dataset):
    def __init__(self,paths,transform,hparams):
        self.paths = paths
        self.transform = transform
        self.hparams= hparams
        self.sim = np.load('/storage1/train_features_vit_large_patch16_384.npy_sim_256.npy')
    def __len__(self):
        return len(self.paths)*self.hparams.batch_size*8
    def __getitem__(self, index):
        rank=self.hparams.rank
        if self.hparams.dataset_ver==5:
            batch_size = self.hparams.batch_size*8
            t=index//batch_size+self.hparams.sim_pt_start
            #sim_pt = torch.load(f'/storage1/sim_pt/{t}_sim2000.pt')
            #path=self.paths[sim_pt[index%batch_size]]
            #batch_index_list = sim_pt[:batch_size]
            path=self.paths[self.sim[t][-batch_size:][index%batch_size]]
            batch_index_list = self.sim[t][-batch_size:]
            with open(path,'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                random.seed(t)
                torch.manual_seed(t)
                if self.transform is not None:
                    img=self.transform(img, batch_index_list)
        return img
class BaseSSL(nn.Module):
    """
    Inspired by the PYTORCH LIGHTNING https://pytorch-lightning.readthedocs.io/en/latest/
    Similar but lighter and customized version.
    """
    DATA_ROOT = os.environ.get('DATA_ROOT', os.path.dirname(os.path.abspath(__file__)) + '/data')
    IMAGENET_PATH = os.environ.get('IMAGENET_PATH', '/home/aashukha/imagenet/raw-data/')

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if hparams.data == 'imagenet':
            print(f"IMAGENET_PATH = {self.IMAGENET_PATH}")
    def get_ckpt(self):
        return {
            'state_dict': self.state_dict(),
            'hparams': self.hparams,
        }
    @classmethod
    def load(cls, ckpt, device=None):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=ckpt['hparams'])

        res = cls(hparams, device=device)
        res.load_state_dict(ckpt['state_dict'])
        return res

    @classmethod
    def default(cls, device=None, **kwargs):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=Namespace(**kwargs))
        res = cls(hparams, device=device)
        return res

    def forward(self, x):
        pass

    def transforms(self):
        pass

    def samplers(self):
        return None, None

    def prepare_data(self):
        basic_transform, train_transform = self.transforms()
        # print('The following train transform is used:\n', train_transform)
        # print('The following test transform is used:\n', test_transform)
        if self.hparams.data == 'cifar':
            self.trainset = datasets.CIFAR10(root=self.DATA_ROOT, train=True, download=True, transform=train_transform)
            self.testset = datasets.CIFAR10(root=self.DATA_ROOT, train=False, download=True, transform=train_transform)
        elif self.hparams.data == 'imagenet':
            traindir = '/facebook/data/images/train1M/train/'#os.path.join(self.IMAGENET_PATH, 'train')
            valdir = '/facebook/data/images/train1M/train/' #os.path.join(self.IMAGENET_PATH, 'val')
            if self.hparams.reference == 1:
                traindir='/facebook/data/images/reference_1M_root/'
                valdir='/facebook/data/images/reference_1M_root/'
            print(f'train_dir : {traindir}')
            train_img_paths = sorted([os.path.join(traindir, x, x+'.jpg') for x in os.listdir(traindir)])#[:3320]
            self.trainset = PathDataset(train_img_paths, transform= TwoCropTransform(basic_transform, train_transform, self.hparams),hparams=self.hparams)
            self.testset = PathDataset(train_img_paths, transform = TwoCropTransform(basic_transform, train_transform, self.hparams),hparams=self.hparams)
            #self.trainset = datasets.ImageFolder(traindir, transform=TwoCropTransform(basic_transform, train_transform,self.hparams))
            #self.testset = datasets.ImageFolder(valdir, transform=TwoCropTransform(basic_transform, train_transform, self.hparams))
        else:
            raise NotImplementedError

    def dataloaders(self, iters=None):
        train_batch_sampler, test_batch_sampler = self.samplers()
        if iters is not None:
            train_batch_sampler = datautils.ContinousSampler(
                train_batch_sampler,
                iters
            )
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            #prefetch_factor=1,#1,
            batch_sampler=train_batch_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=test_batch_sampler,
        )
        return train_loader, test_loader

    @staticmethod
    def add_parent_hparams(add_model_hparams):
        def foo(cls, parser):
            for base in cls.__bases__:
                base.add_model_hparams(parser)
            add_model_hparams(cls, parser)
        return foo

    @classmethod
    def add_model_hparams(cls, parser):
        parser.add_argument('--data', help='Dataset to use', default='cifar')
        parser.add_argument('--arch', default='ResNet50', help='Encoder architecture')
        parser.add_argument('--reference', default=0, type=int)
        parser.add_argument('--batch_size', default=256, type=int, help='The number of unique images in the batch')
        parser.add_argument('--aug', default=True, type=bool, help='Applies random augmentations if True')
        parser.add_argument('--dataset_ver', default=0, type=int)

class SimCLR(BaseSSL):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        # loss params
        parser.add_argument('--temperature', default=0.1, type=float, help='Temperature in the NTXent loss')
        # data params
        parser.add_argument('--multiplier', default=2, type=int)
        parser.add_argument('--color_dist_s', default=1., type=float, help='Color distortion strength')
        parser.add_argument('--scale_lower', default=0.25, type=float, help='The minimum scale factor for RandomResizedCrop')
        parser.add_argument('--aug_ver', default=0, type=int)
        #parser.add_argument('--scale_lower', default=1.0, type=float, help='The minimum scale factor for RandomResizedCrop')
        # ddp
        parser.add_argument('--sync_bn', default=True, type=bool,
            help='Syncronises BatchNorm layers between all processes if True'
        )

    def __init__(self, hparams, device=None):
        super().__init__(hparams)
        self.hparams.dist = getattr(self.hparams, 'dist', 'dp')
        model = models.encoder.EncodeProject(hparams)
        self.reset_parameters()
        if device is not None:
            model = model.to(device)
        if self.hparams.dist == 'ddp':
            if self.hparams.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            dist.barrier()
            if device is not None:
                model = model.to(device)
            self.model = DDP(model, [hparams.gpu], find_unused_parameters=True)
        elif self.hparams.dist == 'dp':
            self.model = nn.DataParallel(model)
        else:
            raise NotImplementedError

        self.criterion = models.losses.NTXent(
            tau=hparams.temperature,
            multiplier=hparams.multiplier,
            distributed=(hparams.dist == 'ddp'),
        )

    def reset_parameters(self):
        def conv2d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1. / fan_in) / .87962566103423978
            r = scipy.stats.truncnorm.rvs(-2, 2, loc=0, scale=1., size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                p.normal_(std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_weight_truncated_normal_init(m.weight)
            elif isinstance(m, nn.Linear):
                linear_normal_init(m.weight)

    def step(self, batch):
        #x, _ = batch
        x = batch
        #x = torch.cat([x[0],x[1]],dim=0)
        z = self.model(x)
        loss, acc = self.criterion(z)
        return {
            'loss': loss,
            'contrast_acc': acc,
        }

    def encode(self, x):
        return self.model(x, out='h')

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch, it=None):
        logs = self.step(batch)

        if self.hparams.dist == 'ddp':
            self.trainsampler.set_epoch(it)
        if it is not None:
            logs['epoch'] = it / len(self.batch_trainsampler)

        return logs

    def test_step(self, batch):
        return self.step(batch)

    def samplers(self):
        if self.hparams.dist == 'ddp':
            # trainsampler = torch.utils.data.distributed.DistributedSampler(self.trainset, num_replicas=1, rank=0)
            trainsampler = torch.utils.data.distributed.DistributedSampler(self.trainset, shuffle=False)
            print(f'Process {dist.get_rank()}: {len(trainsampler)} training samples per epoch')
            testsampler = torch.utils.data.distributed.DistributedSampler(self.testset)
            print(f'Process {dist.get_rank()}: {len(testsampler)} test samples')
        else:
            trainsampler = torch.utils.data.sampler.RandomSampler(self.trainset)
            testsampler = torch.utils.data.sampler.RandomSampler(self.testset)

        batch_sampler = datautils.MultiplyBatchSampler
        # batch_sampler.MULTILPLIER = self.hparams.multiplier if self.hparams.dist == 'dp' else 1
        batch_sampler.MULTILPLIER = 1#self.hparams.multiplier

        # need for DDP to sync samplers between processes
        self.trainsampler = trainsampler
        self.batch_trainsampler = batch_sampler(trainsampler, self.hparams.batch_size, drop_last=True)

        return (
            self.batch_trainsampler,
            batch_sampler(testsampler, self.hparams.batch_size, drop_last=True)
        )

    def transforms(self):
        if self.hparams.data == 'cifar':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    32,
                    scale=(self.hparams.scale_lower, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                datautils.get_color_distortion(s=self.hparams.color_dist_s),
                transforms.ToTensor(),
                datautils.Clip(),
            ])
            test_transform = train_transform

        elif self.hparams.data == 'imagenet':
            from utils.datautils import GaussianBlur
            im_size = self.hparams.im_size
            if self.hparams.aug_ver == 0 :
                #im_size = 224
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        im_size,
                        scale=(self.hparams.scale_lower, 1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=self.hparams.color_dist_s),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 1:
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.25, 1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 2:
                #im_size = 224
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.08, 1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 3:
                #im_size = 224
                basic_transform=transforms.Compose([
                transforms.Resize((im_size, im_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0,360)),
                transforms.RandomGrayscale(p=1.0),
                transforms.RandomPerspective(distortion_scale=0.6,p=1.0),
                transforms.ColorJitter(brightness=.5,hue=.3),
                transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                
                train_transform=transforms.Compose([
                transforms.Resize((im_size, im_size)),
                transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.08, 1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0,360)),
                transforms.RandomGrayscale(p=1.0),
                transforms.RandomPerspective(distortion_scale=0.6,p=1.0),
                transforms.ColorJitter(brightness=.5,hue=.3),
                transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            elif self.hparams.aug_ver == 4:
                #im_size = 224
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    transforms.RandomChoice([transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.25, 1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),OverlayImageOnRandomBackground(size=(im_size,im_size))]),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 5:
                #im_size = 224
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    transforms.RandomChoice([transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.25, 1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1)]),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 6:
                #im_size = 224
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    transforms.RandomChoice([transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.25,1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1)]),
                    AttachEmoji(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 9:
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    transforms.RandomChoice([transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.25,1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1)]),
                    AttachEmoji(emoji_ver=1),
                    OverlayText(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    #transforms.RandomChoice(
                    #    [transforms.RandomAffine(degrees=(0,360)),
                    #     transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                    #transforms.RandomApply([CartoonAug], p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 10:
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    transforms.RandomChoice([transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.25,1.0),
                        ratio=(0.5,2.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1)]),
                    AttachEmoji(emoji_ver=1),
                    OverlayText(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    transforms.RandomApply([transforms.RandomChoice(
                        [transforms.RandomAffine(degrees=(0,360)),
                         transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                    #transforms.RandomApply([CartoonAug], p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 11 :
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform=transforms.Compose([
                    transforms.RandomChoice([transforms.RandomResizedCrop(
                        im_size,
                        scale=(0.25,1.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),transforms.Resize((im_size,im_size))]),
                    AttachEmoji(emoji_ver=1),
                    OverlayText(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    transforms.RandomApply([transforms.RandomChoice(
                        [transforms.RandomAffine(degrees=(0,360)),
                         transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                    #transforms.RandomApply([CartoonAug], p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver==12:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    transforms.RandomChoice([
                        transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.25,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ]),
                        transforms.Compose([
                            OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ]),
                        transforms.Compose([
                            OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=0),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5)
                        ])
                    ]),
                    #transforms.RandomApply([CartoonAug], p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            elif self.hparams.aug_ver == 13:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    transforms.RandomChoice([
                        transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.25,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ]),
                        transforms.Compose([
                            OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ]),
                        transforms.Compose([
                            OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=0),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5)
                        ])
                    ]),
                    #transforms.RandomApply([CartoonAug], p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=0.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

            elif self.hparams.aug_ver==14:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = CustomAug(self.hparams)
            
            elif self.hparams.aug_ver==15:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = CustomAug(self.hparams)

            elif self.hparams.aug_ver==16:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = CustomAug2(self.hparams)
            elif self.hparams.aug_ver ==17:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = CustomAug2(self.hparams)

            elif self.hparams.aug_ver == 18:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size, im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform=CustomAug3(self.hparams)

            elif self.hparams.aug_ver == 19:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size, im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform=CustomAug3_imgcorrupt(self.hparams)
            elif self.hparams.aug_ver == 7 :
                #im_size = 224
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    OverlayImageOnRandomBackground(size= (im_size,im_size),bgblack=1),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
            elif self.hparams.aug_ver == 8: #no crop
                basic_transform = transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform = transforms.Compose([
                    OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1),
                    AttachEmoji(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #datautils.Clip(),
                ])
        return basic_transform, train_transform

    def get_ckpt(self):
        return {
            'state_dict': self.model.module.state_dict(),
            'hparams': self.hparams,
        }

    def load_state_dict(self, state):
        k = next(iter(state.keys()))
        if k.startswith('model.module'):
            super().load_state_dict(state)
        else:
            self.model.module.load_state_dict(state)

class CustomAug(object):
    def __init__(self,hparams):
        im_size=hparams.im_size
        self.cropresize=transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.25,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.bgblack=OverlayImageOnRandomBackground(size=(im_size, im_size), bgblack=1)
        self.bgother=OverlayImageOnRandomBackground(size=(im_size, im_size), bgblack=0)
        self.afterbgblack = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.afterbgother = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)], p=0.5),
                            transforms.RandomApply([OverlayText()], p=0.5)
                        ])
        self.invert=A.InvertImg(p=1.0)
        self.postprocess = transforms.Compose([
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    #transforms.Resize((im_size,im_size//2)),
                    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        if hparams.aug_ver == 15:
            self.postprocess = transforms.Compose([
                    datautils.get_color_distortion(s=1.0, grayp=0.5),
                    #transforms.Resize((im_size,im_size//2)),
                    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        
    def __call__(self, image):
        r=random.random()
        if r<1/3:
            image=self.cropresize(image)
        elif r<2/3:
            image =self.bgblack(image)
            image=self.afterbgblack(image)
        else:
            image =self.bgother(image)
            image=self.afterbgother(image)
        if random.random()<0.1:
            image = Image.fromarray(self.invert(image=np.array(image))['image'])
        if random.random()<0.1:
            self.cartoon = iaa.Cartoon(random_state=1)
            image = Image.fromarray(self.cartoon(image=np.array(image)))
        image=self.postprocess(image)
        return image


class CustomAug2(object):
    def __init__(self,hparams):
        im_size=hparams.im_size
        self.hparams = hparams
        self.cropresize=transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.25,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.bgblack=OverlayImageOnRandomBackground(size=(im_size, im_size), bgblack=1)
        self.bgother=OverlayImageOnRandomBackground(size=(im_size, im_size), bgblack=0)
        self.afterbgblack = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.afterbgother = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)], p=0.5),
                            transforms.RandomApply([OverlayText()], p=0.5)
                        ])
        self.invert=A.InvertImg(p=1.0)
        self.postprocess = transforms.Compose([
                transforms.RandomPosterize(bits=2,p=0.1),
                transforms.RandomSolarize(threshold=192.0,p=0.1),
                transforms.RandomEqualize(p=0.1),
                datautils.get_color_distortion(s=1.0, grayp=1.0),
                #transforms.Resize((im_size,im_size//2)),
                #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5),
                transforms.ToTensor(),
                GaussianBlur(im_size // 10, 0.5, im_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
    def __call__(self, image):
        r=random.random()
        if self.hparams.aug_ver == 16:
            if r<1/3:
                image=self.cropresize(image)
            elif r<2/3:
                image =self.bgblack(image)
                image=self.afterbgblack(image)
            else:
                image =self.bgother(image)
                image=self.afterbgother(image)


        elif self.hparams.aug_ver == 17:
            if r<1/2:
                image=self.cropresize(image)
            else:
                image =self.bgblack(image)
                image=self.afterbgblack(image)
        if random.random()<0.1:
            image = Image.fromarray(self.invert(image=np.array(image))['image'])
        if random.random()<0.1:
            self.cartoon = iaa.Cartoon(random_state=1)
            image = Image.fromarray(self.cartoon(image=np.array(image)))
        image=self.postprocess(image)
        return image

class CustomAug3(object):
    def __init__(self, hparams):
        self.im_size=hparams.im_size
        self.hparams = hparams
        self.cropresize=transforms.RandomResizedCrop(
                            self.im_size,
                            scale=(0.25,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC)
        self.rotate90=transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5)
                        ])
        self.attachemoji_1= AttachEmoji(emoji_ver=1)
        self.attachemoji_2= AttachEmoji(emoji_ver=2)
        self.attachemoji_3= AttachEmoji(emoji_ver=3)

        self.attach_text= OverlayText()
        self.bgblack=OverlayImageOnRandomBackground(size=(self.im_size, self.im_size), bgblack=1)
        self.bgother=OverlayImageOnRandomBackground(size=(self.im_size, self.im_size), bgblack=0)

        self.invert=A.InvertImg(p=1.0)
        
        self.basic_spatial = transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])


        self.invert=A.InvertImg(p=1.0)
        self.postprocess = transforms.Compose([
                transforms.RandomPosterize(bits=2,p=0.1),
                transforms.RandomSolarize(threshold=192.0,p=0.1),
                #transforms.RandomEqualize(p=0.1),
                datautils.get_color_distortion(s=1.0, grayp=0.5),
                #transforms.Resize((im_size,im_size//2)),
                #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5),
                transforms.ToTensor(),
                GaussianBlur(self.im_size // 10, 0.5, self.im_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        self.albu_pixel = A.OneOf([
                        A.Blur(),
                        A.CLAHE(),
                        A.ChannelDropout(),
                        A.ChannelShuffle(),
                        A.ColorJitter(),
                        A.Downscale(),
                        A.Emboss(),
                        A.Equalize(),
                        A.FancyPCA(),
                        A.GaussNoise(),
                        A.GaussianBlur(),
                        A.GlassBlur(),
                        A.HueSaturationValue(),
                        A.ISONoise(),
                        A.ImageCompression(),
                        A.InvertImg(),
                        A.MedianBlur(),
                        A.MotionBlur(),
                        A.MultiplicativeNoise(),
                        A.Posterize(),
                        A.RGBShift(),
                        A.RandomBrightnessContrast(),
                        A.RandomFog(),
                        A.RandomGamma(),
                        A.RandomRain(),
                        A.RandomShadow(),
                        A.RandomSnow(),
                        #A.RandomSunFlare(),
                        A.RandomToneCurve(),
                        A.Sharpen(),
                        A.Solarize(),
                        A.Superpixels(),
                        A.ToSepia(),
                        ],p=1.0)

        self.albu_spatial = A.OneOf([
                        A.Affine(),
                        A.Perspective(),
                        A.ShiftScaleRotate()
                        ],p=1.0)

    def __call__(self, image, batch_index_list):
        r=random.random()
        if r<1/3:
            image=self.cropresize(image)
            image=self.rotate90(image)
            if random.random()<0.5:
                if random.random()<0.5:
                    image=self.basic_spatial(image)
                else:
                    image=Image.fromarray(self.albu_spatial(image=np.array(image))['image'])
            if random.random()<0.5:
                image=self.attachemoji_1(image)
            if random.random()<0.5:
                image=self.attach_text(image)

        elif r<2/3:
            if random.random()<0.5:
                image=self.attachemoji_1(image)
            if random.random()<0.5:
                image=self.attach_text(image)
            image, overlayImageCenter = self.bgblack(image, batch_index_list)
            image=self.rotate90(image)
            if random.random()<0.5:
                if random.random()<0.5:
                    image=self.basic_spatial(image)
                else:
                    image=Image.fromarray(self.albu_spatial(image=np.array(image))['image'])
        else:
            if random.random()<0.5:
                image=self.attachemoji_1(image)
            if random.random()<0.5:
                image=self.attach_text(image)
            image, overlayImageCenter = self.bgother(image, batch_index_list)
            r2 = random.randint(2,10)
            if r2==10:
                image=self.attachemoji_3(image, overlayImageCenter=overlayImageCenter)
            image=self.rotate90(image)
            if random.random()<0.5:
                if random.random()<0.5:
                    image=self.basic_spatial(image)
                else:
                    image=Image.fromarray(self.albu_spatial(image=np.array(image))['image'])
            if r2<=9:
                image=self.attachemoji_2(image)
            if random.random()<0.5:
                image=self.attach_text(image)

        if random.random()<0.2:
            image = Image.fromarray(self.invert(image=np.array(image))['image'])
        if random.random()<0.1:
            self.cartoon = iaa.Cartoon(random_state=1)
            image = Image.fromarray(self.cartoon(image=np.array(image)))
        image= Image.fromarray(self.albu_pixel(image=np.array(image))['image'])
        image=self.postprocess(image)
        return image

class CustomAug3_imgcorrupt(object):
    def __init__(self, hparams):
        self.im_size=hparams.im_size
        self.hparams = hparams
        self.cropresize=transforms.RandomResizedCrop(
                            self.im_size,
                            scale=(0.25,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC)
        self.rotate90=transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5)
                        ])
        self.attachemoji_1= AttachEmoji(emoji_ver=1)
        self.attachemoji_2= AttachEmoji(emoji_ver=2)
        self.attachemoji_3= AttachEmoji(emoji_ver=3)

        self.attach_text= OverlayText()
        self.bgblack=OverlayImageOnRandomBackground(size=(self.im_size, self.im_size), bgblack=1)
        self.bgother=OverlayImageOnRandomBackground(size=(self.im_size, self.im_size), bgblack=0)

        self.invert=A.InvertImg(p=1.0)
        
        self.basic_spatial = transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])


        self.invert=A.InvertImg(p=1.0)
        self.postprocess = transforms.Compose([
                transforms.RandomPosterize(bits=2,p=0.1),
                transforms.RandomSolarize(threshold=192.0,p=0.1),
                #transforms.RandomEqualize(p=0.1),
                datautils.get_color_distortion(s=1.0, grayp=0.5),
                #transforms.Resize((im_size,im_size//2)),
                #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5),
                transforms.ToTensor(),
                GaussianBlur(self.im_size // 10, 0.5, self.im_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        self.albu_pixel = A.OneOf([
                        A.Blur(),
                        A.CLAHE(),
                        A.ChannelDropout(),
                        A.ChannelShuffle(),
                        A.ColorJitter(),
                        A.Downscale(),
                        A.Emboss(),
                        A.Equalize(),
                        A.FancyPCA(),
                        A.GaussNoise(),
                        A.GaussianBlur(),
                        A.GlassBlur(),
                        A.HueSaturationValue(),
                        A.ISONoise(),
                        A.ImageCompression(),
                        A.InvertImg(),
                        A.MedianBlur(),
                        A.MotionBlur(),
                        A.MultiplicativeNoise(),
                        A.Posterize(),
                        A.RGBShift(),
                        A.RandomBrightnessContrast(),
                        A.RandomFog(),
                        A.RandomGamma(),
                        A.RandomRain(),
                        A.RandomShadow(),
                        A.RandomSnow(),
                        #A.RandomSunFlare(),
                        A.RandomToneCurve(),
                        A.Sharpen(),
                        A.Solarize(),
                        A.Superpixels(),
                        A.ToSepia(),
                        ],p=1.0)

        self.albu_spatial = A.OneOf([
                        A.Affine(),
                        A.Perspective(),
                        A.ShiftScaleRotate()
                        ],p=1.0)
        self.color_shift=A.Compose([
                    #A.ChannelDropout(p=0.1),
                    A.ChannelShuffle(p=0.5),
                    A.RGBShift(p=0.5),
                ])

        self.aug1=iaa.SaltAndPepper(0.1)
        self.aug2=iaa.Salt(0.1)
        self.aug3=iaa.Pepper(0.1)
        self.aug4=iaa.AdditiveGaussianNoise(scale=0.2*255)
        self.aug5=iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)
        self.aug6=iaa.AdditiveLaplaceNoise(scale=0.2*255)
        self.aug7=iaa.AdditiveLaplaceNoise(scale=0.2*255,per_channel=True)
        self.aug8=iaa.AdditivePoissonNoise(40)
        self.aug9=iaa.AdditivePoissonNoise(40, per_channel=True)

    def __call__(self, image, batch_index_list):
        r=random.random()
        if r<1/3:
            image=self.cropresize(image)
            image=self.rotate90(image)
            if random.random()<0.5:
                if random.random()<0.5:
                    image=self.basic_spatial(image)
                else:
                    image=Image.fromarray(self.albu_spatial(image=np.array(image))['image'])
            if random.random()<0.5:
                image=self.attachemoji_1(image)
            if random.random()<0.5:
                image=self.attach_text(image)

        elif r<2/3:
            if random.random()<0.5:
                image=self.attachemoji_1(image)
            if random.random()<0.5:
                image=self.attach_text(image)
            image, overlayImageCenter = self.bgblack(image, batch_index_list)
            image=self.rotate90(image)
            if random.random()<0.5:
                if random.random()<0.5:
                    image=self.basic_spatial(image)
                else:
                    image=Image.fromarray(self.albu_spatial(image=np.array(image))['image'])
        else:
            if random.random()<0.5:
                image=self.attachemoji_1(image)
            if random.random()<0.5:
                image=self.attach_text(image)
            image, overlayImageCenter = self.bgother(image, batch_index_list)
            r2 = random.randint(2,10)
            if r2==10:
                image=self.attachemoji_3(image, overlayImageCenter=overlayImageCenter)
            image=self.rotate90(image)
            if random.random()<0.5:
                if random.random()<0.5:
                    image=self.basic_spatial(image)
                else:
                    image=Image.fromarray(self.albu_spatial(image=np.array(image))['image'])
            if r2<=9:
                image=self.attachemoji_2(image)
            if random.random()<0.5:
                image=self.attach_text(image)
        
        if random.random()<0.2:
            image = Image.fromarray(self.invert(image=np.array(image))['image'])
        if random.random()<0.1:
            self.cartoon = iaa.Cartoon(random_state=1)
            image = Image.fromarray(self.cartoon(image=np.array(image)))
    
        image= Image.fromarray(self.albu_pixel(image=np.array(image))['image'])
        if random.random()>0.5:
            image = Image.fromarray(self.color_shift(image=np.array(image))['image'])
        if random.random()>0.5:
            r_type = random.randint(1,9)
            if r_type ==1:
                image = Image.fromarray(self.aug1(image=np.array(image)))
            elif r_type ==2:
                image = Image.fromarray(self.aug2(image=np.array(image)))
            elif r_type ==3:
                image = Image.fromarray(self.aug3(image=np.array(image)))
            elif r_type ==4:
                image = Image.fromarray(self.aug4(image=np.array(image)))
            elif r_type ==5:
                image = Image.fromarray(self.aug5(image=np.array(image)))
            elif r_type ==6:
                image = Image.fromarray(self.aug6(image=np.array(image)))
            elif r_type ==7:
                image = Image.fromarray(self.aug7(image=np.array(image)))
            elif r_type ==8:
                image = Image.fromarray(self.aug8(image=np.array(image)))
            elif r_type ==9:
                image = Image.fromarray(self.aug9(image=np.array(image)))

        image=self.postprocess(image)
        return image


"""
class CustomAug(object):
    def __init__(self,hparams):
        self.hparams=hparams
        self.basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                ])
        self.rotate90=transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5)
            ])
        self.affine = transforms.RandomChoice([
            transforms.RandomAffine(degrees=(0,360)),
            transforms.RandomAffine(degrees=0,shear=45)
            ])
        self.partial_crop=transforms.RandomResizedCrop(
            im_size,
            scale=(0.25,1.0),
            ratio=(0.5,2.0),
            interpolation=PIL.Image.BICUBIC,
            )
        self.overlay_on_black=OverlayImageOnRandomBackground(size=(im_size,im_size), bgblack=1)
        self.overlay_on_other=OverlayImageOnRandomBackground(size=(im_size,im_size), bgblack=0)
        self.attach_emoji = AttachEmoji(emoji_ver=1)
        self.overlay_text = OverlayText()
        self.color_jitter = transforms.ColorJitter(0.8,0.8,0.8,0.2)
    def get_color_distortion(s=1.0, grayp=0.2):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=grayp)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort
    def __call__(self,image):
                train_transform = transforms.Compose([
                    transforms.RandomChoice([
                        transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.25,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ]),
                        transforms.Compose([
                            OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=1),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ]),
                        transforms.Compose([
                            OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=0),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5)
                        ])
                    ]),
                    #transforms.RandomApply([CartoonAug], p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.ToTensor(),
                    GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
"""
class SSLEval(BaseSSL):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        parser.add_argument('--test_bs', default=256, type=int)
        parser.add_argument('--encoder_ckpt', default='', help='Path to the encoder checkpoint')
        parser.add_argument('--precompute_emb_bs', default=-1, type=int,
            help='If it\'s not equal to -1 embeddings are precomputed and fixed before training with batch size equal to this.'
        )
        parser.add_argument('--finetune', default=False, type=bool, help='Finetunes the encoder if True')
        parser.add_argument('--augmentation', default='RandomResizedCrop', help='')
        parser.add_argument('--scale_lower', default=0.08, type=float, help='The minimum scale factor for RandomResizedCrop')

    def __init__(self, hparams, device=None):
        super().__init__(hparams)

        self.hparams.dist = getattr(self.hparams, 'dist', 'dp')

        if hparams.encoder_ckpt != '':
            ckpt = torch.load(hparams.encoder_ckpt, map_location=device)
            if getattr(ckpt['hparams'], 'dist', 'dp') == 'ddp':
                ckpt['hparams'].dist = 'dp'
            if self.hparams.dist == 'ddp':
                ckpt['hparams'].dist = 'gpu:%d' % hparams.gpu

            self.encoder = models.REGISTERED_MODELS[ckpt['hparams'].problem].load(ckpt, device=device)
        else:
            print('===> Random encoder is used!!!')
            self.encoder = SimCLR.default(device=device)
        self.encoder.to(device)

        if not hparams.finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif hparams.dist == 'ddp':
            raise NotImplementedError

        self.encoder.eval()
        if hparams.data == 'cifar':
            hdim = self.encode(torch.ones(10, 3, 32, 32).to(device)).shape[1]
            n_classes = 10
        elif hparams.data == 'imagenet':
            hdim = self.encode(torch.ones(10, 3, 224, 224).to(device)).shape[1]
            n_classes = 1000

        if hparams.arch == 'linear':
            model = nn.Linear(hdim, n_classes).to(device)
            model.weight.data.zero_()
            model.bias.data.zero_()
            self.model = model
        else:
            raise NotImplementedError

        if hparams.dist == 'ddp':
            self.model = DDP(model, [hparams.gpu])

    def encode(self, x):
        return self.encoder.model(x, out='h')

    def step(self, batch):
        if self.hparams.problem == 'eval' and self.hparams.data == 'imagenet':
            batch[0] = batch[0] / 255.
        h, y = batch
        if self.hparams.precompute_emb_bs == -1:
            h = self.encode(h)
        p = self.model(h)
        loss = F.cross_entropy(p, y)
        acc = (p.argmax(1) == y).float()
        return {
            'loss': loss,
            'acc': acc,
        }

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch, it=None):
        logs = self.step(batch)
        if it is not None:
            iters_per_epoch = len(self.trainset) / self.hparams.batch_size
            iters_per_epoch = max(1, int(np.around(iters_per_epoch)))
            logs['epoch'] = it / iters_per_epoch
        if self.hparams.dist == 'ddp' and self.hparams.precompute_emb_bs == -1:
            self.object_trainsampler.set_epoch(it)

        return logs

    def test_step(self, batch):
        logs = self.step(batch)
        if self.hparams.dist == 'ddp':
            utils.gather_metrics(logs)
        return logs

    def prepare_data(self):
        super().prepare_data()

        def create_emb_dataset(dataset):
            embs, labels = [], []
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=self.hparams.workers,
                pin_memory=True,
                batch_size=self.hparams.precompute_emb_bs,
                shuffle=False,
            )
            for x, y in tqdm(loader):
                if self.hparams.data == 'imagenet':
                    x = x.to(torch.device('cuda'))
                    x = x / 255.
                e = self.encode(x)
                embs.append(utils.tonp(e))
                labels.append(utils.tonp(y))
            embs, labels = np.concatenate(embs), np.concatenate(labels)
            dataset = torch.utils.data.TensorDataset(torch.FloatTensor(embs), torch.LongTensor(labels))
            return dataset

        if self.hparams.precompute_emb_bs != -1:
            print('===> Precompute embeddings:')
            assert not self.hparams.aug
            with torch.no_grad():
                self.encoder.eval()
                self.testset = create_emb_dataset(self.testset)
                self.trainset = create_emb_dataset(self.trainset)
        
        print(f'Train size: {len(self.trainset)}')
        print(f'Test size: {len(self.testset)}')

    def dataloaders(self, iters=None):
        if self.hparams.dist == 'ddp' and self.hparams.precompute_emb_bs == -1:
            trainsampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
            testsampler = torch.utils.data.distributed.DistributedSampler(self.testset, shuffle=False)
        else:
            trainsampler = torch.utils.data.RandomSampler(self.trainset)
            testsampler = torch.utils.data.SequentialSampler(self.testset)

        self.object_trainsampler = trainsampler
        trainsampler = torch.utils.data.BatchSampler(
            self.object_trainsampler,
            batch_size=self.hparams.batch_size, drop_last=False,
        )
        if iters is not None:
            trainsampler = datautils.ContinousSampler(trainsampler, iters)

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=trainsampler,
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            sampler=testsampler,
            batch_size=self.hparams.test_bs,
        )
        return train_loader, test_loader

    def transforms(self):
        if self.hparams.data == 'cifar':
            trs = []
            if 'RandomResizedCrop' in self.hparams.augmentation:
                trs.append(
                    transforms.RandomResizedCrop(
                        32,
                        scale=(self.hparams.scale_lower, 1.0),
                        interpolation=PIL.Image.BICUBIC,
                    )
                )
            if 'RandomCrop' in self.hparams.augmentation:
                trs.append(transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
            if 'color_distortion' in self.hparams.augmentation:
                trs.append(datautils.get_color_distortion(self.encoder.hparams.color_dist_s))

            train_transform = transforms.Compose(trs + [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                datautils.Clip(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.hparams.data == 'imagenet':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224,
                    scale=(self.hparams.scale_lower, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lambda x: (255*x).byte(),
            ])
            test_transform = transforms.Compose([
                datautils.CenterCropAndResize(proportion=0.875, size=224),
                transforms.ToTensor(),
                lambda x: (255 * x).byte(),
            ])
        return train_transform if self.hparams.aug else test_transform, test_transform

    def train(self, mode=True):
        if self.hparams.finetune:
            super().train(mode)
        else:
            self.model.train(mode)

    def get_ckpt(self):
        return {
            'state_dict': self.state_dict() if self.hparams.finetune else self.model.state_dict(),
            'hparams': self.hparams,
        }

    def load_state_dict(self, state):
        if self.hparams.finetune:
            super().load_state_dict(state)
        else:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(state)
            else:
                self.model.load_state_dict(state)

class SemiSupervisedEval(SSLEval):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        parser.add_argument('--train_size', default=-1, type=int)
        parser.add_argument('--data_split_seed', default=42, type=int)
        parser.add_argument('--n_augs_train', default=-1, type=int)
        parser.add_argument('--n_augs_test', default=-1, type=int)
        parser.add_argument('--acc_on_unlabeled', default=False, type=bool)

    def prepare_data(self):
        super(SSLEval, self).prepare_data()

        if len(self.trainset) != self.hparams.train_size:
            idxs, unlabeled_idxs = sklearn.model_selection.train_test_split(
                np.arange(len(self.trainset)),
                train_size=self.hparams.train_size,
                random_state=self.hparams.data_split_seed,
            )
            if self.hparams.data == 'cifar' or self.hparams.data == 'cifar100':
                if self.hparams.acc_on_unlabeled:
                    self.trainset_unlabeled = copy.deepcopy(self.trainset)
                    self.trainset_unlabeled.data = self.trainset.data[unlabeled_idxs]
                    self.trainset_unlabeled.targets = np.array(self.trainset.targets)[unlabeled_idxs]
                    print(f'Test size (0): {len(self.testset)}')
                    print(f'Unlabeled train size (1):  {len(self.trainset_unlabeled)}')

                self.trainset.data = self.trainset.data[idxs]
                self.trainset.targets = np.array(self.trainset.targets)[idxs]

                print('Training dataset size:', len(self.trainset))
            else:
                assert not self.hparams.acc_on_unlabeled
                if isinstance(self.trainset, torch.utils.data.TensorDataset):
                    self.trainset.tensors = [t[idxs] for t in self.trainset.tensors]
                else:
                    self.trainset.samples = [self.trainset.samples[i] for i in idxs]

                print('Training dataset size:', len(self.trainset))

        self.encoder.eval()
        with torch.no_grad():
            if self.hparams.n_augs_train != -1:
                self.trainset = EmbEnsEval.create_emb_dataset(self, self.trainset, n_augs=self.hparams.n_augs_train)
            if self.hparams.n_augs_test != -1:
                self.testset = EmbEnsEval.create_emb_dataset(self, self.testset, n_augs=self.hparams.n_augs_test)
                if self.hparams.acc_on_unlabeled:
                    self.trainset_unlabeled = EmbEnsEval.create_emb_dataset(
                        self,
                        self.trainset_unlabeled,
                        n_augs=self.hparams.n_augs_test
                    )
        if self.hparams.acc_on_unlabeled:
            self.testset = torch.utils.data.ConcatDataset([
                datautils.DummyOutputWrapper(self.testset, 0),
                datautils.DummyOutputWrapper(self.trainset_unlabeled, 1)
            ])

    def transforms(self):
        ens_train_transfom, ens_test_transform = EmbEnsEval.transforms(self)
        train_transform, test_transform = SSLEval.transforms(self)
        return (
            train_transform if self.hparams.n_augs_train == -1 else ens_train_transfom,
            test_transform if self.hparams.n_augs_test == -1 else ens_test_transform
        )

    def step(self, batch, it=None):
        if self.hparams.problem == 'eval' and self.hparams.data == 'imagenet':
            batch[0] = batch[0] / 255.
        h, y = batch
        if len(h.shape) == 4:
            h = self.encode(h)
        p = self.model(h)
        loss = F.cross_entropy(p, y)
        acc = (p.argmax(1) == y).float()
        return {
            'loss': loss,
            'acc': acc,
        }

    def test_step(self, batch):
        if not self.hparams.acc_on_unlabeled:
            return super().test_step(batch)
        # TODO: refactor
        x, y, d = batch
        logs = {}
        keys = set()
        for didx in [0, 1]:
            if torch.any(d == didx):
                t = super().test_step([x[d == didx], y[d == didx]])
                for k, v in t.items():
                    keys.add(k)
                    logs[k + f'_{didx}'] = v
        for didx in [0, 1]:
            for k in keys:
                logs[k + f'_{didx}'] = logs.get(k + f'_{didx}', torch.tensor([]))
        return logs


def configure_optimizers(args, model, cur_iter=-1):
    iters = args.iters

    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if args.opt == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]

    LR = args.lr

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
        )
    elif args.opt == 'lars':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
        larc_optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    if args.lr_schedule == 'warmup-anneal':
        scheduler = utils.LinearWarmupAndCosineAnneal(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'linear':
        scheduler = utils.LinearLR(optimizer, iters, last_epoch=cur_iter)
    elif args.lr_schedule == 'const':
        scheduler = None
    else:
        raise NotImplementedError

    if args.opt == 'lars':
        optimizer = larc_optimizer

    # if args.verbose:
    #     print('Optimizer : ', optimizer)
    #     print('Scheduler : ', scheduler)

    return optimizer, scheduler

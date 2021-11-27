from argparse import Namespace,ArgumentParser
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
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self,basic_transform,transform,hparams):
        self.basic_transform = basic_transform
        self.transform = transform
    def __call__(self, x, batch_img_indexes):
        return [self.basic_transform(x), self.transform(x, batch_img_indexes)]
        #return [self.basic_transform(x), self.transform(x)]
class OverlayImageOnRandomBackground(torch.nn.Module):
    def __init__(self,doPartialCrop,p_circleMask=0,p_affine=0,size=(224,224),scale=(0.08,0.5),ratio=(0.5,2.0),opacity=(0.8,1.0),p_scale=(0.25,1.0),p_ratio=(0.5,2.0),bgblack=0):
        super().__init__()
        self.samples=[os.path.join('/facebook/data/images/train1M/train',x) for x in list(np.load('/facebook/data/images/train_imlist.npy'))]
        #self.samples=[os.path.join('../input/reference/ref_truth',x) for x in os.listdir('../input/reference/ref_truth')]
        self.size=size
        self.scale=scale
        self.ratio=ratio
        self.opacity = opacity
        self.rotate90 = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5)
            ])
        self.basic_spatial=transforms.RandomAffine(degrees=360, shear=45)
        """
        self.basic_spatial = transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])
        """
        self.doPartialCrop = doPartialCrop
        self.partialcrop = transforms.RandomResizedCrop(
                            size[0],
                            scale=p_scale,
                            ratio=p_ratio,
                            interpolation=PIL.Image.BICUBIC,)
        self.bgblack=bgblack
        self.p_circleMask = p_circleMask
        self.p_affine= p_affine
        def getCircleMask():
            mask=np.ones((224,224))*255
            for x in range(224):
                for y in range(224):
                    a = np.abs(112-x)
                    b = np.abs(112-y)
                    r = np.sqrt(a*a+b*b)
                    mask[x,y]=max(255*(1-(r*r)/(112*112)),0)
            mask=(mask>0)
            mask=mask*255
            mask=mask.astype(np.uint8)
            mask=Image.fromarray(mask)
            return mask
        self.circleMask=getCircleMask()
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
    def forward(self,img,batch_index_list=None):
        img=img.convert('RGBA')
        if self.bgblack == 0:
            t=random.randint(0,len(self.samples)-1)
            if batch_index_list is not None:
                while 1:
                    if t not in batch_index_list:
                        break
                    t=random.randint(0,len(self.samples)-1)
            background_path=self.samples[t]#self.backgroundFiles[t]#random.choice(self.backgroundFiles)
            #print(background_path)
            with open(background_path, 'rb') as f:
                backImage = Image.open(f)
                backImage = backImage.convert('RGBA')
                backImage = backImage.resize(self.size)
        else:
            backImage = Image.fromarray(np.uint8(np.zeros((self.size[0],self.size[0],4)))) #When opacity is low, Black can be too dark when additaional brightness is applied.
        i,j,h,w,height,width=self.get_params(backImage,self.scale,self.ratio)
        if self.doPartialCrop==1:
            if random.random()>0.5:
                img=self.partialcrop(img)
                                        
        img = img.resize((w,h))
        if self.p_circleMask >0:
            if random.random()<self.p_circleMask:
                circleMask=self.circleMask.resize((w,h))
                img_np=np.array(img)
                img_np[:,:,3]= circleMask
                img = Image.fromarray(img_np)
        
        if random.random()<self.p_affine:
            img=self.basic_spatial(img)
                
        opacity = random.uniform(self.opacity[0],self.opacity[1])
        mask = img.getchannel('A')
        mask = Image.fromarray((np.array(mask)*opacity).astype(np.uint8))

        #overlay=overlay.resize((w,h))
        backImage.paste(im=img,box=(j,i), mask=mask)
        return backImage.convert('RGB'), (j+w//2, i+h//2)

class OverlayRandomImageOnImage(torch.nn.Module):
    def __init__(self,backgroundDir='/facebook/data/images/train1M/train',size=(224,224),scale=(0.08,0.5),ratio=(0.5,2.0),opacity=(0.4,1.0),p_scale=(0.25,1.0),p_ratio=(0.5,2.0)):
        super().__init__()
        #self.backgroundFiles=glob.glob(backgroundDir)#os.listdir(backgroundDir)
        #self.backgroundFiles = [os.path.join(backgroundDir, x, x+'.jpg') for x in os.listdir(backgroundDir)]
        self.dirname = '/facebook/data/images/train1M/train/'
        self.samples = list(np.load('/facebook/data/images/train_imlist.npy'))
        self.size=size
        self.scale=scale
        self.ratio=ratio
        self.opacity = opacity
        self.partialcrop = transforms.RandomResizedCrop(
                            size[0],
                            scale=p_scale,
                            ratio=p_ratio,
                            interpolation=PIL.Image.BICUBIC,)
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
        img=img.convert('RGBA')
        #img=img.resize(self.size)
        img=self.partialcrop(img)
        i,j,h,w,height,width=self.get_params(img, self.scale, self.ratio)
        while 1:
            t=random.randint(0, len(self.samples)-1)
            if t not in batch_index_list:
                break

        overlay_path=os.path.join(self.dirname,self.samples[t])#self.backgroundFiles[t]#random.choice(self.backgroundFiles)
        with open(overlay_path,'rb') as f:
            overImage = Image.open(f)
            overImage = overImage.convert('RGBA')
            overImage = overImage.resize((w,h))
                   
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        mask = overImage.getchannel('A')
        mask = Image.fromarray((np.array(mask)*opacity).astype(np.uint8))

        img.paste(im=overImage,box=(j,i),mask=mask)
        #overlay=overlay.resize((w,h))
        #backImage.paste(im=img,box=(j,i), mask = mask)
        return img.convert('RGB')#, (j+w//2, i+h//2)

class AttachEmoji(torch.nn.Module):
    def __init__(self,p_bg=1.0,p_onlybg=0.0,p_circleMask = 0, p_affine=0, scale=(0.05,0.5), ratio=(0.5,2.0), opacity=(0.2,1.0), cropscale=(0.08,0.5), cropratio=(0.5,2.0)):
        super().__init__()
        self.emojiFiles = [os.path.join('/facebook2/noto-emoji/png/512',x) for x in os.listdir('/facebook2/noto-emoji/png/512')]
        #self.emojiFiles = [os.path.join('noto-emoji/png/512', x) for x in os.listdir('noto-emoji/png/512')]
        self.scale=scale
        self.ratio=ratio
        self.opacity=opacity
        self.emojiCropTransform = transforms.RandomResizedCrop(512, scale=cropscale, ratio=cropratio, interpolation=PIL.Image.BICUBIC)
        """
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
        """
        self.p_bg = p_bg
        self.p_onlybg = p_onlybg
        
        self.p_circleMask = p_circleMask
        self.p_affine= p_affine
        def getCircleMask():
            mask=np.ones((224,224))*255
            for x in range(224):
                for y in range(224):
                    a = np.abs(112-x)
                    b = np.abs(112-y)
                    r = np.sqrt(a*a+b*b)
                    mask[x,y]=max(255*(1-(r*r)/(112*112)),0)
            mask=(mask>0)
            mask=mask*255
            mask=mask.astype(np.uint8)
            mask=Image.fromarray(mask)
            return mask
        self.circleMask=getCircleMask()
        """
        self.basic_spatial = transforms.RandomChoice(
                        [transforms.RandomAffine(degrees=(0,360)),
                         transforms.RandomAffine(degrees=0,shear=45)])
        """
        self.basic_spatial = transforms.RandomAffine(degrees=360,shear=45)
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
    def forward(self, img, overlayImageCenter=None):
        emoji_path = random.choice(self.emojiFiles)
        with open(emoji_path, 'rb') as f:
            emojiImage=Image.open(f)
            emojiImage=emojiImage.convert('RGBA')
            #if self.emoji_ver == 0:
            #    emojiImage = transforms.RandomRotation([0,360])(emojiImage)
            #elif self.emoji_ver in [1,2,3]:
            if random.random()>self.p_onlybg:
                emojiImage = self.emojiCropTransform(emojiImage)
                if random.random()<self.p_bg:
                    a= np.ones([512,512,4])
                    a[:,:,:3] *= (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                    a[:,:,3] *= (np.array(emojiImage.getchannel('A'))==0)*255
                    _a = Image.fromarray(a.astype('uint8'))
                    emojiImage.paste(_a, mask=_a.getchannel('A'))
            else:
                a= np.ones([512,512,4])
                a[:,:,:3] *= (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                a[:,:,3] *= 255
                emojiImage=Image.fromarray(a.astype('uint8'))
                
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
        
        if self.p_circleMask >0:
            if random.random()<self.p_circleMask:
                circleMask=self.circleMask.resize((w,h))
                img_np=np.array(emojiImage)
                img_np[:,:,3]= circleMask
                emojiImage = Image.fromarray(img_np)
        
        if random.random()<self.p_affine:
            emojiImage=self.basic_spatial(emojiImage)
        
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

class PathDataset(torch.utils.data.Dataset):
    def __init__(self,paths,transform,hparams):
        self.paths = paths
        self.transform = transform
        self.hparams= hparams
    def __len__(self):
        return len(self.paths)*self.hparams.batch_size*8
    def __getitem__(self,index):
        imgs1=[]
        imgs2=[]
        rank=self.hparams.rank
        im_size = self.hparams.im_size
        if self.hparams.dataset_ver==6:
            batch_size=self.hparams.batch_size*8
            batch_index_list = torch.load(f'/storage1/sim_pt/{index//batch_size+self.hparams.sim_pt_start}_sim2000.pt')[:batch_size]
            seed = index // batch_size
            seed += self.hparams.sim_pt_start+self.hparams.seed
            for t in batch_index_list:
                random.seed(seed)
                torch.manual_seed(seed)
                path = self.paths[t]
                with open(path,'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img, batch_index_list)
                imgs1.append(img[0])
                imgs2.append(img[1])
            imgs_cat=torch.cat([imgs1[index%batch_size].repeat((batch_size,1,1)).reshape((batch_size,3,im_size,im_size//2)),torch.stack(imgs2)], dim=-1)
            label=torch.zeros(batch_size)
            label[index%batch_size]=1
            """
            path=self.paths[torch.load(f'/facebook2/sim_pt/{index}_sim2000.pt')[index%batch_size]]
            with open(path,'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                random.seed(t)
                torch.manual_seed(t)
                if self.transform is not None:
                    img=self.transform(img)
            """
        elif self.hparams.dataset_ver == 8:
            batch_size=self.hparams.batch_size*8
            group_index = (index%batch_size)//8
            elem_index = index%8
            batch_index_list = torch.load(f'/storage1/sim_pt/{index//batch_size}_sim2000.pt')[group_index*8:group_index*8+8]
            for t in batch_index_list:
                random.seed(group_index)
                torch.manual_seed(group_index)
                path = self.paths[t]
                with open(path,'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img, batch_index_list)
                imgs1.append(img[0])
                imgs2.append(img[1])
            imgs_cat=torch.cat([imgs1[elem_index].repeat((8,1,1)).reshape((8,3,im_size,im_size//2)),torch.stack(imgs2)], dim=-1)
            label=torch.zeros(8)
            label[elem_index]=1

        elif self.hparams.dataset_ver == 7:
            batch_size=self.hparams.batch_size*8
            batch_index_list = torch.load(f'/storage1/sim_pt/{index//batch_size}_sim2000.pt')[:batch_size]
            for t in batch_index_list:
                random.seed(index//batch_size)
                torch.manual_seed(index//batch_size)
                path = self.paths[t]
                with open(path,'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img, batch_index_list)
                imgs1.append(img[0])
                imgs2.append(img[1])
            imgs_cat=torch.cat([imgs1[index%batch_size].repeat((batch_size,1,1)).reshape((batch_size,3,im_size,im_size)),torch.stack(imgs2)],dim=-1)
            label=torch.zeros(batch_size)
            label[index%batch_size]=1
        return imgs_cat, label
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
        """
        self.criterion = models.losses.NTXent(
            tau=hparams.temperature,
            multiplier=hparams.multiplier,
            distributed=(hparams.dist == 'ddp'),
        )
        """
        self.criterion = nn.BCEWithLogitsLoss()

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

    def step(self,batch):
        #x, _ = batch
        x,y=batch
        #x = torch.cat([x[0],x[1]],dim=0)
        z=self.model(x)
        loss=self.criterion(z,y)
        return {
            'loss': loss
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

            elif self.hparams.aug_ver == 14:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
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
                            OverlayImageOnRandomBackground(size=(im_size,im_size),bgblack=0),#To Do : make it not duplicate with target images would be great
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5)
                        ])
                        #To Do: use target image as background
                    ]),
                    #transforms.RandomApply([CartoonAug], p=0.5),
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.Resize((im_size,im_size//2)),
                    transforms.ToTensor(),
                    #GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            elif self.hparams.aug_ver == 15:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform=CustomAug_add_beforeoverlay_scale005(self.hparams)

            elif self.hparams.aug_ver == 16:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform=CustomAug_changecolor(self.hparams)

            elif self.hparams.aug_ver == 17:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform=CustomAug_changeemoji_10012111(self.hparams)

            elif self.hparams.aug_ver == 18:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                train_transform=CustomAug_changeemoji_10020526(self.hparams)
            elif self.hparams.aug_ver == 19:
                basic_transform=transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                ])
                train_transform=CustomAug_10071001(self.hparams)
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

class CustomAug_10071001(object):
    def __init__(self, hparams):
        im_size=hparams.im_size
        self.cropresize=transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.1,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                        ])
        self.attachimage = OverlayRandomImageOnImage(scale=(0.05,0.5),ratio=(0.5,2.0),opacity=(0.5,1.0),p_scale=(0.1,1.0),p_ratio=(0.5,2.0))
        self.afterattachimage = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                        ])
        self.bgblack=OverlayImageOnRandomBackground(doPartialCrop=1,p_circleMask=0, p_affine=0, size=(im_size, im_size),scale=(0.05,1.0),opacity=(0.5,1.0),bgblack=1)
        self.bgother=OverlayImageOnRandomBackground(doPartialCrop=1,p_circleMask=0, p_affine=0, size=(im_size, im_size),scale=(0.05,1.0),opacity=(0.5,1.0),bgblack=0)
        self.afteroverlay =  transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                        transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                    ])
        self.postprocess = transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])

        self.color_shift=A.Compose([
                    A.ChannelShuffle(p=0.5),
                    A.RGBShift(p=0.5),
                ])
        self.color_shift2=transforms.Compose([
                transforms.RandomSolarize(threshold=192.0, p=0.5),
        ])
        self.color_shift2_2=transforms.Compose([
                transforms.RandomPosterize(bits=2,p=0.5),
                transforms.RandomSolarize(threshold=192.0,p=0.5)
        ])
        self.color_shift3=transforms.RandomApply([transforms.ColorJitter(0.5,0.8,0.8,0.2)],p=1.0)
        self.color_shift4=A.OneOf([
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
                                #A.RandomShadow(),
                                A.RandomSnow(),
                                #A.RandomSunFlare(),
                                A.RandomToneCurve(),
                                A.Sharpen(),
                                A.Solarize(),
                                #A.Superpixels(),
                                A.ToSepia()
                                ],p=1.0)
        self.blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5)
    def __call__(self,image,batch_index_list):
        r=random.random()
        if r<1/4:
            image=self.cropresize(image)
        elif r<2/4:
            image=self.attachimage(image, batch_index_list)
            image=self.afterattachimage(image)
        elif r<3/4:
            image,overlayImageCenter = self.bgblack(image, batch_index_list)
            image=self.afteroverlay(image)
        else:
            image,overlayImageCenter=self.bgother(image,batch_index_list)
            image=self.afteroverlay(image)
        color_r = random.random()
        if color_r<0.5:
            image = Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image = Image.fromarray(self.color_shift4(image=np.array(image))['image'])
            image = self.color_shift2(image)
        else:
            image=Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image=self.color_shift2_2(image)
            image=self.color_shift3(image)
        image=self.blur(image)
        image=self.postprocess(image)
        return image


class CustomAug_changeemoji_10020526(object):
    def __init__(self, hparams):
        im_size = hparams.im_size
        self.attachemoji_1 = AttachEmoji(opacity=(0.2,1.0))
        self.cropresize=transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.1,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([self.attachemoji_1],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.attachimage = OverlayRandomImageOnImage()
        self.afterattachimage = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([self.attachemoji_1],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        #self.bgblack=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),opacity=(0.4,1.0),bgblack=1)
        #self.bgother=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),opacity=(0.4,1.0),bgblack=0)


        self.bgblack_big=OverlayImageOnRandomBackground(doPartialCrop=1,p_circleMask=0.5, p_affine=0.5, size=(im_size, im_size),scale=(0.25,1.0),opacity=(0.4,1.0),bgblack=1)
        self.bgother_big=OverlayImageOnRandomBackground(doPartialCrop=1,p_circleMask=0.5, p_affine=0.5, size=(im_size, im_size),scale=(0.25,1.0),opacity=(0.4,1.0),bgblack=0)

        self.bgblack_small=OverlayImageOnRandomBackground(doPartialCrop=1,p_circleMask=0.5, p_affine=0.5, size=(im_size,im_size),scale=(0.05,0.25),opacity=(0.8,1.0),bgblack=1)
        self.bgother_small=OverlayImageOnRandomBackground(doPartialCrop=1,p_circleMask=0.5, p_affine=0.5, size=(im_size,im_size),scale=(0.05,0.25),opacity=(0.8,1.0),bgblack=0)

        #self.attachemoji_easy = AttachEmoji(opacity=(0.1,0.5))
        #self.attachemoji_hard = AttachEmoji(opacity=(0.5,1.0))

        self.beforeoverlay = transforms.Compose([
                                transforms.RandomApply([self.attachemoji_1],p=0.5),
                                transforms.RandomApply([OverlayText()],p=0.5)
                            ])
        self.afteroverlay =  transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                        transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                        transforms.RandomApply([OverlayText()], p=0.5)
                    ])

        self.postprocess = transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])

        self.color_shift=A.Compose([
                    #A.ChannelDropout(p=0.1),
                    A.ChannelShuffle(p=0.5),
                    A.RGBShift(p=0.5),
                ])
        self.color_shift2=transforms.Compose([
                transforms.RandomSolarize(threshold=192.0, p=0.5),
        ])
        self.color_shift2_2=transforms.Compose([
                transforms.RandomPosterize(bits=2,p=0.5),
                transforms.RandomSolarize(threshold=192.0,p=0.5)
        ])
        self.color_shift3=transforms.RandomApply([transforms.ColorJitter(0.5,0.8,0.8,0.2)],p=1.0)
        self.color_shift4=A.OneOf([
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
                                #A.RandomShadow(),
                                A.RandomSnow(),
                                #A.RandomSunFlare(),
                                A.RandomToneCurve(),
                                A.Sharpen(),
                                A.Solarize(),
                                #A.Superpixels(),
                                A.ToSepia()
                                ],p=1.0)
        self.blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5)
    def __call__(self,image,batch_index_list):
        r=random.random()
        if r<1/4:
            image=self.cropresize(image)
        elif r<2/4:
            image=self.attachimage(image, batch_index_list)
            image=self.afterattachimage(image)
        elif r<3/4:
            image = self.beforeoverlay(image)
            if random.random()>0.5:
                image,overlayImageCenter = self.bgblack_big(image, batch_index_list)
            else:
                image,overlayImageCenter = self.bgblack_small(image, batch_index_list)
            """
            if random.random()>0.5:
                if random.random()>0.5:
                    image=self.attachemoji_hard(image,overlayImageCenter=overlayImageCenter)
                else:
                    image=self.attachemoji_easy(image)
            """
            image = self.afteroverlay(image)
        else:
            image = self.beforeoverlay(image)
            if random.random()>0.5:
                image,overlayImageCenter=self.bgother_big(image,batch_index_list)
            else:
                image,overlayImageCenter = self.bgother_small(image,batch_index_list)
            """
            if random.random()>0.5:
                if random.random()>0.5:
                    image=self.attachemoji_hard(image,overlayImageCenter=overlayImageCenter)
                else:
                    image=self.attachemoji_easy(image)
            """
            image = self.afteroverlay(image)

        color_r = random.random()
        if color_r<0.5:
            image = Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image = Image.fromarray(self.color_shift4(image=np.array(image))['image'])
            image = self.color_shift2(image)
        else:
            image=Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image=self.color_shift2_2(image)
            image=self.color_shift3(image)
        image=self.blur(image)
        image=self.postprocess(image)
        return image


class CustomAug_changeemoji_10012111(object):
    def __init__(self, hparams):
        im_size = hparams.im_size
        self.attachemoji_1 = AttachEmoji(opacity=(0.2,1.0))
        self.cropresize=transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.1,1.0),
                            ratio=(0.5,2.0),
                            interpolation=PIL.Image.BICUBIC,),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([self.attachemoji_1],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.attachimage = OverlayRandomImageOnImage()
        self.afterattachimage = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([self.attachemoji_1],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        #self.bgblack=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),opacity=(0.4,1.0),bgblack=1)
        #self.bgother=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),opacity=(0.4,1.0),bgblack=0)

        self.bgblack_big=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.25,1.0),opacity=(0.4,1.0),bgblack=1)
        self.bgother_big=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.25,1.0),opacity=(0.4,1.0),bgblack=0)

        self.bgblack_small=OverlayImageOnRandomBackground(size=(im_size,im_size),scale=(0.05,0.25),opacity=(0.8,1.0),bgblack=1)
        self.bgother_small=OverlayImageOnRandomBackground(size=(im_size,im_size),scale=(0.05,0.25),opacity=(0.8,1.0),bgblack=0)

        #self.attachemoji_easy = AttachEmoji(opacity=(0.1,0.5))
        #self.attachemoji_hard = AttachEmoji(opacity=(0.5,1.0))

        self.beforeoverlay = transforms.Compose([
                                transforms.RandomApply([self.attachemoji_1],p=0.5),
                                transforms.RandomApply([OverlayText()],p=0.5)
                            ])
        self.afteroverlay =  transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                        transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                        transforms.RandomApply([OverlayText()], p=0.5)
                    ])

        self.postprocess = transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])

        self.color_shift=A.Compose([
                    #A.ChannelDropout(p=0.1),
                    A.ChannelShuffle(p=0.5),
                    A.RGBShift(p=0.5),
                ])
        self.color_shift2=transforms.Compose([
                transforms.RandomSolarize(threshold=192.0, p=0.5),
        ])
        self.color_shift2_2=transforms.Compose([
                transforms.RandomPosterize(bits=2,p=0.5),
                transforms.RandomSolarize(threshold=192.0,p=0.5)
        ])
        self.color_shift3=transforms.RandomApply([transforms.ColorJitter(0.5,0.8,0.8,0.2)],p=1.0)
        self.color_shift4=A.OneOf([
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
                                #A.RandomShadow(),
                                A.RandomSnow(),
                                #A.RandomSunFlare(),
                                A.RandomToneCurve(),
                                A.Sharpen(),
                                A.Solarize(),
                                #A.Superpixels(),
                                A.ToSepia()
                                ],p=1.0)
        self.blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5)
    def __call__(self,image,batch_index_list):
        r=random.random()
        if r<1/4:
            image=self.cropresize(image)
        elif r<2/4:
            image=self.attachimage(image, batch_index_list)
            image=self.afterattachimage(image)
        elif r<3/4:
            image = self.beforeoverlay(image)
            if random.random()>0.5:
                image,overlayImageCenter = self.bgblack_big(image, batch_index_list)
            else:
                image,overlayImageCenter = self.bgblack_small(image, batch_index_list)
            """
            if random.random()>0.5:
                if random.random()>0.5:
                    image=self.attachemoji_hard(image,overlayImageCenter=overlayImageCenter)
                else:
                    image=self.attachemoji_easy(image)
            """
            image = self.afteroverlay(image)
        else:
            image = self.beforeoverlay(image)
            if random.random()>0.5:
                image,overlayImageCenter=self.bgother_big(image,batch_index_list)
            else:
                image,overlayImageCenter = self.bgother_small(image,batch_index_list)
            """
            if random.random()>0.5:
                if random.random()>0.5:
                    image=self.attachemoji_hard(image,overlayImageCenter=overlayImageCenter)
                else:
                    image=self.attachemoji_easy(image)
            """
            image = self.afteroverlay(image)

        color_r = random.random()
        if color_r<0.5:
            image = Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image = Image.fromarray(self.color_shift4(image=np.array(image))['image'])
            image = self.color_shift2(image)
        else:
            image=Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image=self.color_shift2_2(image)
            image=self.color_shift3(image)
        image=self.blur(image)
        image=self.postprocess(image)
        return image


class CustomAug_changecolor(object):
    def __init__(self, hparams):
        im_size = hparams.im_size
        self.cropresize=transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.1,1.0),
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
        self.attachimage = OverlayRandomImageOnImage()
        self.afterattachimage = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.bgblack=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),bgblack=1)
        self.bgother=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),bgblack=0)
        
        self.beforeoverlay = transforms.Compose([
                                transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                                transforms.RandomApply([OverlayText()],p=0.5)
                            ])

        self.afterbgblack = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=2)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])

        self.emoji3 = AttachEmoji(emoji_ver=3)
        self.afterbgblack_emojiopacity1=transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])

        self.afterbgother = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=2)], p=0.5),
                            transforms.RandomApply([OverlayText()], p=0.5)
                        ])

        self.afterbgother_emojiopacity1 = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                        transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                        transforms.RandomApply([OverlayText()], p=0.5)
                    ])
        self.postprocess = transforms.Compose([
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
 
        self.color_shift=A.Compose([
                    A.ChannelDropout(p=0.1),
                    A.ChannelShuffle(p=0.5),
                    A.RGBShift(p=0.5),
                ])
        self.color_shift2=transforms.Compose([
                transforms.RandomSolarize(threshold=192.0, p=0.5),
        ])
        self.color_shift2_2=transforms.Compose([
                transforms.RandomPosterize(bits=2,p=0.5),
                transforms.RandomSolarize(threshold=192.0,p=0.5)
        ])
        self.color_shift3=transforms.RandomApply([transforms.ColorJitter(0.5,0.8,0.8,0.2)],p=1.0)
        self.color_shift4=A.OneOf([
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
                                #A.RandomShadow(),
                                A.RandomSnow(),
                                #A.RandomSunFlare(),
                                A.RandomToneCurve(),
                                A.Sharpen(),
                                A.Solarize(),
                                #A.Superpixels(),
                                A.ToSepia()
                                ],p=1.0)
        self.blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5)
    def __call__(self,image,batch_index_list):
        r=random.random()
        if r<1/4:
            image=self.cropresize(image)
        elif r<2/4:
            image=self.attachimage(image, batch_index_list)
            image=self.afterattachimage(image)
        elif r<3/4:
            image = self.beforeoverlay(image)
            image,overlayImageCenter = self.bgblack(image, batch_index_list)
            if random.randint(2,10)<=9:
                image=self.afterbgblack(image)
            else:
                if random.random()>0.5:
                    image = self.emoji3(image,overlayImageCenter=overlayImageCenter)
                image = self.afterbgblack_emojiopacity1(image)
        else:
            image = self.beforeoverlay(image)
            image,overlayImageCenter=self.bgother(image,batch_index_list)
            if random.randint(2,10)<=9:
                image=self.afterbgother(image)
            else:
                if random.random()>0.5:
                    image=self.emoji3(image,overlayImageCenter=overlayImageCenter)
                image=self.afterbgother_emojiopacity1(image)
        
        color_r = random.random()
        if color_r<0.5:
            image = Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image = Image.fromarray(self.color_shift4(image=np.array(image))['image'])
            image = self.color_shift2(image)
        else:
            image=Image.fromarray(self.color_shift(image=np.array(image))['image'])
            image=self.color_shift2_2(image)
            image=self.color_shift3(image)
        image=self.blur(image)
        image=self.postprocess(image)
        return image


class CustomAug_add_beforeoverlay_scale005(object):
    def __init__(self, hparams):
        im_size = hparams.im_size
        self.cropresize=transforms.Compose([
                            transforms.RandomResizedCrop(
                            im_size,
                            scale=(0.1,1.0),
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
        self.attachimage = OverlayRandomImageOnImage()
        self.afterattachimage = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0,shear=45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])
        self.bgblack=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),bgblack=1)
        self.bgother=OverlayImageOnRandomBackground(size=(im_size, im_size),scale=(0.05,1.0),bgblack=0)
        
        self.beforeoverlay = transforms.Compose([
                                transforms.RandomApply([AttachEmoji(emoji_ver=1)],p=0.5),
                                transforms.RandomApply([OverlayText()],p=0.5)
                            ])

        self.afterbgblack = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=2)],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])

        self.emoji3 = AttachEmoji(emoji_ver=3)
        self.afterbgblack_emojiopacity1=transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([OverlayText()],p=0.5),
                        ])

        self.afterbgother = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                            transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                            transforms.RandomApply([AttachEmoji(emoji_ver=2)], p=0.5),
                            transforms.RandomApply([OverlayText()], p=0.5)
                        ])

        self.afterbgother_emojiopacity1 = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                        transforms.RandomApply([transforms.RandomChoice(
                                [transforms.RandomAffine(degrees=(0,360)),
                                 transforms.RandomAffine(degrees=0, shear = 45)])],p=0.5),
                        transforms.RandomApply([OverlayText()], p=0.5)
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
                        A.RandomSunFlare(),
                        A.RandomToneCurve(),
                        A.Sharpen(),
                        A.Solarize(),
                        A.Superpixels(),
                        A.ToSepia()
                        ], p=1.0)
        self.albu_spatial = A.OneOf([
                        A.Affine(),
                        A.CoarseDropout(),
                        A.GridDistortion(),
                        A.GridDropout(),
                        A.OpticalDistortion(),
                        A.Perspective(),
                        A.PiecewiseAffine(),
                        A.ShiftScaleRotate()],p=1.0)
        self.invert=A.InvertImg(p=1.0)
        self.postprocess = transforms.Compose([
                    datautils.get_color_distortion(s=1.0, grayp=1.0),
                    transforms.Resize((im_size,im_size//2)),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5),
                    transforms.ToTensor(),
                    #GaussianBlur(im_size // 10, 0.5, im_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
   
    def __call__(self,image,batch_index_list):
        r=random.random()
        if r<1/4:
            image=self.cropresize(image)
        elif r<2/4:
            image=self.attachimage(image, batch_index_list)
            image=self.afterattachimage(image)
        elif r<3/4:
            image = self.beforeoverlay(image)
            image,overlayImageCenter = self.bgblack(image, batch_index_list)
            if random.randint(2,10)<=9:
                image=self.afterbgblack(image)
            else:
                if random.random()>0.5:
                    image = self.emoji3(image,overlayImageCenter=overlayImageCenter)
                image = self.afterbgblack_emojiopacity1(image)
        else:
            image = self.beforeoverlay(image)
            image,overlayImageCenter=self.bgother(image,batch_index_list)
            if random.randint(2,10)<=9:
                image=self.afterbgother(image)
            else:
                if random.random()>0.5:
                    image=self.emoji3(image,overlayImageCenter=overlayImageCenter)
                image=self.afterbgother_emojiopacity1(image)
        
        if random.random()<0.2:
            image = Image.fromarray(self.invert(image=np.array(image))['image'])
        """
        if random.random()<0.2:
            self.cartoon = iaa.Cartoon(random_state=1)
            image = Image.fromarray(self.cartoon(image=np.array(image)))
        """
        image = Image.fromarray(self.albu_pixel(image=np.array(image))['image'])
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

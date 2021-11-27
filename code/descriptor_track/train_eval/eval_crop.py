import cv2
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import models
from utils.logger import Logger
import myexman
from utils import utils
import sys
import torch.multiprocessing as mp
import torch.distributed as dist
import socket
from torchvision import transforms,datasets
from eval_metrics import get_matching_from_descs, evaluate_metrics
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import h5py
def add_learner_params(parser):
    parser.add_argument('--problem', default='sim-clr',
        help='The problem to train',
        choices=models.REGISTERED_MODELS,
    )
    parser.add_argument('--name', default='',
        help='Name for the experiment',
    )
    parser.add_argument('--ckpt', default='',
        help='Optional checkpoint to init the model.'
    )
    parser.add_argument('--split',default='')
    parser.add_argument('--croptype',default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--num_classes', default=3, type=int)
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-anneal')
    parser.add_argument('--opt', default='lars', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=-1, type=int, help='The number of optimizer updates')
    parser.add_argument('--warmup', default=0, type=float, help='The number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=0.1, type=float, help='Base learning rate')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, dest='weight_decay')
    # trainer params
    parser.add_argument('--save_freq', default=1000, type=int, help='Frequency to save the model')
    parser.add_argument('--log_freq', default=100, type=int, help='Logging frequency')
    parser.add_argument('--eval_freq', default=10000000000000000, type=int, help='Evaluation frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, help='The number of data loader workers')
    parser.add_argument('--eval_only', default=False, type=bool, help='Skips the training step if True')
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')
    # transfrom params
    parser.add_argument('--im_size', default=224, type=int)
    parser.add_argument('--allgray', default=0, type=int)
    # parallelizm params:
    parser.add_argument('--dist', default='dp', type=str,
        help='dp: DataParallel, ddp: DistributedDataParallel',
        choices=['dp', 'ddp'],
    )
    parser.add_argument('--dist_address', default='127.0.0.1:1234', type=str,
        help='the address and a port of the main node in the <address>:<port> format'
    )
    parser.add_argument('--node_rank', default=0, type=int,
        help='Rank of the node (script launched): 0 for the main node and 1,... for the others',
    )
    parser.add_argument('--world_size', default=1, type=int,
        help='the number of nodes (scripts launched)',
    )
    parser.add_argument('--best_valid_score', default=0, type=float)
    
def main():
    parser = myexman.ExParser(file=os.path.basename(__file__))
    add_learner_params(parser)

    is_help = False
    if '--help' in sys.argv or '-h' in sys.argv:
        sys.argv.pop(sys.argv.index('--help' if '--help' in sys.argv else '-h'))
        is_help = True

    args, _ = parser.parse_known_args(log_params=False)

    models.REGISTERED_MODELS[args.problem].add_model_hparams(parser)

    if is_help:
        sys.argv.append('--help')

    args = parser.parse_args(namespace=args)

    if args.data == 'imagenet' and args.aug == False:
        raise Exception('ImageNet models should be eval with aug=True!')

    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.gpu = 0
    ngpus = torch.cuda.device_count()
    args.number_of_processes = 1
    if args.dist == 'ddp':
        # add additional argument to be able to retrieve # of processes from logs
        # and don't change initial arguments to reproduce the experiment
        args.number_of_processes = args.world_size * ngpus
        parser.update_params_file(args)

        args.world_size *= ngpus
        mp.spawn(
            main_worker,
            nprocs=ngpus,
            args=(ngpus, args),
        )
    else:
        parser.update_params_file(args)
        main_worker(args.gpu, -1, args)

class FacebookDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, imsize=None):
        if split == 'query':
            self.dirname = '/facebook/data/images/query/'
            self.samples = list(np.load('/facebook/data/images/query_imlist.npy'))
            #gt_df=pd.read_csv('/facebook/data/public_ground_truth.csv')
            #gt_df=gt_df[~gt_df['reference_id'].isnull()]
            #self.samples = [x+'.jpg' for x in list(gt_df['query_id'])]
        elif split == 'query_total':
            self.dirname = '/facebook/data/images/query/'
            self.samples = list(np.load('/facebook/data/images/query_total_imlist.npy'))
        else:
            self.dirname = '/facebook/data/images/reference_1M_root/reference_1M/'
            self.samples = list(np.load('/facebook/data/images/ref_imlist.npy'))
        self.transform = transform
        self.imsize = imsize
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path = os.path.join(self.dirname, self.samples[index])
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.imsize is not None:
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img,index

class FacebookDataset_Crop(torch.utils.data.Dataset):
    def __init__(self,split,transform,croptype):
        if split == 'query_total':
            self.dirname = '/facebook/data/images/query/'
            self.samples = list(np.load('/facebook/data/images/query_total_imlist.npy'))
        elif split == 'ref':
            self.dirname = '/facebook/data/images/reference_1M_root/reference_1M/'
            self.samples = list(np.load('/facebook/data/images/ref_imlist.npy'))
        self.croptype = croptype
        if croptype=='half_grid':
            n=9
        elif croptype=='1of4_grid':
            n=16
        elif croptype=='24of4_zic':
            n=6
        elif croptype=='2of6_grid':
            n=25
        elif croptype=='24of6_zic':
            n=30
        elif croptype=='2of5_grid':
            n=16
        elif croptype=='2of3_grid':
            n=4
        self.modes = np.repeat(np.arange(n),len(self.samples))
        self.samples = self.samples*n
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        path=os.path.join(self.dirname,self.samples[index])
        mode=self.modes[index]
        with open(path,'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            if self.croptype == 'half_grid':
                img = transforms.functional.crop(img,(mode//3)*(img.size[1]//4),(mode%3)*(img.size[0]//4),img.size[1]//2,img.size[0]//2) #top,left,height,width
            elif self.croptype == '1of4_grid':
                img = transforms.functional.crop(img,(mode//4)*(img.size[1]//4),(mode%4)*(img.size[0]//4),img.size[1]//4,img.size[0]//4)
            elif self.croptype == '24of4_zic':
                if mode<3:
                    img = transforms.functional.crop(img,0,(mode%4)*(img.size[0]//4),img.size[1],img.size[0]//2)
                else:
                    img = transforms.functional.crop(img,(mode%4)*(img.size[1]//4),0, img.size[1]//2,img.size[0])
            elif self.croptype == '2of6_grid':
                img = transforms.functional.crop(img,(mode//5)*(img.size[1]//6),(mode%5)*(img.size[0]//6),img.size[1]//3,img.size[0]//3)
            elif self.croptype == '2of5_grid':
                img = transforms.functional.crop(img,(mode//4)*(img.size[1]//5),(mode%4)*(img.size[0]//5),img.size[1]*2//5,img.size[0]*2//5)
            elif self.croptype== '2of3_grid':
                img = transforms.functional.crop(img,(mode//2)*(img.size[1]//3),(mode%2)*(img.size[0]//3),img.size[1]*2//3,img.size[0]*2//3)
            elif self.croptype == '24of6_zic':
                if mode<15:
                    img=transforms.functional.crop(img,(mode//3)*(img.size[1]//6),(mode%3)*(img.size[0]//6),img.size[1]//3,img.size[0]*2//3)
                else:
                    img = transforms.functional.crop(img,(mode%3)*(img.size[1]//6),(mode//3)*(img.size[0]//6),img.size[1]*2//3,img.size[0]//3)
        if self.transform is not None:
            img = self.transform(img)
        return img,index

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    features = None
    for samples, index in tqdm(data_loader):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()
        feats = feats.reshape((feats.shape[0],-1))
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

def valid_all(model, args):
    transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    if args.allgray >= 1:
        transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size)),
        transforms.RandomGrayscale(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
    dataset_total=FacebookDataset_Crop(split=args.split,transform=transform,croptype=args.croptype)
    sampler = torch.utils.data.DistributedSampler(dataset_total, shuffle=False)
    data_loader_total = torch.utils.data.DataLoader(
        dataset_total,
        sampler=sampler,
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    total_features = extract_features(model, data_loader_total, False)
    if args.rank==0:
        total_features=nn.functional.normalize(total_features,dim=1,p=2).cpu().numpy()
        print(total_features.shape)
        if args.ckpt!='':
            newdir=os.path.join(os.path.dirname(args.ckpt),f'{args.split}_{args.croptype}')
        else:
            newdir=os.path.join(args.output_dir,f'{args.split}_{args.croptype}')
        os.makedirs(newdir,exist_ok=True)
        np.save(os.path.join(newdir,f'{args.split}_features_{args.croptype}.npy'),total_features)

def main_worker(gpu, ngpus, args):
    fmt = {
        'train_time': '.3f',
        'val_time': '.3f',
        'lr': '.1e',
    }
    logger = Logger('logs', base=args.root, fmt=fmt)

    args.gpu = gpu
    torch.cuda.set_device(gpu)
    args.rank = args.node_rank * ngpus + gpu

    device = torch.device('cuda:%d' % args.gpu)

    if args.dist == 'ddp':
        dist.init_process_group(
            backend='nccl',
            init_method = 'tcp://%s' % args.dist_address, #'env://',
            world_size=args.world_size,
            rank=args.rank,
        )
        n_gpus_total = dist.get_world_size()
        assert args.batch_size % n_gpus_total == 0
        args.batch_size //= n_gpus_total
        if args.rank == 0:
            print(f'===> {n_gpus_total} GPUs total; batch_size={args.batch_size} per GPU')

        print(f'===> Proc {dist.get_rank()}/{dist.get_world_size()}@{socket.gethostname()}', flush=True)

    # create model
    model = models.REGISTERED_MODELS[args.problem](args, device=device)

    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    # Data loading code
    #model.prepare_data()
    #train_loader, val_loader = model.dataloaders(iters=args.iters)

    # define optimizer
    cur_iter=0
    #optimizer,scheduler = models.ssl.configure_optimizers(args, model, cur_iter - 1)

    # optionally resume from a checkpoint
    #if args.ckpt and not args.eval_only:
    #    optimizer.load_state_dict(ckpt['opt_state_dict'])

    cudnn.benchmark = True

    continue_training = args.iters != 0
    data_time, it_time = 0, 0

    while continue_training:
        model.eval()
        valid_all(model,args)
        break
        """
        train_logs = []
        model.train()

        start_time = time.time()
        for _, (batch,labels) in enumerate(train_loader):
            #print(len(batch))
            #print(batch)
            cur_iter += 1
            #batch = torch.cat([batch[0], batch[1]],dim=0)
            #batch = batch.to(device)#[x.to(device) for x in batch]
            batch = [x.to(device) for x in batch]
            data_time += time.time() - start_time
            logs = {}
            if not args.eval_only:
                # forward pass and compute loss
                logs = model.train_step(batch, cur_iter)
                loss = logs['loss']
                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # save logs for the batch
            train_logs.append({k: utils.tonp(v) for k, v in logs.items()})

            #if cur_iter % args.save_freq == 0 and args.rank == 0:
            #    save_checkpoint(args.root, model, optimizer, cur_iter)
            if cur_iter%args.eval_freq==0 or cur_iter>=args.iters or cur_iter==1:
                model.eval()
                valid_one_epoch(model,args)
                model.train()
            it_time += time.time() - start_time

            if (cur_iter % args.log_freq == 0 or cur_iter >= args.iters) and args.rank == 0:
                save_checkpoint(args.root, model, optimizer, cur_iter = cur_iter)
                train_logs = utils.agg_all_metrics(train_logs)

                logger.add_logs(cur_iter, train_logs, pref='train_')
                logger.add_scalar(cur_iter, 'lr', optimizer.param_groups[0]['lr'])
                logger.add_scalar(cur_iter, 'data_time', data_time)
                logger.add_scalar(cur_iter, 'it_time', it_time)
                logger.iter_info()
                logger.save()
                data_time, it_time = 0, 0
                train_logs = []

            if scheduler is not None:
                scheduler.step()

            if cur_iter >= args.iters:
                continue_training = False
                break

            start_time = time.time()
        """
    # save_checkpoint(args.root, model, optimizer)

    if args.dist == 'ddp':
        dist.destroy_process_group()

def save_checkpoint(path, model, optimizer, cur_iter=None, is_best=False):
    if cur_iter is None:
        fname = os.path.join(path,'checkpoint.pth.tar')
        if is_best :
            fname = os.path.join(path,'checkpoint_best.pth.tar')
    else:
        fname = os.path.join(path, 'checkpoint-%d.pth.tar' % cur_iter)
    ckpt = model.get_ckpt()
    ckpt.update(
        {
            'opt_state_dict': optimizer.state_dict(),
            'iter': cur_iter,
        }
    )
    torch.save(ckpt,fname)

if __name__ == '__main__':
    main()


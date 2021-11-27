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
    parser.add_argument('--doTrainEval',default=0,type=int)
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
        main_worker(args.gpu,-1,args)

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
        elif split == 'train':
            self.dirname = '/facebook/data/images/train1M/train/'
            self.samples =  list(np.load('/facebook/data/images/train_imlist.npy'))
        elif split == 'ref':
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
        return img, index

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

def valid_train(model,args):
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
    dataset_train=FacebookDataset(split="train", transform=transform)
    sampler1 = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler1,
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    train_features=extract_features(model,data_loader_train,True)
    if args.rank==0:
        if args.ckpt!='':        
            newdir = os.path.join(os.path.dirname(args.ckpt), 'basic_query')
            os.makedirs(newdir, exist_ok=True)
            np.save(os.path.join(newdir,f'train_features.npy'),train_features.cpu().numpy())
        else:
            train_features=nn.functional.normalize(train_features,dim=1,p=2).cpu()
            newdir='basic_query'
            os.makedirs(newdir,exist_ok=True)
            torch.save(train_features,os.path.join(newdir,f'train_features_{args.arch}.pt'))
            np.save(os.path.join(newdir,f'train_features_{args.arch}.npy'),train_features.numpy())
            
            newdir='/storage1'
            os.makedirs(newdir,exist_ok=True)
            torch.save(train_features,os.path.join(newdir,f'train_features_{args.arch}.pt'))
            np.save(os.path.join(newdir,f'train_features_{args.arch}.npy'),train_features.numpy())
        

def valid_query_tta(model,args):
    transform_list = []
    transform_list.append(transforms.Compose([
        transforms.Resize((args.im_size,args.im_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ]))
    for i in range(1,4):
        transform_list.append(transforms.Compose([
        transforms.Resize((args.im_size,args.im_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomApply([transforms.RandomRotation((i*90,i*90))],p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ]))
    for i in range(1,4):
        transform_list.append(transforms.Compose([
        transforms.Resize((args.im_size,args.im_size)),
        transforms.RandomApply([transforms.RandomRotation((i*90,i*90))],p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ]))
    if args.allgray >= 1:
        transform_list=[]
        transform_list.append(transforms.Compose([
        transforms.Resize((args.im_size,args.im_size)),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ]))
        for i in range(1,4):
            transform_list.append(transforms.Compose([
            transforms.Resize((args.im_size,args.im_size)),
            transforms.RandomGrayscale(p=1.0),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomApply([transforms.RandomRotation((i*90,i*90))],p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ]))
        for i in range(1,4):
            transform_list.append(transforms.Compose([
            transforms.Resize((args.im_size,args.im_size)),
            transforms.RandomGrayscale(p=1.0),
            transforms.RandomApply([transforms.RandomRotation((i*90,i*90))],p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ]))
    split = 'query_total'#'query_total','ref'
    for i in range(len(transform_list)):
        dataset_query_total = FacebookDataset(split=split, transform=transform_list[i])
        sampler = torch.utils.data.DistributedSampler(dataset_query_total, shuffle=False)
        data_loader_query_total = torch.utils.data.DataLoader(
            dataset_query_total,
            sampler=sampler,
            batch_size=16,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )
        query_total_features = extract_features(model, data_loader_query_total, True)
        if args.rank==0:
            if args.ckpt!='':
                newdir = os.path.join(os.path.dirname(args.ckpt),'basic_query')
            else:
                newdir = os.path.join(args.output_dir,'basic_query')
            os.makedirs(newdir,exist_ok=True)
            query_total_features = query_total_features.cpu().numpy()
            #query_total_features = nn.functional.normalize(query_total_features, dim=1, p=2).cpu().numpy()
            np.save(os.path.join(newdir,f'{split}_features_tta{i}.npy'), query_total_features)
        dist.barrier()

def valid_ref(model, args):
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
    dataset_ref=FacebookDataset(split="ref",transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_ref, shuffle=False)
    data_loader_ref = torch.utils.data.DataLoader(
        dataset_ref,
        sampler=sampler,
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    ref_features = extract_features(model, data_loader_ref, True)
    if args.rank==0:
        if args.ckpt!='':
            newdir = os.path.join(os.path.dirname(args.ckpt), 'basic_query')
        else:
            newdir = os.path.join(args.output_dir,'basic_query')
        os.makedirs(newdir, exist_ok=True)
        np.save(os.path.join(newdir,'ref_features.npy'),ref_features.cpu().numpy())

def valid_total_query(model, args):
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
    dataset_query_total = FacebookDataset(split="query_total", transform=transform)
    sampler3 = torch.utils.data.DistributedSampler(dataset_query_total, shuffle=False)
    data_loader_query_total = torch.utils.data.DataLoader(
        dataset_query_total,
        sampler=sampler3,
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    query_total_features = extract_features(model, data_loader_query_total, True)
    if args.rank==0:
        if args.ckpt!='':
            newdir = os.path.join(os.path.dirname(args.ckpt), 'basic_query')
        else:
            newdir = os.path.join(args.output_dir,'basic_query')
        os.makedirs(newdir, exist_ok=True)
        np.save(os.path.join(newdir,'query_total_features.npy'),query_total_features.cpu().numpy())

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
        if args.doTrainEval == 1:
            valid_train(model,args)
        elif args.doTrainEval == 2:
            valid_query_tta(model, args)
        elif args.doTrainEval==3:
            valid_total_query(model,args)
        elif args.doTrainEval==0:
            valid_ref(model,args)
        elif args.doTrainEval==5:
            valid_train(model,args)
            valid_query_tta(model, args)
            valid_ref(model,args)
            valid_total_query(model,args)
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


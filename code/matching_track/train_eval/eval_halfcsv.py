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
    parser.add_argument('--total_df', default='')
    parser.add_argument('--gridtype',default='')
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
    parser.add_argument('--allgray', default=1, type=int)
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

class FacebookDataset_match_halfcsv(torch.utils.data.Dataset):
    def __init__(self,transform_q=None,transform_r=None,submission_df=None,ref_rotate=0):
        valid_df = submission_df #pd.read_csv('/facebook/data/matching/valid_df.csv')
        self.query_paths = ['/facebook/data/images/query/'+x+'.jpg' for x in valid_df['query_id']]
        self.ref_paths = ['/facebook/data/images/reference_1M_root/reference_1M/'+x+'.jpg' for x in valid_df['reference_id']]
        self.modes = np.array(valid_df['mode'])
        self.transform_q = transform_q
        self.transform_r= transform_r
        self.ref_rotate=ref_rotate
    def __len__(self):
        return len(self.query_paths)
    def __getitem__(self, index):
        with open(self.query_paths[index], 'rb') as f:
            img_query=Image.open(f)
            img_query=img_query.convert('RGB')
            mode = self.modes[index]
            if mode=='left':
                img_query=transforms.functional.crop(img_query,0,0,img_query.size[1],img_query.size[0]//2) #top,left,height,width
            elif mode=='right':
                img_query=transforms.functional.crop(img_query,0,img_query.size[0]//2,img_query.size[1],img_query.size[0]//2)
            elif mode=='vertical_center':
                img_query=transforms.functional.crop(img_query,0,img_query.size[0]//4,img_query.size[1],img_query.size[0]//2)
            elif mode=='bottom':
                img_query=img_query.resize((min(img_query.size),min(img_query.size)))
                img_query=transforms.functional.rotate(img_query,90)
                img_query=transforms.functional.crop(img_query,0,0,img_query.size[1],img_query.size[0]//2)
            elif mode=='top':
                img_query=img_query.resize((min(img_query.size),min(img_query.size)))
                img_query=transforms.functional.rotate(img_query,90)
                img_query=transforms.functional.crop(img_query,0,img_query.size[0]//2,img_query.size[1],img_query.size[0]//2)
            elif mode=='horizontal_center':
                img_query=img_query.resize((min(img_query.size),min(img_query.size)))
                img_query=transforms.functional.rotate(img_query,90)
                img_query=transforms.functional.crop(img_query,0,img_query.size[0]//4,img_query.size[1],img_query.size[0]//2)
        with open(self.ref_paths[index],'rb') as f:
            img_ref = Image.open(f)
            img_ref = img_ref.convert('RGB')
            if self.ref_rotate==1:
                img_ref= img_ref.resize((min(img_ref.size),min(img_ref.size)))
                img_ref=transforms.functional.rotate(img_ref,90)
        img_query = self.transform_q(img_query)
        img_ref = self.transform_r(img_ref)
        return torch.cat([img_ref,img_query],dim=-1),index
    

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

def valid_all(model,args):
    transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size//2)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    gray_transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size//2)),
        transforms.RandomGrayscale(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    submission_df=pd.read_csv(args.total_df)
    """
    if args.gridtype == 'half':
        d={0:'left', 1:'vertical_center', 2:'right'}
        submission_df['mode']=[d[x] for x in np.repeat(np.repeat(np.arange(9),50000),10)%3]

    elif args.gridtype == '2of6' :
        d={0:'left',1:'left',2:'vertical_center',3:'right',4:'right'}
        submission_df['mode']= [d[x] for x in np.repeat(np.repeat(np.arange(25),50000),5)%5]
    """

    submission_df=submission_df.drop_duplicates(subset=['query_id','reference_id','mode'],keep='last').reset_index(drop=True)
    ref_rotate=0
    if args.allgray == 1:
        dataset = FacebookDataset_match_halfcsv(transform_q=gray_transform, transform_r=gray_transform, submission_df = submission_df,  ref_rotate=ref_rotate)
    else:
        dataset = FacebookDataset_match_halfcsv(transform_q=transform, transform_r=transform, submission_df = submission_df, ref_rotate=ref_rotate)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    outputs=extract_features(model,data_loader,True)
    if args.rank == 0:
        gt_df=pd.read_csv('/facebook/data/public_ground_truth.csv')
        outputs=torch.sigmoid(outputs).cpu().numpy()
        target_df=submission_df.copy()
        target_df['score']=outputs
        target_df.to_csv(args.total_df+f'_withhalfmatchscore.csv', index=False)
    dist.barrier()
    """
    submission_df = pd.read_csv(args.total_df)
    submission_df= submission_df.drop_duplicates(subset=['query_id','reference_id'],keep='last').reset_index(drop=True)
    best_ap=0
    for ref_rotate in [0,1]:
        for mode in ['whole','left','right','vertical_center','bottom','top','horizontal_center']:
            if args.allgray == 1:
                dataset = FacebookDataset_match_half(transform_q=gray_transform, transform_r=gray_transform, submission_df = submission_df,mode=mode, ref_rotate=ref_rotate)
            else:
                dataset = FacebookDataset_match_half(transform_q=transform, transform_r=transform, submission_df = submission_df, mode=mode, ref_rotate=ref_rotate)
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=16,
                num_workers=16,
                pin_memory=True,
                drop_last=False,
            )
            outputs=extract_features(model,data_loader,True)
            if args.rank == 0:
                gt_df=pd.read_csv('/facebook/data/public_ground_truth.csv')
                outputs=torch.sigmoid(outputs).cpu().numpy()
                target_df=submission_df.copy()
                if best_ap ==0:
                    target_df['score']=outputs
                else:
                    target_df['score']=[max(a,b) for a,b in zip(np.array(target_df['score']),outputs)]
                with open(args.total_df+'_halfmatchlogs.out',"a") as f:
                    f.write(f'{mode},{ref_rotate}\n')
                print(mode,ref_rotate)
                ap,rp90=evaluate_metrics(target_df,gt_df)
                with open(args.total_df+'_halfmatchlogs.out',"a") as f:
                    f.write(f'ap : {ap}, rp90 : {rp90}\n')
                print(f'ap : {ap}, rp90 : {rp90}')
                if ap>best_ap:
                    print('best found!')
                    with open(args.total_df+'_halfmatchlogs.out',"a") as f:
                        f.write(f'best found!\n')
                    best_ap=ap
                    submission_df=target_df.copy()
                    submission_df.to_csv(args.total_df+f'_halfeval.csv',index=False)
            dist.barrier()
    """
    """
    ref_rotate=1
    for mode in ['whole','left','right','vertical_center','bottom','top','horizontal_center']:
        submission_df = pd.read_csv(args.total_df)
        submission_df= submission_df.drop_duplicates(subset=['query_id','reference_id'],keep='last').reset_index(drop=True)
        if args.allgray == 1:
            dataset = FacebookDataset_match_half(transform_q=gray_transform, transform_r=gray_transform, submission_df = submission_df,mode=mode, ref_rotate=ref_rotate)
        else:
            dataset = FacebookDataset_match_half(transform_q=transform, transform_r=transform, submission_df = submission_df, mode=mode, ref_rotate=ref_rotate)
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=16,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )
        outputs=extract_features(model,data_loader,True)
        if args.rank == 0:
            gt_df=pd.read_csv('/facebook/data/public_ground_truth.csv')
            outputs=torch.sigmoid(outputs).cpu().numpy()
            submission_df['score']=outputs
            submission_df.to_csv(args.total_df+f'_withmatchscore_{mode}_refrotate{ref_rotate}.csv', index=False)
            ap,rp90=evaluate_metrics(submission_df, gt_df)
            with open(args.total_df+'_matchlogs.out',"a") as f:
                f.write(f'ap : {ap}, rp90 : {rp90}\n')
            print(f'ap : {ap}, rp90 : {rp90}')
        dist.barrier()
    """
def valid_one_epoch(model, args):
    transform = transforms.Compose([
        transforms.Resize((args.im_size,args.im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    if args.allgray >= 1:
        transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size)),
        transforms.RandomGrayscale(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
    #dataset_ref = FacebookDataset(split="ref", transform=transform)
    dataset_query = FacebookDataset(split="query", transform=transform)
    """
    sampler = torch.utils.data.DistributedSampler(dataset_ref, shuffle=False)
    data_loader_ref = torch.utils.data.DataLoader(
        dataset_ref,
        sampler=sampler,
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    """
    """
    sampler2 = torch.utils.data.DistributedSampler(dataset_query,shuffle=False)
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        sampler=sampler2,
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    """
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    #ref_features = extract_features(model, data_loader_ref, True)
    #query_features = extract_features(model, data_loader_query, True)
    criterion = nn.BCEWithLogitsLoss()

    query_features = []
    for samples, index in data_loader_query:
        samples = samples.cuda()
        feats=model(samples).clone()
        
    if args.rank==0:
        #ref_features = nn.functional.normalize(ref_features, dim=1, p=2).cpu().numpy()
        #query_features = nn.functional.normalize(query_features, dim=1, p=2).cpu().numpy()
        #query_id_list = np.array([x[:-4] for x in list(np.load('/facebook/data/images/query_imlist.npy'))])
        #ref_truth_id_list = np.array([x[:-4] for x in list(np.load('/facebook/data/images/ref_truth_imlist.npy'))])
        #gt_df=pd.read_csv('/facebook/data/public_ground_truth.csv')
        #submission_df = get_matching_from_descs(query_features, ref_features, query_id_list, ref_truth_id_list, gt_df)
        #ap, rp90 = evaluate_metrics(submission_df, gt_df)
        """
        if args.best_valid_score < ap :
            args.best_valid_score = ap
            np.save(os.path.join(args.root,f'ref_features.npy'),ref_features)
            np.save(os.path.join(args.root,f'query_features.npy'),query_features)
        with open(os.path.join(args.root,'logs.out'),"a") as f:
            f.write(f'ap : {ap}, rp90 : {rp90}\n')
        print(f'ap : {ap}, rp90 : {rp90}')
        """
        pts=[]
        pts.append([65,60])
        pts.append([105,60])
        pts.append([105,135])
        pts.append([65,135])
        blank_image = np.zeros((224,224),np.uint8)
        mask = cv2.fillPoly(blank_image, pts=[np.array(pts)],color=1)
        mask = np.expand_dims(mask,-1)
        mask = mask.astype(np.float32)
        mask = mask.transpose(2,0,1).clip(0,1)
        mask = np.expand_dims(mask,0)
        loss_value = criterion(feats,torch.tensor(mask).cuda()).item()
        print(f'valid loss : {loss_value}')
        with open(os.path.join(args.root,'logs.out'),"a") as f:
            f.write(f'valid loss : {loss_value}\n')

        
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


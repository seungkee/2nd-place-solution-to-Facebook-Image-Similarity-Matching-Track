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
    parser.add_argument('--verbose', default=False, type=bool)
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
    parser.add_argument('--sim_pt_start',default=0,type=int)
    parser.add_argument('--repo_path',default='')
    parser.add_argument('--debug',default=1, type=int)
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
    """
    if args.debug==1:
        args.iters=100
        args.sim_pt_start=0
        #args.batch_size=4
    """
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
       
def main_worker(gpu,ngpus,args):
    fmt={
        'train_time': '.3f',
        'val_time': '.3f',
        'lr': '.1e',
    }
    logger = Logger('logs',base=args.repo_path, fmt=fmt)

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
    model.prepare_data()
    train_loader, val_loader = model.dataloaders(iters=args.iters)

    # define optimizer
    cur_iter=0
    optimizer,scheduler = models.ssl.configure_optimizers(args, model, cur_iter - 1)

    # optionally resume from a checkpoint
    if args.ckpt and not args.eval_only:
        optimizer.load_state_dict(ckpt['opt_state_dict'])

    if args.ckpt != '':
        del ckpt
        torch.cuda.empty_cache()

    cudnn.benchmark = True
    continue_training=args.iters!=0
    data_time,it_time=0,0

    #model.eval()
    #valid_one_epoch(model,args)
    #model.train()
    #dist.barrier()
    while continue_training:
        train_logs=[]
        model.train()
        start_time=time.time()
        for _, (batch,label) in enumerate(train_loader):
            #print(batch[0].shape)
            #batch0 = torch.cat(batch[0])
            #batch1 = torch.cat(batch[1])
            one_gpu_batch = batch.shape[0]*batch.shape[1]
            cur_iter += 1
            
            im_size = args.im_size
            #print('called')
            batch = batch.reshape((one_gpu_batch,3,im_size,im_size))
            label = label.reshape((one_gpu_batch,1))
            #batch = torch.cat(batch)
            #label = torch.cat(label)

            batch = batch.to(device)
            label = label.to(device)
            
            batch = [batch,label]
            data_time+=time.time() - start_time
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

            it_time += time.time() - start_time

            if (cur_iter % args.log_freq == 0 or cur_iter >= args.iters) and args.rank == 0:
                save_checkpoint(args.repo_path, model, optimizer)
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

    save_checkpoint(args.repo_path, model, optimizer)

    if args.dist == 'ddp':
        dist.destroy_process_group()

def save_checkpoint(path, model, optimizer, cur_iter=None, is_best=False):
    if cur_iter is None:
        fname = os.path.join(path, 'checkpoint.pth.tar')
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

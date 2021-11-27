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
from sklearn import decomposition
import h5py
import joblib
def getPCAfeatures(query_features,ref_features):
    X=np.concatenate([query_features, ref_features])
    pca = joblib.load('pca_fromtrain.joblib')
    X=pca.transform(X)
    query_features=X[:len(query_features)]
    ref_features=X[len(query_features):]
    return query_features,ref_features

def evalPCA_total(query_features,ref_features):
    q,r = getPCAfeatures(query_features,ref_features)
    q = nn.functional.normalize(torch.tensor(q),dim=1,p=2).cpu().numpy()
    r = nn.functional.normalize(torch.tensor(r),dim=1,p=2).cpu().numpy()
    query_id_list = np.array([x[:-4] for x in list(np.load('/facebook/data/images/query_total_imlist.npy'))])
    ref_truth_id_list = np.array([x[:-4] for x in list(np.load('/facebook/data/images/ref_imlist.npy'))])
    gt_df=pd.read_csv('/facebook/data/public_ground_truth.csv')
    submission_df = get_matching_from_descs(q,r, query_id_list, ref_truth_id_list, gt_df)
    ap,rp90 = evaluate_metrics(submission_df, gt_df)
    print(ap,rp90)

def eval_total(query_features,ref_features):
    query_id_list = np.array([x[:-4] for x in list(np.load('/facebook/data/images/query_total_imlist.npy'))])
    ref_truth_id_list = np.array([x[:-4] for x in list(np.load('/facebook/data/images/ref_imlist.npy'))])
    gt_df=pd.read_csv('/facebook/data/public_ground_truth.csv')
    submission_df = get_matching_from_descs(query_features, ref_features, query_id_list, ref_truth_id_list, gt_df)
    ap,rp90 = evaluate_metrics(submission_df, gt_df)
    print(ap,rp90)

def make_PCA_h5(query_total_features, ref_features,filename):
    query_total_features, ref_features = getPCAfeatures(query_total_features, ref_features )
    query_total_features = nn.functional.normalize(torch.tensor(query_total_features),dim=1,p=2).cpu().numpy()
    ref_features = nn.functional.normalize(torch.tensor(ref_features),dim=1,p=2).cpu().numpy()
    qry_ids = [x[:-4] for x in list(np.load('/facebook/data/images/query_total_imlist.npy'))]
    ref_ids = [x[:-4] for x in list(np.load('/facebook/data/images/ref_imlist.npy'))]
    #qry_ids = ['Q' + str(x).zfill(5) for x in range(50000,100000)]
    #ref_ids = ['R' + str(x).zfill(6) for x in range(1_000_000)]
    out = filename
    with h5py.File(out,"w") as f:
        f.create_dataset("query", data=query_total_features)
        f.create_dataset("reference", data=ref_features)
        f.create_dataset('query_ids', data=qry_ids)
        f.create_dataset('reference_ids', data=ref_ids)

def make_PCA_h5_withgt(query_total_features,ref_features,filename):
    query_total_features, ref_features = getPCAfeatures(query_total_features, ref_features)
    query_total_features = nn.functional.normalize(torch.tensor(query_total_features),dim=1,p=2).cpu().numpy()
    ref_features = nn.functional.normalize(torch.tensor(ref_features),dim=1,p=2).cpu().numpy()
    gt = pd.read_csv('/facebook/data/public_ground_truth.csv')
    gt = gt[~gt.reference_id.isnull()]
    query_total_features[:25000,:]=0.0
    query_total_features[gt.index]= ref_features[np.array([int(x[1:]) for x in gt['reference_id']])]
    qry_ids = [x[:-4] for x in list(np.load('/facebook/data/images/query_total_imlist.npy'))]
    ref_ids = [x[:-4] for x in list(np.load('/facebook/data/images/ref_imlist.npy'))]
    #qry_ids = ['Q' + str(x).zfill(5) for x in range(50_000)]
    #ref_ids = ['R' + str(x).zfill(6) for x in range(1_000_000)]
    out=filename
    with h5py.File(out,"w") as f:
        f.create_dataset("query", data=query_total_features)
        f.create_dataset("reference", data=ref_features)
        f.create_dataset('query_ids', data=qry_ids)
        f.create_dataset('reference_ids', data=ref_ids)


def get_total_features(path):
    query_features=np.load(os.path.join(path,'query_total_features.npy'))
    ref_features = np.load(os.path.join(path,'ref_features.npy'))
    query_features = nn.functional.normalize(torch.tensor(query_features),dim=1,p=2).cpu().numpy()
    ref_features = nn.functional.normalize(torch.tensor(ref_features),dim=1,p=2).cpu().numpy()
    return query_features,ref_features


def get_total_features_tta(path):
    query_features=np.load(os.path.join(path,'query_total_features.npy'))
    ref_features=np.load(os.path.join(path,'ref_features.npy'))
    for i in range(7):
        query_features += np.load(os.path.join(path,f'query_total_features_tta{i}.npy'))
    query_features /= 8.0
    query_features = nn.functional.normalize(torch.tensor(query_features), dim=1,p=2).cpu().numpy()
    ref_features = nn.functional.normalize(torch.tensor(ref_features), dim=1,p=2).cpu().numpy()
    return query_features, ref_features

def makePCA(train_features):
    X=train_features
    pca = decomposition.PCA(n_components=256)
    pca.fit(X)
    joblib.dump(pca,'pca_fromtrain.joblib')

def get_train_features(path):
    train_features=np.load(os.path.join(path,'train_features.npy'))
    train_features = nn.functional.normalize(torch.tensor(train_features),dim=1,p=2).cpu().numpy()
    return train_features

p1='repo/0/10120830_v3_1_other_0927newbatch_queryref'
p2='repo/1/10120830_v3_1_other_0927newbatch_queryref_2'
p3='repo/2/queryref_finetune_1005_2'
p4='repo/3/swin_1006_1_queryref'
p5='repo/4/10100058_v3_1_queryref'
p6='repo/5/simclr-pytorch-1003-from0922-newbatch'
#p7='repo/6_from0/10120830_v3_1_other_0927newbatch_queryref'

t1= get_train_features(os.path.join(p1,'basic_query'))
t2= get_train_features(os.path.join(p2,'basic_query'))
t3= get_train_features(os.path.join(p3,'basic_query'))
t4= get_train_features(os.path.join(p4,'basic_query'))
t5= get_train_features(os.path.join(p5,'basic_query'))
t6= get_train_features(os.path.join(p6,'basic_query'))
#t7= get_train_features(os.path.join(p7,'basic_query'))
makePCA(np.concatenate([t1*3,t2*2,t3,t4,t5,t6*0.25],-1))

"""
q1,r1=get_total_features_tta(os.path.join(p1,'basic_query'))
q2,r2=get_total_features_tta(os.path.join(p2,'basic_query'))
q3,r3=get_total_features_tta(os.path.join(p3,'basic_query'))
q4,r4=get_total_features_tta(os.path.join(p4,'basic_query'))
q5,r5=get_total_features_tta(os.path.join(p5,'basic_query'))
q6,r6=get_total_features_tta(os.path.join(p6,'basic_query'))
q7,r7=get_total_features_tta(os.path.join(p7,'basic_query'))
"""

q1,r1=get_total_features_tta(os.path.join(p1,'basic_query'))
q2,r2=get_total_features_tta(os.path.join(p2,'basic_query'))
q3,r3=get_total_features(os.path.join(p3,'basic_query'))
q4,r4=get_total_features(os.path.join(p4,'basic_query'))
q5,r5=get_total_features(os.path.join(p5,'basic_query'))
q6,r6=get_total_features(os.path.join(p6,'basic_query'))
#q7,r7=get_total_features_tta(os.path.join(p7,'basic_query'))

#evalPCA_total(np.concatenate([q1*3,q2*2,q3,q4,q5,q6*0.25,q7*2],-1),np.concatenate([r1*3,r2*2,r3,r4,r5,r6*0.25,r7*2],-1))
make_PCA_h5(np.concatenate([q1*3,q2*2,q3,q4,q5,q6*0.25],-1),np.concatenate([r1*3,r2*2,r3,r4,r5,r6*0.25],-1),'submission.h5')
#make_PCA_h5_withgt(np.concatenate([q1*3,q2*2,q3,q4,q5,q6*0.25,q7*2],-1),np.concatenate([r1*3,r2*2,r3,r4,r5,r6*0.25,r7*2],-1), 'submission_withgt.h5')

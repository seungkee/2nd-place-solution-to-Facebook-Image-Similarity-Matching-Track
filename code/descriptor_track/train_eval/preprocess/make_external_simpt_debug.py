import torch
from tqdm import tqdm
import os
import numpy as np
train_features=torch.load('train_features.pt')
n = len(list(np.load('/facebook/data/images/train_imlist.npy')))
print(n)
os.makedirs('/siim/sim_pt_256', exist_ok=True)
for i in tqdm(range(n)):
    a=torch.mm(train_features[i:i+1],train_features.t())
    torch.save(torch.tensor(np.argpartition(np.array(a),-256)[0][-256:]),os.path.join('/siim/sim_pt_256',f'{i}_sim256.pt'))
for i in tqdm(range(n)):
    a=torch.mm(train_features[i:i+1],train_features.t())
    torch.save(torch.tensor(np.argpartition(np.array(a),-512)[0][-512:]),os.path.join('/siim/sim_pt',f'{i}_sim512.pt'))

os.makedirs('/storage1/sim_pt',exist_ok=True)
if n < 65746:
    for i in tqdm(range(n)):
        a=torch.mm(train_features[i:i+1],train_features.t())
        torch.save(torch.argsort(a,descending=True)[0][:300],os.path.join('/storage1/sim_pt', f'{i}_sim2000.pt'))
else:
    for i in tqdm(range(65746)):
        a=torch.mm(train_features[i:i+1],train_features.t())
        torch.save(torch.argsort(a,descending=True)[0][:300],os.path.join('/storage1/sim_pt', f'{i}_sim2000.pt'))
    for i in tqdm(range(65746,1000000)):
        a=torch.mm(train_features[i:i+1],train_features.t())
        torch.save(torch.tensor(np.argpartition(np.array(a),-24)[0][-24:]), os.path.join('/storage1/sim_pt',f'{i}_sim2000.pt'))

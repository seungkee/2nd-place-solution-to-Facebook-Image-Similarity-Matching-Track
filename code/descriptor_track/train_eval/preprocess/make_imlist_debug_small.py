import numpy as np
train_imlist = []
for i in range(1000):
    train_imlist.append(f'T{i:06d}/T{i:06d}.jpg')
ref_imlist = []
for i in range(1000):
    ref_imlist.append(f'R{i:06d}.jpg')
query_total_imlist=[]
for i in range(50):
    query_total_imlist.append(f'Q{i:05d}.jpg')

np.save('/facebook/data/images/train_imlist.npy', train_imlist)
np.save('/facebook/data/images/ref_imlist.npy', ref_imlist)
np.save('/facebook/data/images/query_total_imlist.npy', query_total_imlist)

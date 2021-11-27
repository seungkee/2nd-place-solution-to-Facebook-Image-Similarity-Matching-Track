import numpy as np
import sys
import pandas as pd
def readhalfcsv(path):
    df=pd.read_csv(path)
    qn=len(list(np.load('/facebook/data/images/query_total_imlist.npy')))
    if 'half' in path:
        d={0:'left', 1:'vertical_center', 2:'right'}
        df['mode']=[d[x] for x in np.repeat(np.repeat(np.arange(9),qn),10)%3]
    elif '2of6' in path:
        d={0:'left',1:'left',2:'vertical_center',3:'right',4:'right'}
        df['mode']=[d[x] for x in np.repeat(np.repeat(np.arange(25),qn),5)%5]
    return df

df=readhalfcsv(sys.argv[1])
print(sys.argv[1])
for i in range(len(sys.argv)-2):
    path=sys.argv[i+2]
    print(path)
    new_df = readhalfcsv(path)
    df=df.append(new_df)
    df=df.drop_duplicates(subset=['query_id','reference_id','mode'],keep='last').reset_index(drop=True)

#df=df[:5000].copy()
df.to_csv(f'half_merge.csv',index=False)

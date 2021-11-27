import pandas as pd
import numpy as np
from tqdm import tqdm
p1='repo/0/matching-1009-from-1008-from-1001-nochange'
p2='repo/1/matching-1009-from-1008-from-1001-nochange'
p3='repo/2/matching-1009-from-1008-from-1001-nochange_000002'

df1_1=pd.read_csv(p1+'/basic_merge.csv_withmatchscore.csv')
df1_2=pd.read_csv(p1+'/half_merge.csv_withhalfmatchscore.csv')
df2_1=pd.read_csv(p2+'/basic_merge.csv_withmatchscore.csv')
df2_2=pd.read_csv(p2+'/half_merge.csv_withhalfmatchscore.csv')
df3_1=pd.read_csv(p3+'/basic_merge.csv_withmatchscore.csv')
df3_2=pd.read_csv(p3+'/half_merge.csv_withhalfmatchscore.csv')

df1_1['score']=(np.array(df1_1['score'])+np.array(df2_1['score'])+np.array(df3_1['score']))/3.0
df1_2['score']=(np.array(df1_2['score'])+np.array(df2_2['score'])+np.array(df3_2['score']))/3.0

df=df1_1[['query_id','reference_id','score']].append(df1_2[['query_id','reference_id','score']])

df=df.drop_duplicates(subset=['query_id','reference_id','score'],keep='last').reset_index(drop=True)

#idx = df.groupby(['query_id'])['score'].transform(max) == df['score']
#df = df[idx].reset_index(drop=True)

idx = df.groupby(['query_id','reference_id'])['score'].transform(max) == df['score']
df = df[idx].reset_index(drop=True)

query_ids = np.array(df['query_id'])
ref_ids = np.array(df['reference_id'])
scores = np.array(df['score'])
candPerQuery={}
for i in tqdm(range(len(ref_ids))):
    if query_ids[i] in candPerQuery:
        candPerQuery[query_ids[i]].append((ref_ids[i],scores[i]))
    else:
        candPerQuery[query_ids[i]]=[]
        candPerQuery[query_ids[i]].append((ref_ids[i],scores[i]))
for _key in candPerQuery.keys():
    candPerQuery[_key]=sorted(candPerQuery[_key], key=lambda tup:tup[1])

new_query_ids=[]
new_ref_ids=[]
n=100
for _key in candPerQuery.keys():
    new_query_ids+=[_key]*min(n,len(candPerQuery[_key]))
    new_ref_ids+=[x[0] for x in candPerQuery[_key][-n:]]
pd.DataFrame({'query_id':new_query_ids,'reference_id':new_ref_ids}).to_csv(f'final_cand_n.csv',index=False)

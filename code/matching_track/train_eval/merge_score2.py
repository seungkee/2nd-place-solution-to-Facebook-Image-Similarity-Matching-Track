import pandas as pd
import numpy as np
p1='repo/0/matching-1009-from-1008-from-1001-nochange'
p2='repo/1/matching-1009-from-1008-from-1001-nochange'
p3='repo/2/matching-1009-from-1008-from-1001-nochange_000002'

df1_1=pd.read_csv(p1+'/final_cand_n.csv_halfeval.csv')
df2_1=pd.read_csv(p2+'/final_cand_n.csv_halfeval.csv')
df3_1=pd.read_csv(p3+'/final_cand_n.csv_halfeval.csv')

df1_1['score']=(np.array(df1_1['score'])+np.array(df2_1['score'])+np.array(df3_1['score']))/3.0

df=df1_1.drop_duplicates(subset=['query_id','reference_id','score'],keep='last').reset_index(drop=True)

idx = df.groupby(['query_id'])['score'].transform(max) == df['score']
df = df[idx].reset_index(drop=True)

df.to_csv(f'final.csv',index=False)

import numpy as np
import sys
import pandas as pd
df=pd.read_csv(sys.argv[1])
print(sys.argv[1])
for i in range(len(sys.argv)-2):
    path=sys.argv[i+2]
    print(path)
    new_df = pd.read_csv(path)
    df=df.append(new_df)
    df=df.drop_duplicates(subset=['query_id','reference_id'],keep='last').reset_index(drop=True)
#df=df.sample(frac=0.01,replace=False,random_state=1)
#df=df[:10000].copy()
df.to_csv(f'basic_merge.csv', index=False)


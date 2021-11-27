import pandas as pd
q=['Q00000']*25000
r=['R000000']*25000
pd.DataFrame({'query_id':q,'reference_id':r}).to_csv('/facebook/data/public_ground_truth.csv',index=False)

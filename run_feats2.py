import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from feats import preprocess
from feats2 import *
D='signal-hunters-round2/'
for split,mc in (('test',False),('train',True)):
    X=np.load(f'{D}{split}_cadences.npy',mmap_mode='r'); m=pd.read_csv(f'{D}{split}_meta.csv')
    out=[]
    for s in range(0,len(X),500):
        z,_=preprocess(np.array(X[s:s+500]),mc)
        M,n=track_features(z,m.slew_gap_rows.values[s:s+500]); out.append(M)
        if s%5000==0: print(split,s,flush=True)
    np.save(f'F2_{split}.npy',np.concatenate(out)); pd.Series(n).to_csv('names2.csv',index=False)
    print(split,'done',flush=True)

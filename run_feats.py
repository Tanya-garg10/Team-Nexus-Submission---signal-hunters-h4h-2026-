import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from feats import *
D='signal-hunters-round2/'
for split,mc in (('train',True),('test',False)):
    X=np.load(f'{D}{split}_cadences.npy',mmap_mode='r'); m=pd.read_csv(f'{D}{split}_meta.csv')
    out=[]
    for s in range(0,len(X),1000):
        M,n=cadence_features(np.array(X[s:s+1000]),m.slew_gap_rows.values[s:s+1000],mask_center=mc)
        out.append(M)
    np.save(f'F_{split}.npy',np.concatenate(out)); 
    pd.Series(n).to_csv('names.csv',index=False)
    print(split,'done',flush=True)

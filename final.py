import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
D='signal-hunters-round2/'
n1=list(pd.read_csv('names.csv').iloc[:,0]);n2=['v2_'+s for s in pd.read_csv('names2.csv').iloc[:,0]]
A=np.hstack([np.load('F_train.npy'),np.load('F2_train.npy')]);B=np.hstack([np.load('F_test.npy'),np.load('F2_test.npy')]);names=n1+n2
y=pd.read_csv(D+'train_meta.csv').label.values;test=pd.read_csv(D+'test_meta.csv')
rng=np.random.default_rng(0);ia=rng.choice(len(A),4000,replace=False);ib=rng.choice(len(B),4000,replace=False)
au=np.array([abs(roc_auc_score(np.r_[np.zeros(4000),np.ones(4000)],np.nan_to_num(np.r_[A[ia,j],B[ib,j]]))-.5)+.5 for j in range(A.shape[1])])
keep=[i for i,n in enumerate(names) if not (n.endswith('_c') or n in('on_track_c','off_track_c','v2_on_best_c')) and au[i]<=0.62]
print('features kept',len(keep))
P=0;classes=None
for seed in range(4):
    m=H(max_iter=300,learning_rate=.06,class_weight='balanced',random_state=seed,max_leaf_nodes=31).fit(A[:,keep],y)
    P=P+m.predict_proba(B[:,keep])/4;classes=m.classes_
# label-prior shift (EM, Saerens et al.) relative to the effective uniform prior of the balanced model
pri=np.ones(len(classes))/len(classes);q=pri.copy()
for _ in range(50):
    w=P*(q/pri);w/=w.sum(1,keepdims=True);q=w.mean(0)
Padj=P*(q/pri);Padj/=Padj.sum(1,keepdims=True)
print(pd.Series(q,index=classes).round(3).to_dict())
pred=classes[Padj.argmax(1)]
sub=pd.DataFrame({'id':test.id,'label':pred});sub.to_csv('/mnt/user-data/outputs/submission.csv',index=False)
s=pd.read_csv('/mnt/user-data/outputs/submission.csv');ss=pd.read_csv(D+'sample_submission.csv')
assert len(s)==10000 and s.id.is_unique and set(s.id)==set(ss.id) and s.label.isin(set(y)).all()
print(s.label.value_counts(normalize=True).round(3))

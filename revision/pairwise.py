"""95% CIs for pairwise RMSE differences in P1R1 (30-day moving-block bootstrap, 2000 resamples,
same resampled days for both models), plus seed-to-seed spread of each difference."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, json
d=np.load('results/main/P1R1/preds.npz',allow_pickle=True); y=d['obs_test']; n=len(y)
M=['MLR','XGBoost','LSTM','BiLSTM','Attention-BiLSTM','TCN','Attention-TCN','Transformer']
P={m:d[f'{m}_test'].mean(0) for m in M}
rng=np.random.default_rng(0); B,bl=2000,30; nb=int(np.ceil(n/bl))
IDX=[(st[:,None]+np.arange(bl)).ravel()[:n] for st in (rng.integers(0,n-bl+1,nb) for _ in range(B))]
rm=lambda p,i: np.sqrt(np.mean((p[i]-y[i])**2))
def ci(a,b):
    diff=np.array([rm(P[a],i)-rm(P[b],i) for i in IDX]); return np.percentile(diff,[2.5,97.5])
rows=[]
for ref in ['BiLSTM','XGBoost']:
    for m in M:
        if m==ref: continue
        dpt=rm(P[m],np.arange(n))-rm(P[ref],np.arange(n)); lo,hi=ci(m,ref)
        sd=d[f'{m}_test']; sr=d[f'{ref}_test']
        seedd=[np.sqrt(np.mean((a-y)**2))-np.sqrt(np.mean((b-y)**2)) for a in sd for b in sr]
        rows.append(dict(ref=ref,model=m,diff=dpt,lo=lo,hi=hi,rel=dpt/rm(P[ref],np.arange(n)),
                         seed_diff_mean=np.mean(seedd),seed_diff_sd=np.std(seedd)))
T=pd.DataFrame(rows); print(T.round(4).to_string()); T.to_csv('results/revision/pairwise_ci.csv',index=False)
# seed SD of each model's RMSE
print({m:round(float(np.std([np.sqrt(np.mean((p-y)**2)) for p in d[f'{m}_test']])),4) for m in M})

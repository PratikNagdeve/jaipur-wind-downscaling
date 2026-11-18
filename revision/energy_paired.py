import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, json, os
from scipy import stats
exec(open('revision/energy.py').read().split("obs_v3=np.mean")[0])
obs_v3=np.mean(y**3)
rng=np.random.default_rng(1); B=300; bl=30; nb=int(np.ceil(n/bl))
IDX=[(s_[:,None]+np.arange(bl)).ravel()[:n] for s_ in (rng.integers(0,n-bl+1,nb) for _ in range(B))]
ko=[weib(y[i])[0] for i in IDX]; eo=[np.mean(y[i]**3)/np.mean(y[i])**3 for i in IDX]
rows=[]
for name,V in S.items():
    if name.startswith('Station'): continue
    dk=[weib(V[i])[0]-ko[j] for j,i in enumerate(IDX)]
    de=[np.mean(np.clip(V[i],0,None)**3)/np.mean(np.clip(V[i],0,None))**3-eo[j] for j,i in enumerate(IDX)]
    p=V.mean(1) if V.shape[1]>1 else V[:,0]
    rows.append(dict(series=name,rmse=np.sqrt(np.mean((p-y)**2)) if V.shape[1]==1 else np.nan,
       dk=weib(V)[0]-weib(y)[0],dk_lo=np.percentile(dk,2.5),dk_hi=np.percentile(dk,97.5),
       depf_lo=np.percentile(de,2.5),depf_hi=np.percentile(de,97.5)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('results/revision/energy_paired.csv',index=False)

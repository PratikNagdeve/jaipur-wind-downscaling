"""Energy-relevant distribution: bootstrap CIs (30-day blocks, 500 resamples) for Weibull k, c,
mean v^3 ratio and energy pattern factor EPF = <v^3>/<v>^3; deterministic post-processing
competitors (variance inflation, quantile mapping, both fitted on validation years);
tail-extension sensitivity of member sampling. Optional ERA5 rows if era5 file given."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, json, sys, os
from scipy import stats
O='results/main'; d=np.load(f'{O}/P1R1/preds.npz',allow_pickle=True); y=d['obs_test']; yv=d['obs_val']; n=len(y)
d0=np.load(f'{O}/P0R0/preds.npz',allow_pickle=True)
Z=dict(np.load(f'{O}/prob/quantiles.npz',allow_pickle=True)); QS=Z['QS']
S={}
S['Station (observed)']=y[:,None]
extra=json.load(open('results/revision/era5_series.json')) if os.path.exists('results/revision/era5_series.json') else {}
for k,v in extra.items(): S[k]=np.array(v)[:,None]
S['XGBoost, original set-up']=d0['XGBoost_test'].mean(0)[:,None]; S['TCN, original set-up']=d0['TCN_test'].mean(0)[:,None]
for m in ['XGBoost','TCN']:
    p=d[f'{m}_test'].mean(0); pv=d[f'{m}_val'].mean(0); S[f'{m}, final set-up']=p[:,None]
    # variance inflation (validation-fitted): keep mean, scale anomalies so validation SD matches observed
    a=yv.std()/pv.std(); S[f'{m} + variance inflation']=np.clip(pv.mean()+a*(p-pv.mean())+0*(yv.mean()-pv.mean()),0,None)[:,None]
    # empirical quantile mapping of output, fitted on validation years
    q=np.linspace(0.005,0.995,199); qp,qo=np.quantile(pv,q),np.quantile(yv,q)
    S[f'{m} + quantile mapping']=np.interp(p,qp,qo,left=None,right=None)[:,None]
def members(Q,mode,rng):
    grid=np.r_[0,QS,1]
    if mode=='none': lo,hi=Q[:,:1],Q[:,-1:]
    else:
        f={'linear':1.0,'half':0.5,'double':2.0}[mode]
        lo=np.maximum(0,Q[:,:1]-f*(Q[:,1:2]-Q[:,:1])); hi=Q[:,-1:]+f*(Q[:,-1:]-Q[:,-2:-1])
    Qe=np.c_[lo,Q,hi]; u=rng.uniform(size=(len(Q),50))
    return np.array([np.interp(u[i],grid,Qe[i]) for i in range(len(Q))])
TAIL={}
for nm,key in [('Quantile XGBoost','Quantile XGBoost|test'),('Quantile TCN','Quantile TCN|test')]:
    for mode in ['linear','none','half','double']:
        M=members(Z[key],mode,np.random.default_rng(0))
        if mode=='linear': S[f'{nm}, 50 sampled members']=M
        TAIL[f'{nm}|{mode}']=M
def weib(v):
    v=np.ravel(v); v=v[v>0.05]; k,_,c=stats.weibull_min.fit(v,floc=0); return k,c
def st(V):
    v=np.ravel(V); k,c=weib(v); v3=np.mean(np.clip(v,0,None)**3)
    return dict(mean=v.mean(),sd=v.std(),p99=np.percentile(v,99),k=k,c=c,v3=v3,epf=v3/np.mean(np.clip(v,0,None))**3)
obs_v3=np.mean(y**3)
rng=np.random.default_rng(1); B,bl=int(os.environ.get('B',500)),30; nb=int(np.ceil(n/bl))
IDX=[(st_[:,None]+np.arange(bl)).ravel()[:n] for st_ in (rng.integers(0,n-bl+1,nb) for _ in range(B))]
rows=[]
for name,V in S.items():
    base=st(V); base['v3_ratio']=base['v3']/obs_v3
    bs=[]
    for i in IDX:
        s=st(V[i]); s['v3_ratio']=s['v3']/np.mean(y[i]**3); bs.append(s)
    bs=pd.DataFrame(bs)
    r=dict(series=name,**base)
    for k_ in ['k','c','v3_ratio','epf','sd','mean']: r[k_+'_lo'],r[k_+'_hi']=np.percentile(bs[k_],[2.5,97.5])
    rows.append(r); print(name, {k:round(v,3) for k,v in r.items() if k!='series'},flush=True)
T=pd.DataFrame(rows); T.to_csv('results/revision/energy_ci.csv',index=False)
tr=[]
for k,V in TAIL.items():
    s=st(V); tr.append(dict(case=k,**s,v3_ratio=s['v3']/obs_v3))
TT=pd.DataFrame(tr); print(TT.round(3).to_string()); TT.to_csv('results/revision/tail_sensitivity.csv',index=False)

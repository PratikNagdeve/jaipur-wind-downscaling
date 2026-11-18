"""Conformal variants for the 90% interval: CQR on all validation years (paper), CQR on 2017 only,
and adaptive conformal inference (ACI; Gibbs and Candes 2021) updated online through the test period."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, json
Z=dict(np.load('results/main/prob/quantiles.npz',allow_pickle=True)); y=Z['obs_test']; yv=Z['obs_val']
dt=pd.to_datetime(Z['test_dates']); n=len(y); alpha=0.1
d=np.load('results/main/P1R1/preds.npz',allow_pickle=True)
# validation dates: reconstruct from test_idx not available; val = 2015-2017 days with valid target in order
res={}
def summarize(lo,hi,lab,name):
    c=(y>=lo)&(y<=hi); r=dict(cov=c.mean(),width=np.mean(hi-lo))
    for yr in [2018,2019,2020]: r[f'cov_{yr}']=c[dt.year==yr].mean()
    m=(dt>='2020-07-01')&(dt<'2020-11-01'); r['cov_JulOct2020']=c[m].mean(); r['cov_excl']=c[~m].mean()
    # rolling 90-day coverage range
    s=pd.Series(c.astype(float)).rolling(90).mean().dropna(); r['roll90_min']=s.min(); r['roll90_max']=s.max()
    res[f'{name}|{lab}']=r
for name in ['Quantile XGBoost','Quantile TCN']:
    Qv,Qt=Z[f'{name}|val'],Z[f'{name}|test']
    sv=np.maximum(Qv[:,0]-yv,yv-Qv[:,-1])
    summarize(Qt[:,0],Qt[:,-1],'raw',name)
    q=np.quantile(sv,min(1,(1-alpha)*(1+1/len(sv)))); summarize(Qt[:,0]-q,Qt[:,-1]+q,'CQR 2015-2017',name)
    # last validation year = last 365 valid samples of validation set (validation is chronological)
    s17=sv[-365:]; q=np.quantile(s17,min(1,(1-alpha)*(1+1/len(s17)))); summarize(Qt[:,0]-q,Qt[:,-1]+q,'CQR 2017 only',name)
    st=np.maximum(Qt[:,0]-y,y-Qt[:,-1])
    for g in [0.005,0.01]:
        a=alpha; scores=list(sv); lo=np.empty(n); hi=np.empty(n)
        for t in range(n):
            lev=np.clip(1-a,0,1); q=np.quantile(scores,lev) if lev<1 else np.max(scores)+1
            lo[t],hi[t]=Qt[t,0]-q,Qt[t,-1]+q
            err=float(not(lo[t]<=y[t]<=hi[t])); a=a+g*(alpha-err); scores.append(st[t])
        summarize(lo,hi,f'ACI gamma={g}',name)
R=pd.DataFrame(res).T; print(R.round(3).to_string()); R.to_csv('results/revision/conformal.csv')

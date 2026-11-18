"""Homogeneity re-tests: (i) full 2005-2020 vs training years 2005-2014 only; (ii) significance under
AR(1) autocorrelation (Monte Carlo with AR(1) noise, phi estimated after removing the detected step);
(iii) second reference series (IMDAA; ERA5 vector-mean); reference = ERA5 hourly-mean if available."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, json, sys, os
A='data/raw/isd/'
E=pd.read_csv('data/era5_daily_ext.csv',parse_dates=['date']); E['era5_vec']=E.ws_vec; E=E.rename(columns={'imdaa_average_speed':'imdaa'})
S=pd.read_csv('data/jaipur_daily_wind.csv',parse_dates=['date']).rename(columns={'wind_speed_ms':'obs'})
D=E.merge(S,on='date',how='left'); D=D[D.date.dt.year>=2005]
def series(ref,end='2020-12-31'):
    d=D[D.date<=end].set_index('date'); M=d[['obs',ref]].resample('MS').mean(); cnt=d.obs.resample('MS').count()
    M=M[cnt>=20].dropna(); r=np.log(M.obs/M[ref]); r=r-r.groupby(r.index.month).transform('mean'); return r
def pettitt(x):
    n=len(x); U=np.array([np.sum(np.sign(x[t+1:,None]-x[None,:t+1])) for t in range(n-1)]); K=np.abs(U).max(); t=int(np.abs(U).argmax())
    return t,K,min(1,2*np.exp(-6*K**2/(n**3+n**2)))
def snht(x):
    z=(x-x.mean())/x.std(); n=len(z); T=np.array([k*z[:k].mean()**2+(n-k)*z[k:].mean()**2 for k in range(1,n)]); return int(T.argmax())+1,T.max()
def ar1(n,phi,rng):
    e=rng.standard_normal(n+50); x=np.zeros(n+50)
    for i in range(1,n+50): x[i]=phi*x[i-1]+e[i]
    return x[50:]
out={}
refs=['ws_scalar','era5_vec','imdaa'] if 'ws_scalar' in D else ['era5_vec','imdaa']
for ref in refs:
    for lab,end in [('2005-2020','2020-12-31'),('2005-2014 (training)','2014-12-31')]:
        r=series(ref,end); x=r.values; n=len(x); idx=r.index
        tp,K,pp=pettitt(x); ks,T0=snht(x)
        # phi from residuals after removing the step at the SNHT break
        res=x.copy(); res[:ks]-=x[:ks].mean(); res[ks:]-=x[ks:].mean(); phi=float(np.corrcoef(res[:-1],res[1:])[0,1])
        phi_raw=float(np.corrcoef(x[:-1],x[1:])[0,1])
        rng=np.random.default_rng(0); N=2000
        Tn=np.array([snht(rng.standard_normal(n))[1] for _ in range(N)])
        Ta=[];Ka=[]
        for _ in range(N):
            z=ar1(n,max(phi,0),rng); Ta.append(snht(z)[1]); Ka.append(pettitt(z)[1])
        Ta,Ka=np.array(Ta),np.array(Ka)
        rat=np.exp(np.log(D.set_index('date').obs.resample('MS').mean()/D.set_index('date')[ref].resample('MS').mean()))
        before=x[:ks].mean(); after=x[ks:].mean()
        out[f'{ref}|{lab}']=dict(n_months=n,pettitt_after=str(idx[tp].date()),pettitt_p_asym=pp,
            pettitt_p_ar1=float((np.sum(Ka>=K)+1)/(N+1)),snht_at=str(idx[ks].date()),snht_T=float(T0),
            snht_p_white=float((np.sum(Tn>=T0)+1)/(N+1)),snht_p_ar1=float((np.sum(Ta>=T0)+1)/(N+1)),
            phi_resid=phi,phi_raw=phi_raw,step_pct=float(100*(np.exp(after-before)-1)))
        print(ref,lab,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[f'{ref}|{lab}'].items()},flush=True)
# mean ratio before/after Feb 2011 per reference and Aug 2020
for ref in refs:
    d=D.set_index('date'); M=d[['obs',ref]].resample('MS').mean(); cnt=d.obs.resample('MS').count(); M=M[cnt>=20].dropna(); q=M.obs/M[ref]
    a=q[q.index<'2011-02-01'].mean(); b=q[q.index>='2011-02-01'].mean(); aug=q[q.index.month==8]
    out[f'{ref}|ratio']=dict(before=a,after=b,change_pct=100*(b/a-1),aug2020=float(aug.get(pd.Timestamp('2020-08-01'),np.nan)),aug_other_min=float(aug[aug.index.year!=2020].min()),aug_other_max=float(aug[aug.index.year!=2020].max()))
    print(ref,{k:round(v,3) for k,v in out[f'{ref}|ratio'].items()})
json.dump(out,open('results/revision/homog2.json','w'),indent=1)

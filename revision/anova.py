"""Factorial ANOVA of test RMSE: predictor set (P) x regime (R) x model (M); cell-mean and seed-level."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, json, itertools
O='results/main'; NL=['XGBoost','LSTM','BiLSTM','Attention-BiLSTM','TCN','Attention-TCN','Transformer']
rows=[]
for c in ['P0R0','P0R1','P1R0','P1R1']:
    d=np.load(f'{O}/{c}/preds.npz',allow_pickle=True); y=d['obs_test']
    for m in NL+['MLR']:
        for s,p in enumerate(d[f'{m}_test']):
            rows.append(dict(P=c[:2],R=c[2:],M=m,seed=s,RMSE=float(np.sqrt(np.mean((p-y)**2)))))
L=pd.DataFrame(rows); L.to_csv('results/revision/rmse_seed_level.csv',index=False)
def anova(df, factors):
    gm=df.RMSE.mean(); SST=((df.RMSE-gm)**2).sum(); res={}
    # balanced design: SS of effect via inclusion-exclusion of cell means
    def ss_marg(fs):
        if not fs: return 0.0
        g=df.groupby(list(fs)).RMSE; return float((g.transform('mean')-gm).pow(2).sum())
    eff={}
    for k in range(1,len(factors)+1):
        for fs in itertools.combinations(factors,k):
            s=ss_marg(fs)
            for j in range(1,k):
                for sub in itertools.combinations(fs,j): s-=eff[sub]
            eff[fs]=s
    out={' x '.join(k):v/SST for k,v in eff.items()}
    out['residual (seed)']=1-sum(out.values()); return out,SST
cell=L[L.M!='MLR'].groupby(['P','R','M']).RMSE.mean().reset_index()
a1,_=anova(cell,['P','R','M'])
a2,_=anova(L[L.M!='MLR'],['P','R','M'])
print('cell means (28):'); [print(f'  {k:12s} {100*v:6.2f}%') for k,v in a1.items()]
print('seed level (280):'); [print(f'  {k:12s} {100*v:6.2f}%') for k,v in a2.items()]
# F tests at seed level
import statsmodels.formula.api as smf, statsmodels.api as sm
mod=smf.ols('RMSE ~ C(P)*C(R)*C(M)',data=L[L.M!='MLR']).fit(); tab=sm.stats.anova_lm(mod,typ=2); tab['share']=tab.sum_sq/tab.sum_sq.sum(); print(tab.round(5))
tab.to_csv('results/revision/anova_seed_level.csv')
# alternative contrast: with MLR included (8 models, cell means)
a3,_=anova(L.groupby(['P','R','M']).RMSE.mean().reset_index(),['P','R','M']); print('with MLR, cells:',{k:round(100*v,2) for k,v in a3.items()})
# P effect restricted: only R1 configs
json.dump(dict(cell=a1,seed=a2,with_mlr=a3),open('results/revision/anova.json','w'),indent=1)

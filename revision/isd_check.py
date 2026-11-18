"""ISD reporting frequency, report types, wind-speed resolution and calm share per year (VIJP)."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import glob, numpy as np, pandas as pd, sys, json
A='data/raw/isd/'
fs=sorted(glob.glob(A+'*.csv'))
rows=[]; allobs=[]
for f in fs:
    df=pd.read_csv(f,usecols=['DATE','REPORT_TYPE','SOURCE','WND'],dtype=str)
    w=df.WND.str.split(',',expand=True)
    df['spd']=pd.to_numeric(w[3],errors='coerce')/10; df['qc']=w[4]; df['type']=w[2]
    df['t']=pd.to_datetime(df.DATE); df['rt']=df.REPORT_TYPE.str.strip()
    allobs.append(df)
D=pd.concat(allobs); D['year']=D.t.dt.year
ok=D[(D.spd<999)&D.qc.isin(list('01459'))]
syn=ok[(ok.t.dt.minute==0)&(ok.t.dt.hour%3==0)]
for y,g in D.groupby('year'):
    o=ok[ok.year==y]; s=syn[syn.year==y]
    s1=s.assign(pref=(s.rt!='FM-12').astype(int)).sort_values(['t','pref']).drop_duplicates('t')
    nz=s1.spd[s1.spd>0]
    vals=np.round(nz.values,1)
    kn=np.round(np.arange(1,60)*0.5144,1)
    rows.append(dict(year=y, n_reports=len(g), types=dict(g.rt.value_counts().head(3)),
        frac_FM12_in_used=float((s1.rt=='FM-12').mean()) if len(s1) else np.nan,
        hours_per_day=len(s1)/max(1,s1.t.dt.floor('D').nunique()),
        days=s1.t.dt.floor('D').nunique(),
        calm_share=float((s1.spd==0).mean()) if len(s1) else np.nan,
        n_distinct=len(np.unique(vals)), top_values=list(pd.Series(vals).value_counts().head(5).index),
        frac_on_knot_grid=float(np.isin(vals,kn).mean()) if len(vals) else np.nan,
        frac_integer_ms=float(np.isclose(vals%1,0).mean()) if len(vals) else np.nan,
        mean_used=float(s1.spd.mean()) if len(s1) else np.nan))
R=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(R.to_string())
R.to_csv('results/revision/isd_check.csv',index=False)

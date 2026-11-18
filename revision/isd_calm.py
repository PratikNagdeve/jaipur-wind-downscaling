import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import glob, numpy as np, pandas as pd
A='data/raw/isd/'
L=[]
for f in sorted(glob.glob(A+'*.csv')):
    df=pd.read_csv(f,usecols=['DATE','REPORT_TYPE','WND'],dtype=str); w=df.WND.str.split(',',expand=True)
    df['spd']=pd.to_numeric(w[3],errors='coerce')/10; df['t']=pd.to_datetime(df.DATE)
    df=df[(df.spd<999)&w[4].isin(list('01459'))&(df.t.dt.minute==0)&(df.t.dt.hour%3==0)]
    df=df.assign(pref=(df.REPORT_TYPE.str.strip()!='FM-12').astype(int)).sort_values(['t','pref']).drop_duplicates('t'); L.append(df[['t','spd']])
D=pd.concat(L); D['y']=D.t.dt.year
g=D.groupby('y').spd
out=pd.DataFrame({'mean':g.mean(),'calm':g.apply(lambda s:(s==0).mean()),'mean_noncalm':g.apply(lambda s:s[s>0].mean()),'min_noncalm':g.apply(lambda s:s[s>0].min())})
print(out.round(3).to_string())
for a,b,lab in [('2005-01-01','2011-02-01','2005-Jan2011'),('2011-02-01','2015-01-01','Feb2011-2014'),('2015-01-01','2021-01-01','2015-2020')]:
    s=D[(D.t>=a)&(D.t<b)].spd; print(lab, 'calm',round((s==0).mean(),3),'mean noncalm',round(s[s>0].mean(),3),'mean',round(s.mean(),3))
# hour-of-day calm share before/after
D['h']=D.t.dt.hour
pre=D[(D.t>='2005-01-01')&(D.t<'2011-02-01')]; post=D[(D.t>='2011-02-01')&(D.t<'2015-01-01')]
print(pd.DataFrame({'pre':pre.groupby('h').spd.apply(lambda s:(s==0).mean()),'post':post.groupby('h').spd.apply(lambda s:(s==0).mean())}).round(3).T)
D.to_pickle('results/revision/isd_3h.pkl')

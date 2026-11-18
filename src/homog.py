"""Homogeneity of the station record: Pettitt and SNHT (Alexandersson) on the monthly
deseasonalised station/ERA5 ratio (2005-2020), with Monte Carlo p-value for SNHT."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, json
D = pd.read_csv('data/era5_daily_ext.csv', parse_dates=['date']).merge(
    pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'obs'}), on='date', how='left')
D = D[D.date.dt.year >= 2005]
M = D.set_index('date')[['obs', 'ws_scalar']].resample('MS').mean()
cnt = D.set_index('date').obs.resample('MS').count()
M = M[cnt >= 20].dropna()
r = np.log(M.obs / M.ws_scalar)
r = r - r.groupby(r.index.month).transform('mean')
x = r.values; n = len(x)


def pettitt(x):
    n = len(x); U = np.array([np.sum(np.sign(x[t + 1:, None] - x[None, :t + 1])) for t in range(n - 1)])
    K = np.abs(U).max(); t = int(np.abs(U).argmax()); return t, K, min(1, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))


def snht(x):
    z = (x - x.mean()) / x.std(); n = len(z)
    T = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(1, n)])
    return int(T.argmax()) + 1, T.max()


tp, K, pp = pettitt(x); ks, T0 = snht(x)
rng = np.random.default_rng(0)
Tnull = np.array([snht(rng.standard_normal(n))[1] for _ in range(2000)])
ps = (np.sum(Tnull >= T0) + 1) / 2001
before, after = M.index[:tp + 1], M.index[tp + 1:]
ratio = M.obs / M.ws_scalar
res = dict(n_months=n, pettitt_change_after=str(M.index[tp].date()), pettitt_p=pp,
           snht_break_at=str(M.index[ks].date()), snht_T=float(T0), snht_p=float(ps),
           ratio_before=float(ratio[ratio.index < '2011-02-01'].mean()), ratio_after=float(ratio[ratio.index >= '2011-02-01'].mean()))
# August anomaly
aug = ratio[ratio.index.month == 8]
res['aug_ratio_2020'] = float(aug.loc['2020'].iloc[0]); res['aug_ratio_other_min'] = float(aug[aug.index.year != 2020].min()); res['aug_ratio_other_max'] = float(aug[aug.index.year != 2020].max())
json.dump(res, open('results/main/homogeneity.json', 'w'), indent=1); print(json.dumps(res, indent=1))

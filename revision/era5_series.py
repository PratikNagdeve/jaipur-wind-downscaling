"""ERA5 baseline series on the test days (vector-mean, hourly-mean, linearly corrected) for revision/energy.py."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, numpy as np, pandas as pd
D = pd.read_csv('data/era5_daily_ext.csv', parse_dates=['date'])
S = pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'obs'})
M = D.merge(S, on='date', how='left'); tr = M[M.date.dt.year.between(2005, 2014)].dropna(subset=['obs'])
d = np.load('results/main/P1R1/preds.npz', allow_pickle=True); T = M.set_index('date').loc[pd.to_datetime(d['test_dates'])]
assert np.allclose(T.obs.values, d['obs_test'], atol=1e-4)
b, a = np.polyfit(tr.ws_scalar, tr.obs, 1)
json.dump({'Raw ERA5, vector-mean speed': T.ws_vec.tolist(), 'Raw ERA5, hourly-mean speed': T.ws_scalar.tolist(),
           'ERA5 + linear correction': (a + b * T.ws_scalar).tolist()}, open('results/revision/era5_series.json', 'w'))
print('saved results/revision/era5_series.json')

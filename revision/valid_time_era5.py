"""ERA5 daily speed averaged only at the hours of the valid station (ISD synoptic) reports of each day.
Writes data/era5_daily_ext_syn.csv (= era5_daily_ext.csv + ws_valid). Needs data/raw/isd/*.csv and the ERA5 hourly CSV.
usage: python revision/valid_time_era5.py <era5_hourly.csv>"""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import glob, numpy as np, pandas as pd
L = []
for f in sorted(glob.glob('data/raw/isd/*.csv')):
    df = pd.read_csv(f, usecols=['DATE', 'REPORT_TYPE', 'WND'], dtype=str); w = df.WND.str.split(',', expand=True)
    df['spd'] = pd.to_numeric(w[3], errors='coerce') / 10; df['t'] = pd.to_datetime(df.DATE)
    df = df[(df.spd < 999) & w[4].isin(list('01459')) & (df.t.dt.minute == 0) & (df.t.dt.hour % 3 == 0)]
    L.append(df[['t']].drop_duplicates())
V = pd.concat(L).rename(columns={'t': 'valid_time'})
H = pd.read_csv(_sys.argv[1], parse_dates=['valid_time']); H['ws'] = np.hypot(H.u10, H.v10)
M = H.merge(V, on='valid_time', how='inner'); M['date'] = M.valid_time.dt.floor('D')
syn = M.groupby('date').ws.mean().rename('ws_valid').reset_index()
E = pd.read_csv('data/era5_daily_ext.csv', parse_dates=['date']).merge(syn, on='date', how='left')
E['ws_valid'] = E.ws_valid.fillna(E.ws_syn)   # days without valid station reports: all eight synoptic hours
E.to_csv('data/era5_daily_ext_syn.csv', index=False); print('saved data/era5_daily_ext_syn.csv')

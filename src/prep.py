import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, sys
H = pd.read_csv(sys.argv[1], parse_dates=['valid_time'])
H['ws'] = np.hypot(H.u10, H.v10)
H['date'] = H.valid_time.dt.floor('D'); H['hr'] = H.valid_time.dt.hour
g = H.groupby('date')
D = g[['u10','v10','d2m','t2m','blh','msl','sp','tp','imdaa_average_speed']].mean()
D['ws_vec'] = np.hypot(D.u10, D.v10)              # paper's definition
D['ws_scalar'] = g.ws.mean()                       # mean of hourly speed
syn = H[H.hr % 3 == 0].groupby('date').ws.mean(); D['ws_syn'] = syn
D['ws_max'] = g.ws.max(); D['ws_sd'] = g.ws.std()
day = H[H.hr.between(4, 12)]; night = H[(H.hr >= 15) | (H.hr <= 1)]   # 09:30-17:30 IST / night
D['ws_day'] = day.groupby('date').ws.mean(); D['ws_night'] = night.groupby('date').ws.mean()
D['blh_max'] = g.blh.max(); D['blh_min'] = g.blh.min()
D['t2m_range'] = g.t2m.max() - g.t2m.min()
D['steady'] = D.ws_vec / D.ws_scalar
D['dir_sin'] = np.sin(np.arctan2(D.u10, D.v10)); D['dir_cos'] = np.cos(np.arctan2(D.u10, D.v10))
D['rh_proxy'] = D.t2m - D.d2m
D.reset_index().to_csv('data/era5_daily_ext.csv', index=False)
print(D.shape); print(D.head(3).T)

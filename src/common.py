"""Shared data loading, splits and metrics for the Jaipur wind experiments."""
import numpy as np
import pandas as pd

DATA = 'data/era5_imdaa_daily.csv'  # original-paper input (daily ERA5 + IMDAA)
STATION = 'data/jaipur_daily_wind.csv'
FEATURES = ['u10', 'v10', 'd2m', 't2m', 'blh', 'msl', 'sp', 'tp']
TARGET = 'imdaa_average_speed'
SPLITS = {'train': (2000, 2014), 'val': (2015, 2017), 'test': (2018, 2020)}
STATION_FIRST_GOOD_YEAR = 2005  # station series has a level shift before 2005


def load():
    d = pd.read_csv(DATA, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    d['era5_speed'] = np.hypot(d.u10, d.v10)  # speed of daily-mean wind vector
    s = pd.read_csv(STATION, parse_dates=['date'])
    d = d.merge(s.rename(columns={'wind_speed_ms': 'station'}), on='date', how='left')
    d.loc[d.date.dt.year < STATION_FIRST_GOOD_YEAR, 'station'] = np.nan
    d['split'] = None
    for k, (a, b) in SPLITS.items():
        d.loc[d.date.dt.year.between(a, b), 'split'] = k
    return d


def metrics(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = ~(np.isnan(pred) | np.isnan(obs))
    p, o = pred[ok], obs[ok]
    e = p - o
    return {'n': int(ok.sum()),
            'RMSE': np.sqrt(np.mean(e ** 2)),
            'MAE': np.mean(np.abs(e)),
            'Bias': np.mean(e),
            'r': np.corrcoef(p, o)[0, 1],
            'R2': 1 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2)}

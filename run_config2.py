"""Revision experiments, same protocol as run_config.py (L = 7, 30 random trials on validation RMSE,
10 seeds of the best setting, train 2005-2014 / val 2015-2017 / test 2018-2020).
usage: python run_config2.py TAG [models...]
TAG:
  A_WS      original 8 + hourly-mean speed WS                       + regime indicator
  B_SPD     original 8 + five speed predictors                       + regime indicator
  C_NOSPD   original 8 + physics predictors without the five speeds  + regime indicator
  D_POST    physics set, training restricted to Feb 2011-2014, no indicator
  E_RATIO   physics set, pre-break target scaled by the training-period ratio change, no indicator
  H_VALID   physics set with the hourly-mean speed replaced by ERA5 speed at valid station report times
            (needs ERA5EXT=data/era5_daily_ext_syn.csv from revision/valid_time_era5.py)
  F_PRE     keep 2000-2004 in training with a second (pre-2005) regime indicator
  G_PREADJ  keep 2000-2004 with the pre-2005 station values ratio-adjusted"""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, os, sys, time
sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, torch, xgboost as xgb
from sklearn.linear_model import LinearRegression
from lib import Data, SPACES, XGB_SPACE, sample, train_net, predict, n_params
torch.set_num_threads(1)
TAG = sys.argv[1]; names = sys.argv[2:] or ['MLR', 'XGBoost', 'TCN', 'BiLSTM']
OUT = os.environ.get('OUTDIR', 'results/revision/exp') + f'/{TAG}'; os.makedirs(OUT, exist_ok=True)
N_TRIALS, SEEDS, BREAK = int(os.environ.get('NT', 30)), list(range(int(os.environ.get('NS', 10)))), '2011-02-01'
BASE = ['u10', 'v10', 'd2m', 't2m', 'blh', 'msl', 'sp', 'tp']
SPD = ['ws_scalar', 'ws_max', 'ws_day', 'ws_night', 'ws_sd']
NOSPD = ['blh_max', 'blh_min', 't2m_range', 'steady', 'dir_sin', 'dir_cos', 'doy_sin', 'doy_cos']
PHYS = BASE + SPD + NOSPD
SPLIT = {'train': (2005, 2014), 'val': (2015, 2017), 'test': (2018, 2020)}
D = pd.read_csv(os.environ.get('ERA5EXT', 'data/era5_daily_ext.csv'), parse_dates=['date'])
S = pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'station'})
D = D.merge(S, on='date', how='left'); D.loc[D.date.dt.year < 2005, 'station'] = np.nan
doy = D.date.dt.dayofyear; D['doy_sin'], D['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
D['regime'] = (D.date >= BREAK).astype(float)
info = {}
if TAG == 'A_WS': feats = BASE + ['ws_scalar', 'regime']
elif TAG == 'B_SPD': feats = BASE + SPD + ['regime']
elif TAG == 'C_NOSPD': feats = BASE + NOSPD + ['regime']
elif TAG == 'D_POST':
    feats = PHYS; D.loc[D.date < BREAK, 'station'] = np.nan; SPLIT = dict(SPLIT, train=(2011, 2014))
elif TAG == 'E_RATIO':
    feats = PHYS
    tr = D[D.date.dt.year.between(2005, 2014)].set_index('date')
    M = tr[['station', 'ws_scalar']].resample('MS').mean(); cnt = tr.station.resample('MS').count(); M = M[cnt >= 20].dropna()
    q = M.station / M.ws_scalar; f = q[q.index >= BREAK].mean() / q[q.index < BREAK].mean()
    D.loc[D.date < BREAK, 'station'] *= f; info['ratio_factor'] = float(f)
elif TAG == 'H_VALID':
    feats = [('ws_valid' if f == 'ws_scalar' else f) for f in PHYS] + ['regime']
elif TAG in ('F_PRE', 'G_PREADJ'):
    S2 = pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'station'})
    D = D.drop(columns='station').merge(S2, on='date', how='left'); SPLIT = dict(SPLIT, train=(2000, 2014))
    if TAG == 'F_PRE':
        D['pre2005'] = (D.date < '2005-01-01').astype(float); feats = PHYS + ['regime', 'pre2005']
    else:
        tr = D[D.date.dt.year.between(2000, 2014)].set_index('date')
        M = tr[['station', 'ws_scalar']].resample('MS').mean(); cnt = tr.station.resample('MS').count(); M = M[cnt >= 20].dropna()
        q = M.station / M.ws_scalar; f = q[(q.index >= '2005-01-01') & (q.index < BREAK)].mean() / q[q.index < '2005-01-01'].mean()
        D.loc[D.date < '2005-01-01', 'station'] *= f; info['pre2005_factor'] = float(f); feats = PHYS + ['regime']
d = Data(7, 0, split=SPLIT, df=D, target='station', features=feats)
Xtr, ytr, _ = d.get('train'); Xva, yva, _ = d.get('val'); Xte, yte, tte = d.get('test')
flat = lambda X: X.reshape(len(X), -1)
hp_file, pf, cf = f'{OUT}/best_hp.json', f'{OUT}/preds.npz', f'{OUT}/cost.json'
best_hp = json.load(open(hp_file)) if os.path.exists(hp_file) else {}
preds = dict(np.load(pf, allow_pickle=True)) if os.path.exists(pf) else {}
cost = json.load(open(cf)) if os.path.exists(cf) else {}
def save(): json.dump(best_hp, open(hp_file, 'w'), indent=1); np.savez(pf, **preds); json.dump(dict(cost, **info, n_train=len(ytr), features=feats), open(cf, 'w'), indent=1)
preds['obs_test'] = d.to_ms(yte); preds['obs_val'] = d.to_ms(yva); preds['test_dates'] = d.df.date.values[tte].astype('datetime64[D]').astype(str)
print(TAG, len(feats), 'features; train/val/test', len(ytr), len(yva), len(yte), info, flush=True)
for name in names:
    if f'{name}_test' in preds: continue
    if name == 'MLR':
        m = LinearRegression().fit(flat(Xtr), ytr)
        preds['MLR_test'] = d.to_ms(m.predict(flat(Xte)))[None]; preds['MLR_val'] = d.to_ms(m.predict(flat(Xva)))[None]; save(); continue
    if name == 'XGBoost':
        if 'XGBoost' not in best_hp:
            rng = np.random.default_rng(123); trials = []
            for i in range(N_TRIALS):
                hp = sample(XGB_SPACE, rng)
                m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=1, random_state=0, **hp)
                m.fit(flat(Xtr), ytr, eval_set=[(flat(Xva), yva)], verbose=False)
                trials.append((float(np.sqrt(np.mean((m.predict(flat(Xva)) - yva) ** 2))), hp))
            best_hp['XGBoost'] = min(trials, key=lambda x: x[0])[1]; save()
        P, V = [], []
        for s in SEEDS:
            m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=1, random_state=s, **best_hp['XGBoost'])
            m.fit(flat(Xtr), ytr, eval_set=[(flat(Xva), yva)], verbose=False)
            P.append(d.to_ms(m.predict(flat(Xte)))); V.append(d.to_ms(m.predict(flat(Xva))))
        preds['XGBoost_test'] = np.array(P); preds['XGBoost_val'] = np.array(V); save(); print(TAG, 'XGBoost done', flush=True); continue
    if name not in best_hp:
        rng = np.random.default_rng(sum(map(ord, name))); trials = []
        for i in range(N_TRIALS):
            hp = sample(SPACES[name], rng); _, inf = train_net(name, hp, d, seed=1000 + i, max_epochs=60, patience=5); trials.append((inf['val_rmse_z'], hp))
        best_hp[name] = min(trials, key=lambda x: x[0])[1]; save()
    P, V = [], []
    for s in SEEDS:
        m, inf = train_net(name, best_hp[name], d, seed=s, max_epochs=300, patience=15)
        P.append(d.to_ms(predict(m, Xte))); V.append(d.to_ms(predict(m, Xva)))
    preds[f'{name}_test'] = np.array(P); preds[f'{name}_val'] = np.array(V); save()
    print(TAG, name, 'done', round(float(np.sqrt(np.mean((np.mean(P, 0) - preds['obs_test']) ** 2))), 4), flush=True)
print('finished', TAG, flush=True)

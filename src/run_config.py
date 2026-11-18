"""Full protocol for one configuration of the factorial design.
usage: python run_config.py TAG {base|phys} {0|1} [model names...]
  base = the paper's 8 daily-mean ERA5 variables
  phys = 8 variables + hourly-consistent / diurnal predictors + day-of-year
  regime 1 = adds a regime indicator (1 from 2011-02-01, the Pettitt break), 0 = none
Same protocol as s2_tune_seeds.py: L = 7, equal budget of 30 random trials scored on
validation RMSE, then 10 seeds of the best setting. Train 2005-2014, val 2015-2017, test 2018-2020."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, os, sys, time
import numpy as np, pandas as pd
import torch, xgboost as xgb
from sklearn.linear_model import LinearRegression
import lib
from lib import STATION_SPLIT, Data, SPACES, XGB_SPACE, sample, train_net, predict, n_params

torch.set_num_threads(1)
TAG, FSET, REG = sys.argv[1], sys.argv[2], int(sys.argv[3])
names = sys.argv[4:] or ['MLR', 'XGBoost', 'TCN', 'Attention-TCN', 'Transformer', 'LSTM', 'BiLSTM', 'Attention-BiLSTM']
OUT = f'results/main/{TAG}'; os.makedirs(OUT, exist_ok=True)
N_TRIALS, SEEDS, BREAK = 30, list(range(10)), '2011-02-01'

BASE = ['u10', 'v10', 'd2m', 't2m', 'blh', 'msl', 'sp', 'tp']
PHYS = BASE + ['ws_scalar', 'ws_max', 'ws_day', 'ws_night', 'ws_sd', 'blh_max', 'blh_min', 't2m_range',
               'steady', 'dir_sin', 'dir_cos', 'doy_sin', 'doy_cos']


def make_df():
    D = pd.read_csv('data/era5_daily_ext.csv', parse_dates=['date'])
    S = pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'station'})
    D = D.merge(S, on='date', how='left')
    D.loc[D.date.dt.year < 2005, 'station'] = np.nan
    doy = D.date.dt.dayofyear
    D['doy_sin'], D['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    D['regime'] = (D.date >= BREAK).astype(float)
    return D


feats = (PHYS if FSET == 'phys' else BASE) + (['regime'] if REG else [])
d = Data(7, 0, split=STATION_SPLIT, df=make_df(), target='station', features=feats)
Xtr, ytr, _ = d.get('train'); Xva, yva, _ = d.get('val'); Xte, yte, tte = d.get('test')
flat = lambda X: X.reshape(len(X), -1)
log = open(f'{OUT}/log.txt', 'a')


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


hp_file, pf, cf = f'{OUT}/best_hp.json', f'{OUT}/preds.npz', f'{OUT}/cost.json'
best_hp = json.load(open(hp_file)) if os.path.exists(hp_file) else {}
preds = dict(np.load(pf, allow_pickle=True)) if os.path.exists(pf) else {}
cost = json.load(open(cf)) if os.path.exists(cf) else {}


def save():
    json.dump(best_hp, open(hp_file, 'w'), indent=1); np.savez(pf, **preds); json.dump(cost, open(cf, 'w'), indent=1)


preds['obs_test'] = d.to_ms(yte); preds['obs_val'] = d.to_ms(yva); preds['test_idx'] = tte
preds['test_dates'] = d.df.date.values[tte].astype('datetime64[D]').astype(str)
say('config', TAG, FSET, REG, 'features', len(feats), 'train/val/test', len(ytr), len(yva), len(yte))

for name in names:
    if f'{name}_test' in preds:
        continue
    if name == 'MLR':
        t0 = time.time(); m = LinearRegression().fit(flat(Xtr), ytr); tt = time.time() - t0
        preds['MLR_test'] = d.to_ms(m.predict(flat(Xte)))[None]; preds['MLR_val'] = d.to_ms(m.predict(flat(Xva)))[None]
        cost['MLR'] = dict(train_s=tt, params=int(flat(Xtr).shape[1] + 1)); save(); say('MLR done'); continue
    if name == 'XGBoost':
        if 'XGBoost' not in best_hp:
            rng = np.random.default_rng(123); trials = []
            for i in range(N_TRIALS):
                hp = sample(XGB_SPACE, rng)
                m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=1, random_state=0, **hp)
                m.fit(flat(Xtr), ytr, eval_set=[(flat(Xva), yva)], verbose=False)
                trials.append((float(np.sqrt(np.mean((m.predict(flat(Xva)) - yva) ** 2))), hp))
            best_hp['XGBoost'] = min(trials, key=lambda x: x[0])[1]; save(); say('XGB best', best_hp['XGBoost'])
        P, V, T, NT = [], [], [], []
        for s in SEEDS:
            m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=1, random_state=s, **best_hp['XGBoost'])
            t0 = time.time(); m.fit(flat(Xtr), ytr, eval_set=[(flat(Xva), yva)], verbose=False); T.append(time.time() - t0)
            P.append(d.to_ms(m.predict(flat(Xte)))); V.append(d.to_ms(m.predict(flat(Xva)))); NT.append(m.best_iteration + 1)
        preds['XGBoost_test'] = np.array(P); preds['XGBoost_val'] = np.array(V)
        cost['XGBoost'] = dict(train_s=float(np.mean(T)), params=f'{int(np.mean(NT))} trees'); save(); say('XGBoost done')
        continue
    if name not in best_hp:
        rng = np.random.default_rng(sum(map(ord, name))); trials = []
        for i in range(N_TRIALS):
            hp = sample(SPACES[name], rng)
            _, info = train_net(name, hp, d, seed=1000 + i, max_epochs=60, patience=5)
            trials.append((info['val_rmse_z'], hp))
        best_hp[name] = min(trials, key=lambda x: x[0])[1]; save(); say(name, 'best', best_hp[name])
    P, V, T, E = [], [], [], []
    for s in SEEDS:
        m, info = train_net(name, best_hp[name], d, seed=s, max_epochs=300, patience=15)
        P.append(d.to_ms(predict(m, Xte))); V.append(d.to_ms(predict(m, Xva))); T.append(info['train_s']); E.append(info['epochs'])
        os.makedirs(f'{OUT}/models', exist_ok=True); torch.save(m.state_dict(), f'{OUT}/models/{name}_s{s}.pt')
    preds[f'{name}_test'] = np.array(P); preds[f'{name}_val'] = np.array(V)
    cost[name] = dict(train_s=float(np.mean(T)), params=n_params(m), epochs=float(np.mean(E))); save()
    say(name, 'done; seed-mean test RMSE', round(float(np.sqrt(np.mean((np.mean(P, 0) - preds['obs_test']) ** 2))), 4))
say('finished', TAG)

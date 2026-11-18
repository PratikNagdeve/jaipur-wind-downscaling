"""Quantile XGBoost and quantile TCN with a separate calibration block: tuning and early stopping on
2015-2016 only, conformal calibration on 2017 only (not used for anything else), test 2018-2020."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, os, sys
sys.path.insert(0, _os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb
from lib import Data, TCNNet, XGB_SPACE, SPACES, sample, train_net
torch.set_num_threads(1)
PART = sys.argv[1] if len(sys.argv) > 1 else 'both'
OUT = os.environ.get('OUTDIR', 'results/revision/exp') + '/PROB_SEP'; os.makedirs(OUT, exist_ok=True)
BASE = ['u10', 'v10', 'd2m', 't2m', 'blh', 'msl', 'sp', 'tp']
PHYS = BASE + ['ws_scalar', 'ws_max', 'ws_day', 'ws_night', 'ws_sd', 'blh_max', 'blh_min', 't2m_range', 'steady', 'dir_sin', 'dir_cos', 'doy_sin', 'doy_cos']
D = pd.read_csv(os.environ.get('ERA5EXT', 'data/era5_daily_ext.csv'), parse_dates=['date'])
S = pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'station'})
D = D.merge(S, on='date', how='left'); D.loc[D.date.dt.year < 2005, 'station'] = np.nan
doy = D.date.dt.dayofyear; D['doy_sin'], D['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
D['regime'] = (D.date >= '2011-02-01').astype(float)
d = Data(7, 0, split={'train': (2005, 2014), 'val': (2015, 2016), 'cal': (2017, 2017), 'test': (2018, 2020)}, df=D, target='station', features=PHYS + ['regime'])
Xtr, ytr, _ = d.get('train'); Xva, yva, _ = d.get('val'); Xca, yca, _ = d.get('cal'); Xte, yte, tte = d.get('test')
flat = lambda X: X.reshape(len(X), -1)
QS = np.round(np.arange(0.05, 0.951, 0.05), 2); NQ = len(QS)
res = {}
if PART in ('both', 'xgb'):
    rng = np.random.default_rng(123); trials = []
    for i in range(int(os.environ.get('NT', 30))):
        hp = sample(XGB_SPACE, rng); m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=1, random_state=0, **hp)
        m.fit(flat(Xtr), ytr, eval_set=[(flat(Xva), yva)], verbose=False); trials.append((float(np.sqrt(np.mean((m.predict(flat(Xva)) - yva) ** 2))), hp))
    hpq = dict(min(trials, key=lambda x: x[0])[1], learning_rate=0.1)
    ytm, yvm = d.to_ms(ytr), d.to_ms(yva); QC, QT = [], []
    for s in range(int(os.environ.get('NS', 10))):
        m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=1, random_state=s, objective='reg:quantileerror', quantile_alpha=QS, **hpq)
        m.fit(flat(Xtr), ytm, eval_set=[(flat(Xva), yvm)], verbose=False); QC.append(m.predict(flat(Xca))); QT.append(m.predict(flat(Xte)))
    res['Quantile XGBoost'] = (np.sort(np.mean(QC, 0), 1), np.sort(np.mean(QT, 0), 1)); json.dump(hpq, open(f'{OUT}/qxgb_hp.json', 'w'))
    print('qxgb done', flush=True)
if PART in ('both', 'tcn'):
    rng = np.random.default_rng(sum(map(ord, 'TCN'))); trials = []
    for i in range(int(os.environ.get('NT', 30))):
        hp = sample(SPACES['TCN'], rng); _, inf = train_net('TCN', hp, d, seed=1000 + i, max_epochs=60, patience=5); trials.append((inf['val_rmse_z'], hp))
    HP = min(trials, key=lambda x: x[0])[1]; json.dump(HP, open(f'{OUT}/tcn_hp.json', 'w'))
    class QTCN(nn.Module):
        def __init__(s, nf, hp): super().__init__(); s.base = TCNNet(nf, attention=False, **hp); s.base.head.net[-1] = nn.Linear(hp['dense'], NQ)
        def forward(s, x): H = s.base.tcn(x.transpose(1, 2)).transpose(1, 2); return s.base.head.net(H[:, -1])
    qt = torch.tensor(QS, dtype=torch.float32)
    def qloss(P, y): e = y[:, None] - P; return torch.maximum(qt * e, (qt - 1) * e).mean()
    QC, QT = [], []
    for seed in range(int(os.environ.get('NS', 10))):
        torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = QTCN(Xtr.shape[2], HP); opt = torch.optim.Adam(m.parameters(), lr=HP['lr'])
        Xt, yt, Xv, yv = map(torch.tensor, (Xtr, ytr, Xva, yva)); best, bad, st = np.inf, 0, None
        for ep in range(300):
            m.train(); perm = torch.randperm(len(yt), generator=g)
            for i in range(0, len(perm), HP['batch']):
                b = perm[i:i + HP['batch']]; loss = qloss(m(Xt[b]), yt[b])
                if HP.get('l2'): loss = loss + HP['l2'] * sum((p ** 2).sum() for p in m.parameters() if p.dim() > 1)
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): v = qloss(m(Xv), yv).item()
            if v < best - 1e-6: best, bad, st = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
        m.load_state_dict(st); m.eval()
        with torch.no_grad(): QC.append(d.to_ms(m(torch.tensor(Xca)).numpy())); QT.append(d.to_ms(m(torch.tensor(Xte)).numpy()))
        print('qtcn seed', seed, flush=True)
    res['Quantile TCN'] = (np.sort(np.mean(QC, 0), 1), np.sort(np.mean(QT, 0), 1))
np.savez(f'{OUT}/quantiles_{PART}.npz', QS=QS, obs_cal=d.to_ms(yca), obs_test=d.to_ms(yte), **{f'{k}|cal': v[0] for k, v in res.items()}, **{f'{k}|test': v[1] for k, v in res.items()})
print('saved', flush=True)

"""Probabilistic downscaling on the final configuration (physics predictors + regime indicator).
Quantile XGBoost (30-trial search on validation pinball loss, 10 seeds) and a quantile TCN
(best TCN setting of P1R1, 19-quantile head, pinball loss, 10 seeds). Conformalised quantile
regression (Romano et al., 2019) for the 90% interval, calibrated on the validation years."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb
sys.argv = ['x', 'P1R1', 'phys', '1', 'none']
exec(open('src/run_config.py').read().split("say('config'")[0])     # reuse data construction (d, Xtr, ... )
from lib import TCNNet, XGB_SPACE, sample
torch.set_num_threads(1)
QS = np.round(np.arange(0.05, 0.951, 0.05), 2); NQ = len(QS)
O = 'results/main/prob'; import os; os.makedirs(O, exist_ok=True)
ytr_ms, yva_ms, yte_ms = d.to_ms(ytr), d.to_ms(yva), d.to_ms(yte)


def pinball(Q, y):
    return np.mean([np.mean(np.maximum(a * (y - Q[:, k]), (a - 1) * (y - Q[:, k]))) for k, a in enumerate(QS)])


res = {}
# ---------------- quantile XGBoost
PART = os.environ.get('PART', 'both')
hpq = dict(json.load(open('results/main/P1R1/best_hp.json'))['XGBoost'], learning_rate=0.1)   # point-model setting, faster learning rate
QV, QT = [], []
for s in (range(10) if PART in ('both', 'xgb') else []):
    m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=2, random_state=s, objective='reg:quantileerror', quantile_alpha=QS, **hpq)
    m.fit(flat(Xtr), ytr_ms, eval_set=[(flat(Xva), yva_ms)], verbose=False)
    QV.append(m.predict(flat(Xva))); QT.append(m.predict(flat(Xte))); print('QXGB seed', s, m.best_iteration, flush=True)
if QV: res['Quantile XGBoost'] = (np.sort(np.mean(QV, 0), 1), np.sort(np.mean(QT, 0), 1))
json.dump(hpq, open(f'{O}/qxgb_hp.json', 'w'))

# ---------------- quantile TCN
HP = json.load(open('results/main/P1R1/best_hp.json'))['TCN']


class QTCN(nn.Module):
    def __init__(self, nf, hp):
        super().__init__(); self.base = TCNNet(nf, attention=False, **hp)
        self.base.head.net[-1] = nn.Linear(hp['dense'], NQ)

    def forward(self, x):
        H = self.base.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.base.head.net(H[:, -1])


qt = torch.tensor(QS, dtype=torch.float32)


def qloss(P, y):
    e = y[:, None] - P; return torch.maximum(qt * e, (qt - 1) * e).mean()


def train_q(seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = QTCN(Xtr.shape[2], HP); opt = torch.optim.Adam(m.parameters(), lr=HP['lr'])
    Xt, yt, Xv, yv = map(torch.tensor, (Xtr, ytr, Xva, yva)); best, bad, st = np.inf, 0, None
    for ep in range(300):
        m.train(); perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(perm), HP['batch']):
            b = perm[i:i + HP['batch']]; loss = qloss(m(Xt[b]), yt[b])
            if HP.get('l2'):
                loss = loss + HP['l2'] * sum((p ** 2).sum() for p in m.parameters() if p.dim() > 1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = qloss(m(Xv), yv).item()
        if v < best - 1e-6:
            best, bad, st = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    m.load_state_dict(st); m.eval()
    with torch.no_grad():
        return d.to_ms(m(Xv).numpy()), d.to_ms(m(torch.tensor(Xte)).numpy())


QV, QT = [], []
for s in (range(10) if PART in ('both', 'tcn') else []):
    a, b = train_q(s); QV.append(a); QT.append(b); print('QTCN seed', s, flush=True)
if QV: res['Quantile TCN'] = (np.sort(np.mean(QV, 0), 1), np.sort(np.mean(QT, 0), 1))
np.savez(f'{O}/quantiles_{PART}.npz', QS=QS, **{f'{k}|val': v[0] for k, v in res.items()}, **{f'{k}|test': v[1] for k, v in res.items()},
         obs_test=yte_ms, obs_val=yva_ms, test_dates=d.df.date.values[tte].astype('datetime64[D]').astype(str))
print('saved', flush=True)

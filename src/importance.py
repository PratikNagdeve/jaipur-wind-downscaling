"""Permutation importance in the final configuration (P1R1) for XGBoost, TCN and BiLSTM.
Each predictor (or physical group) is shuffled across test days in all seven days of the window;
increase in test RMSE (m/s), 10 seeds x 10 repeats. Also: IMDAA comparison on the test days."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, sys
import numpy as np, pandas as pd, torch, xgboost as xgb
sys.argv = ['x', 'P1R1', 'phys', '1', 'none']
exec(open('src/run_config.py').read().split("say('config'")[0])
from lib import build, predict
torch.set_num_threads(2)
HP = json.load(open('results/main/P1R1/best_hp.json'))
o = d.to_ms(yte); rng = np.random.default_rng(7)
GROUPS = {'Wind speed (5 hourly-based)': ['ws_scalar', 'ws_max', 'ws_day', 'ws_night', 'ws_sd'],
          'Wind components (u10, v10)': ['u10', 'v10'], 'Wind direction and steadiness': ['dir_sin', 'dir_cos', 'steady'],
          'Boundary layer height (mean, max, min)': ['blh', 'blh_max', 'blh_min'], 'Temperature and humidity (t2m, d2m, t2m range)': ['t2m', 'd2m', 't2m_range'],
          'Pressure (sp, msl)': ['sp', 'msl'], 'Precipitation (tp)': ['tp'], 'Day of year': ['doy_sin', 'doy_cos'], 'Regime indicator': ['regime']}
SINGLE = feats


def models():
    out = {'XGBoost': [], 'TCN': [], 'BiLSTM': []}
    for s in range(10):
        m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=2, random_state=s, **HP['XGBoost'])
        m.fit(flat(Xtr), ytr, eval_set=[(flat(Xva), yva)], verbose=False)
        out['XGBoost'].append(lambda X, m=m: m.predict(flat(X)))
        for n in ('TCN', 'BiLSTM'):
            net = build(n, Xtr.shape[2], HP[n]); net.load_state_dict(torch.load(f'results/main/P1R1/models/{n}_s{s}.pt')); net.eval()
            out[n].append(lambda X, net=net: predict(net, X))
    return out


M = models(); rows = []
for name, fs in M.items():
    for s, f in enumerate(fs):
        base = np.sqrt(np.mean((d.to_ms(f(Xte)) - o) ** 2))
        for gname, cols in list(GROUPS.items()) + [(c, [c]) for c in SINGLE]:
            idx = [feats.index(c) for c in cols]; inc = []
            for r in range(10):
                Xp = Xte.copy(); perm = rng.permutation(len(Xp)); Xp[:, :, idx] = Xp[perm][:, :, idx]
                inc.append(np.sqrt(np.mean((d.to_ms(f(Xp)) - o) ** 2)) - base)
            rows.append(dict(model=name, seed=s, kind='group' if gname in GROUPS else 'single', predictor=gname, inc=np.mean(inc)))
    print(name, 'done', flush=True)
I = pd.DataFrame(rows); I.to_csv('results/main/importance_runs.csv', index=False)
S = I.groupby(['kind', 'predictor', 'model']).inc.agg(['mean', 'std']).reset_index(); S.to_csv('results/main/importance.csv', index=False)
print(S[S.kind == 'group'].pivot(index='predictor', columns='model', values='mean').round(3).sort_values('TCN', ascending=False))
print(S[S.kind == 'single'].pivot(index='predictor', columns='model', values='mean').round(3).sort_values('TCN', ascending=False).head(12))

# ---- IMDAA on the test days
D = pd.read_csv('data/era5_daily_ext.csv', parse_dates=['date']).merge(
    pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'obs'}), on='date', how='left')
D.loc[D.date.dt.year < 2005, 'obs'] = np.nan
Di = D.set_index('date'); dates = d.df.date.values[tte]
tr = D[D.date.dt.year.between(2005, 2014) & D.obs.notna()]
res = {}
for col, lab in [('imdaa_average_speed', 'IMDAA (raw)'), ('ws_scalar', 'ERA5 hourly-mean (raw)')]:
    x = Di.loc[dates, col].values; b, a = np.polyfit(tr[col], tr.obs, 1)
    for k, p in [(lab, x), (lab.replace('(raw)', '+ linear correction'), a + b * x)]:
        e = p - o; sw = o > np.percentile(tr.obs, 90)
        res[k] = dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))), Bias=float(e.mean()), r=float(np.corrcoef(p, o)[0, 1]),
                      RMSE_strong=float(np.sqrt(np.mean(e[sw] ** 2))))
# IMDAA as an extra input to XGBoost in the final configuration (10 seeds)
d2 = Data(7, 0, split=STATION_SPLIT, df=make_df(), target='station', features=feats + ['imdaa_average_speed'])
X2tr, y2tr, _ = d2.get('train'); X2va, y2va, _ = d2.get('val'); X2te, y2te, _ = d2.get('test')
P = []
for s in range(10):
    m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, n_jobs=2, random_state=s, **HP['XGBoost'])
    m.fit(flat(X2tr), y2tr, eval_set=[(flat(X2va), y2va)], verbose=False); P.append(d2.to_ms(m.predict(flat(X2te))))
P = np.array(P); per = [np.sqrt(np.mean((p - o) ** 2)) for p in P]
res['XGBoost final + IMDAA input'] = dict(RMSE=float(np.mean(per)), RMSE_sd=float(np.std(per, ddof=1)), Bias=float(np.mean(P.mean(0) - o)), r=float(np.corrcoef(P.mean(0), o)[0, 1]))
from stats_util import dm_test
base = np.load('results/main/P1R1/preds.npz')['XGBoost_test']
res['XGBoost final + IMDAA input']['DM_p_vs_final_XGBoost'] = float(dm_test(np.mean((P - o) ** 2, 0), np.mean((base - o) ** 2, 0))[1])
json.dump(res, open('results/main/imdaa.json', 'w'), indent=1); print(json.dumps(res, indent=1))

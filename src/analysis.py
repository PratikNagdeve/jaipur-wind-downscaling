"""Evaluation of the factorial experiment (out_rev/P{0,1}R{0,1}) and the probabilistic models."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, os
import numpy as np, pandas as pd
from scipy import stats
from stats_util import dm_test

O = 'results/main'
CFG = {'P0R0': 'Original predictors', 'P0R1': 'Original + regime', 'P1R0': 'Physics predictors', 'P1R1': 'Physics + regime'}
MODELS = ['MLR', 'XGBoost', 'LSTM', 'BiLSTM', 'Attention-BiLSTM', 'TCN', 'Attention-TCN', 'Transformer']
R = {c: dict(np.load(f'{O}/{c}/preds.npz', allow_pickle=True)) for c in CFG if os.path.exists(f'{O}/{c}/preds.npz')}
ref = R['P1R1']
obs = ref['obs_test']; dates = pd.to_datetime(ref['test_dates'])
for c in R:
    assert np.allclose(R[c]['obs_test'], obs)

D = pd.read_csv('data/era5_daily_ext.csv', parse_dates=['date']).merge(
    pd.read_csv('data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'obs'}), on='date', how='left')
D.loc[D.date.dt.year < 2005, 'obs'] = np.nan
Di = D.set_index('date')
tr = D[(D.date.dt.year.between(2005, 2014)) & D.obs.notna()]
va = D[(D.date.dt.year.between(2015, 2017)) & D.obs.notna()]


def met(p, o=obs):
    e = p - o
    return dict(RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)), Bias=e.mean(), r=np.corrcoef(p, o)[0, 1],
                R2=1 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2))


def boot_ci(p, o=obs, B=2000, block=30, seed=0):
    rng = np.random.default_rng(seed); n = len(o); nb = int(np.ceil(n / block)); out = []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, nb); idx = (st[:, None] + np.arange(block)).ravel()[:n]
        out.append(np.sqrt(np.mean((p[idx] - o[idx]) ** 2)))
    return np.percentile(out, [2.5, 97.5])


# ---------------- baselines on the same test days
base = {}
for col, lab in [('ws_vec', 'vector-mean speed'), ('ws_scalar', 'hourly-mean speed')]:
    x = Di.loc[dates, col].values
    base[f'Raw ERA5 ({lab})'] = x
    b, a = np.polyfit(tr[col], tr.obs, 1); base[f'ERA5 {lab} + linear correction'] = a + b * x
    q = np.linspace(0.01, 0.99, 99); base[f'ERA5 {lab} + quantile mapping'] = np.interp(x, np.quantile(tr[col], q), np.quantile(tr.obs, q))
RAW = base['Raw ERA5 (hourly-mean speed)']
rmse_raw = met(RAW)['RMSE']; rmse_raw_vec = met(base['Raw ERA5 (vector-mean speed)'])['RMSE']

rows = []
for k, p in base.items():
    rows.append(dict(config='Baseline', model=k, **met(p), RMSE_sd=0, CI_lo=boot_ci(p)[0], CI_hi=boot_ci(p)[1]))
for c in R:
    for mname in MODELS:
        if f'{mname}_test' not in R[c]:
            continue
        P = R[c][f'{mname}_test']; per = pd.DataFrame([met(p) for p in P]); pm = P.mean(0); ci = boot_ci(pm)
        rows.append(dict(config=c, model=mname, **per.mean().to_dict(), RMSE_sd=per.RMSE.std(ddof=1) if len(P) > 1 else 0,
                         RMSE_seedmean=met(pm)['RMSE'], CI_lo=ci[0], CI_hi=ci[1]))
T = pd.DataFrame(rows)
T['SS_vs_hourly_ERA5'] = 1 - T.RMSE / rmse_raw
T['SS_vs_vector_ERA5'] = 1 - T.RMSE / rmse_raw_vec


# ---------------- DM tests (seed-averaged squared errors), Holm within config vs best
def sqerr(c, m): return np.mean((R[c][f'{m}_test'] - obs) ** 2, 0)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (len(ps) - i) * ps[j])); adj[j] = run
    return adj


T['DM_p_vs_best_in_config'] = np.nan
for c in R:
    sub = T[(T.config == c)]; best = sub.loc[sub.RMSE.idxmin(), 'model']
    others = [m for m in sub.model if m != best]
    ps = holm([dm_test(sqerr(c, m), sqerr(c, best))[1] for m in others])
    for m, p in zip(others, ps):
        T.loc[(T.config == c) & (T.model == m), 'DM_p_vs_best_in_config'] = p
    T.loc[(T.config == c) & (T.model == best), 'best_in_config'] = True
T.to_csv(f'{O}/table_factorial.csv', index=False)

# DM: each model, final config vs original config
dm_rows = []
if 'P0R0' in R:
    for m in MODELS:
        if f'{m}_test' in R['P0R0'] and f'{m}_test' in R['P1R1']:
            for a, b in [('P1R0', 'P0R0'), ('P1R1', 'P1R0'), ('P1R1', 'P0R0')]:
                if a in R and b in R:
                    s, p = dm_test(sqerr(a, m), sqerr(b, m)); dm_rows.append(dict(model=m, comparison=f'{a} vs {b}', stat=s, p=p))
    pd.DataFrame(dm_rows).to_csv(f'{O}/dm_between_configs.csv', index=False)

# ---------------- factorial summary: architecture spread vs predictor/regime effects
learned = T[T.config.isin(CFG)]
piv = learned.pivot(index='model', columns='config', values='RMSE')
piv.to_csv(f'{O}/factorial_rmse_pivot.csv')
summ = {}
for c in piv.columns:
    nn_ = piv.loc[[m for m in MODELS if m not in ('MLR',)], c]
    summ[c] = dict(mean_learned=float(piv[c].mean()), mean_nonlinear=float(nn_.mean()), min=float(nn_.min()), max=float(nn_.max()),
                   arch_spread=float(nn_.max() - nn_.min()), best=nn_.idxmin())
if set(CFG) <= set(piv.columns):
    nl = [m for m in MODELS if m != 'MLR']
    summ['effects_nonlinear_mean'] = dict(
        predictor_effect_R0=float((piv.loc[nl, 'P0R0'] - piv.loc[nl, 'P1R0']).mean()),
        predictor_effect_R1=float((piv.loc[nl, 'P0R1'] - piv.loc[nl, 'P1R1']).mean()),
        regime_effect_P0=float((piv.loc[nl, 'P0R0'] - piv.loc[nl, 'P0R1']).mean()),
        regime_effect_P1=float((piv.loc[nl, 'P1R0'] - piv.loc[nl, 'P1R1']).mean()))
    # two-way ANOVA-style variance decomposition of RMSE (models x predictors x regime)
    L = learned[learned.model != 'MLR'].copy(); L['P'] = L.config.str[1]; L['Rg'] = L.config.str[3]
    gm = L.RMSE.mean(); ss = {}
    for f in ['model', 'P', 'Rg']:
        ss[f] = float(sum(L.groupby(f).RMSE.mean().sub(gm).pow(2) * L.groupby(f).size()))
    ss['total'] = float(((L.RMSE - gm) ** 2).sum()); summ['variance_share'] = {k: v / ss['total'] for k, v in ss.items() if k != 'total'}
json.dump(summ, open(f'{O}/factorial_summary.json', 'w'), indent=1)

# ---------------- seasons, strong wind, per year, sensitivity (final config, seed-mean)
thr = np.percentile(tr.obs, 90)
seas = {'Winter (JF)': [1, 2], 'Pre-monsoon (MAM)': [3, 4, 5], 'Monsoon (JJAS)': [6, 7, 8, 9], 'Post-monsoon (OND)': [10, 11, 12]}
sus = (dates >= '2020-07-01') & (dates <= '2020-10-31')
SR = {'Raw ERA5 (hourly-mean)': RAW, 'ERA5 + linear correction': base['ERA5 hourly-mean speed + linear correction']}
for c, lab in [('P0R0', 'original'), ('P1R1', 'final')]:
    if c in R:
        for m in ['XGBoost', 'TCN', 'Attention-TCN', 'MLR']:
            SR[f'{m} ({lab})'] = R[c][f'{m}_test'].mean(0)
srows = []
for k, p in SR.items():
    r = {'series': k}
    for s, mm in seas.items():
        msk = dates.month.isin(mm); r[s] = np.sqrt(np.mean((p[msk] - obs[msk]) ** 2))
    sw = obs > thr; r['Strong-wind RMSE'] = np.sqrt(np.mean((p[sw] - obs[sw]) ** 2)); r['Strong-wind bias'] = np.mean(p[sw] - obs[sw])
    for y in (2018, 2019, 2020):
        msk = dates.year == y; r[str(y)] = np.sqrt(np.mean((p[msk] - obs[msk]) ** 2)); r[f'bias {y}'] = np.mean(p[msk] - obs[msk])
    r['RMSE excl. Jul-Oct 2020'] = np.sqrt(np.mean((p[~sus] - obs[~sus]) ** 2)); r['RMSE all'] = np.sqrt(np.mean((p - obs) ** 2))
    srows.append(r)
S = pd.DataFrame(srows); S.to_csv(f'{O}/table_seasons_years.csv', index=False)
json.dump(dict(strong_threshold=float(thr), n_strong=int((obs > thr).sum()), n_suspect=int(sus.sum())), open(f'{O}/meta.json', 'w'))


# ---------------- probabilistic
def weib(v):
    v = np.ravel(v); v = v[v > 0.05]; k, _, c = stats.weibull_min.fit(v, floc=0); return k, c


def dstats(v):
    v = np.ravel(v); k, c = weib(v)
    return dict(mean=v.mean(), sd=v.std(), p99=np.percentile(v, 99), weibull_k=k, weibull_c=c, v3=np.mean(np.clip(v, 0, None) ** 3))


E = {'Station': dstats(obs), 'Raw ERA5 (vector-mean)': dstats(base['Raw ERA5 (vector-mean speed)']), 'Raw ERA5 (hourly-mean)': dstats(RAW),
     'ERA5 + linear correction': dstats(SR['ERA5 + linear correction'])}
for c, lab in [('P0R0', 'original'), ('P1R1', 'final')]:
    if c in R:
        for m in ['XGBoost', 'TCN']:
            E[f'{m} deterministic ({lab})'] = dstats(R[c][f'{m}_test'].mean(0))
PR = {}
if os.path.exists(f'{O}/prob/quantiles_xgb.npz') and os.path.exists(f'{O}/prob/quantiles_tcn.npz'):
    Z = {**dict(np.load(f'{O}/prob/quantiles_xgb.npz')), **dict(np.load(f'{O}/prob/quantiles_tcn.npz'))}
    np.savez(f'{O}/prob/quantiles.npz', **Z); QS = Z['QS']
    clim = np.array([np.quantile(tr.obs[tr.date.dt.month == mm], QS) for mm in range(1, 13)])
    Qc = clim[dates.month.values - 1]

    def crps_q(Q, y):  # average quantile score x2 approximates CRPS (Gneiting & Raftery 2007)
        return np.mean([2 * np.mean(((y < Q[:, k]).astype(float) - a) * (Q[:, k] - y)) for k, a in enumerate(QS)])

    vobs = Z['obs_val']; rng = np.random.default_rng(0)
    p_clim = np.mean(tr.obs > thr); sw = obs > thr
    for name in ['Quantile XGBoost', 'Quantile TCN']:
        Qv, Qt = Z[f'{name}|val'], Z[f'{name}|test']; i0, i1 = 0, len(QS) - 1
        sc = np.maximum(Qv[:, i0] - vobs, vobs - Qv[:, i1]); qh = np.quantile(sc, min(1, 0.9 * (1 + 1 / len(vobs))))
        lo, hi = Qt[:, i0] - qh, Qt[:, i1] + qh
        pit = np.array([np.interp(y, Qt[i], QS, left=0, right=1) for i, y in enumerate(obs)])
        # exceedance probability from the predictive distribution (interpolated CDF)
        pex = np.array([1 - np.interp(thr, Qt[i], QS, left=0, right=1) for i in range(len(obs))])
        bs, bsc = np.mean((pex - sw) ** 2), np.mean((p_clim - sw) ** 2)
        med = Qt[:, list(np.round(QS, 2)).index(0.5)]
        # stochastic members: inverse-CDF sampling with linear tails, 50 members
        ext = lambda Q: np.c_[np.maximum(0, Q[:, :1] - (Q[:, 1:2] - Q[:, :1])), Q, Q[:, -1:] + (Q[:, -1:] - Q[:, -2:-1])]
        grid = np.r_[0, QS, 1]; Qe = ext(Qt)
        u = rng.uniform(size=(len(obs), 50)); draws = np.array([np.interp(u[i], grid, Qe[i]) for i in range(len(obs))])
        PR[name] = dict(CRPS=crps_q(Qt, obs), CRPS_clim=crps_q(Qc, obs), CRPSS=1 - crps_q(Qt, obs) / crps_q(Qc, obs),
                        cov90_raw=np.mean((obs >= Qt[:, i0]) & (obs <= Qt[:, i1])), width90_raw=np.mean(Qt[:, i1] - Qt[:, i0]),
                        cov90_conformal=np.mean((obs >= lo) & (obs <= hi)), width90_conformal=np.mean(hi - lo),
                        Brier=bs, Brier_clim=bsc, BSS=1 - bs / bsc, median_RMSE=met(med)['RMSE'], median_bias=met(med)['Bias'],
                        PIT_deciles=np.histogram(pit, bins=10, range=(0, 1))[0].tolist())
        E[f'{name}, 50 sampled members'] = dstats(draws)
        np.save(f'{O}/prob/pit_{name.replace(" ", "_")}.npy', pit)
        np.save(f'{O}/prob/pex_{name.replace(" ", "_")}.npy', pex)
    json.dump(PR, open(f'{O}/prob_metrics.json', 'w'), indent=1, default=float)
EN = pd.DataFrame(E).T; EN['v3_ratio'] = EN.v3 / EN.loc['Station', 'v3']; EN.to_csv(f'{O}/table_energy.csv')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(T[['config', 'model', 'RMSE', 'RMSE_sd', 'RMSE_seedmean', 'CI_lo', 'CI_hi', 'MAE', 'Bias', 'r', 'SS_vs_hourly_ERA5', 'DM_p_vs_best_in_config']].round(3).to_string())
print(json.dumps(summ, indent=1))
print(S.round(3).to_string()); print(EN.round(3).to_string()); print(json.dumps(PR, indent=1, default=float))
if dm_rows: print(pd.DataFrame(dm_rows).round(4).to_string())

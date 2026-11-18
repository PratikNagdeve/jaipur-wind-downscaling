"""Collect all revision numbers into out/numbers.json and markdown tables into out/tables.json."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/rev/base')
from stats_util import dm_test
R = ROOT; B = f'{R}/results/main'; X = f'{R}/results/revision/exp'
N = {}; TB = {}
rm = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
def load(path): return dict(np.load(path, allow_pickle=True))
P = {c: load(f'{B}/{c}/preds.npz') for c in ['P0R0', 'P0R1', 'P1R0', 'P1R1']}
for t in ['A_WS', 'B_SPD', 'C_NOSPD', 'D_POST', 'E_RATIO']:
    try: P[t] = load(f'{X}/{t}/preds.npz')
    except FileNotFoundError: pass
y = P['P1R1']['obs_test']
for k, v in P.items(): assert np.allclose(v['obs_test'], y, atol=1e-4), k
def seedmean(c, m): return float(np.mean([rm(p, y) for p in P[c][f'{m}_test']]))
def seedsd(c, m): a = P[c][f'{m}_test']; return float(np.std([rm(p, y) for p in a], ddof=1)) if len(a) > 1 else 0.0
def bias(c, m): return float(np.mean(P[c][f'{m}_test'].mean(0) - y))
def dmp(c1, c2, m):
    l1 = np.mean((P[c1][f'{m}_test'] - y) ** 2, 0); l2 = np.mean((P[c2][f'{m}_test'] - y) ** 2, 0); return float(dm_test(l1, l2)[1])
f3 = lambda v: f'{v:.3f}'; sg = lambda v: f'{v:+.3f}'.replace('-', '−')
pf = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'
# ANOVA
A = json.load(open(f'{R}/results/revision/anova.json'))
N['anova_P_cell'] = f"{100*A['cell']['P']:.0f}"; N['anova_P_seed'] = f"{100*A['seed']['P']:.0f}"
N['anova_M_main'] = f"{100*A['seed']['M']:.1f}"; N['anova_M_main_cell'] = f"{100*A['cell']['M']:.1f}"
mall = sum(v for k, v in A['seed'].items() if 'M' in k.split(' x ')); N['anova_M_all'] = f'{100*mall:.1f}'
mallc = sum(v for k, v in A['cell'].items() if 'M' in k.split(' x ')); N['anova_M_all_cell'] = f'{100*mallc:.1f}'
for k, v in A['seed'].items(): N['anova_seed_' + k.replace(' x ', '').replace(' ', '_').replace('(', '').replace(')', '')] = f'{100*v:.1f}'
for k, v in A['cell'].items(): N['anova_cell_' + k.replace(' x ', '').replace(' ', '_').replace('(', '').replace(')', '')] = f'{100*v:.1f}'
N['anova_withMLR_M'] = f"{100*A['with_mlr']['M']:.0f}"; N['anova_withMLR_P'] = f"{100*A['with_mlr']['P']:.0f}"
# pairwise
PW = pd.read_csv(f'{R}/results/revision/pairwise_ci.csv')
nl = ['XGBoost', 'LSTM', 'Attention-BiLSTM', 'TCN', 'Attention-TCN', 'Transformer']
b = PW[PW.ref == 'BiLSTM'].set_index('model')
N['max_nl_gap'] = f"{b.loc[nl,'diff'].max():.3f}"; N['max_nl_gap_pct'] = f"{100*b.loc[nl,'rel'].max():.1f}"
N['max_nl_ci_hi'] = f"{b.loc[nl,'hi'].max():.3f}"; N['mlr_gap'] = f"{b.loc['MLR','diff']:.3f}"; N['mlr_gap_pct'] = f"{100*b.loc['MLR','rel']:.1f}"
N['mlr_ci'] = f"{b.loc['MLR','lo']:.3f}–{b.loc['MLR','hi']:.3f}"; N['lstm_ci'] = f"{b.loc['LSTM','lo']:.3f}–{b.loc['LSTM','hi']:.3f}"
N['pred_gain_mean'] = f"{np.mean([seedmean('P0R1', m) - seedmean('P1R1', m) for m in ['XGBoost','LSTM','BiLSTM','Attention-BiLSTM','TCN','Attention-TCN','Transformer']]):.3f}"
# intermediate predictor sets
M4 = ['MLR', 'XGBoost', 'TCN', 'BiLSTM']; rows = []
lab = {'P0R1': 'Original + regime (P0R1)', 'A_WS': 'A: original + *WS*', 'B_SPD': 'B: original + five speed predictors', 'C_NOSPD': 'C: original + non-speed physics predictors', 'P1R1': 'Physics-based + regime (P1R1)'}
if all(t in P and all(f'{m}_test' in P[t] for m in M4) for t in ['A_WS', 'B_SPD', 'C_NOSPD']):
    for c in ['P0R1', 'A_WS', 'B_SPD', 'C_NOSPD', 'P1R1']:
        rows.append('| ' + lab[c] + ' | ' + ' | '.join(f3(seedmean(c, m)) for m in M4) + ' | ' + f3(np.mean([seedmean(c, m) for m in M4])) + ' |')
    fr = {t: [(seedmean('P0R1', m) - seedmean(t, m)) / (seedmean('P0R1', m) - seedmean('P1R1', m)) for m in M4] for t in ['A_WS', 'B_SPD', 'C_NOSPD']}
    rows.append('| Share of the P0R1 → P1R1 gain reached by A | ' + ' | '.join(f'{100*v:.0f}%' for v in fr['A_WS']) + f" | {100*np.mean(fr['A_WS']):.0f}% |")
    rows.append('| Share of the P0R1 → P1R1 gain reached by B | ' + ' | '.join(f'{100*v:.0f}%' for v in fr['B_SPD']) + f" | {100*np.mean(fr['B_SPD']):.0f}% |")
    rows.append('| Share of the P0R1 → P1R1 gain reached by C | ' + ' | '.join(f'{100*v:.0f}%' for v in fr['C_NOSPD']) + f" | {100*np.mean(fr['C_NOSPD']):.0f}% |")
    TB['T_inter'] = '| Predictor set (all with regime indicator) | MLR | XGBoost | TCN | BiLSTM | Mean |\n|---|---|---|---|---|---|\n' + '\n'.join(rows)
    for t in ['A_WS', 'B_SPD', 'C_NOSPD']:
        N[f'frac_{t}_min'] = f'{100*min(fr[t]):.0f}'; N[f'frac_{t}_max'] = f'{100*max(fr[t]):.0f}'; N[f'frac_{t}_mean'] = f'{100*np.mean(fr[t]):.0f}'
    N['mean_P0R1_4'] = f3(np.mean([seedmean('P0R1', m) for m in M4])); N['mean_A_4'] = f3(np.mean([seedmean('A_WS', m) for m in M4])); N['mean_P1R1_4'] = f3(np.mean([seedmean('P1R1', m) for m in M4]))
    N['p_P1B_MLR'] = pf(pBP['MLR']) if False else ''
    N['dm_A_vs_P0R1_max'] = pf(max(dmp('A_WS', 'P0R1', m) for m in M4))
    pBP = {m: dmp('P1R1', 'B_SPD', m) for m in M4}; [N.__setitem__(f'p_P1B_{m}', pf(v)) for m, v in pBP.items()]; N['dm_P1_vs_B'] = ', '.join(f'{m} {pf(v)}' for m, v in pBP.items())
    N['gain_B_to_P1'] = f"{np.min([seedmean('B_SPD', m) - seedmean('P1R1', m) for m in M4]):.3f}–{np.max([seedmean('B_SPD', m) - seedmean('P1R1', m) for m in M4]):.3f}"
    N['gain_A_to_B'] = f"{np.min([seedmean('A_WS', m) - seedmean('B_SPD', m) for m in M4]):.3f}–{np.max([seedmean('A_WS', m) - seedmean('B_SPD', m) for m in M4]):.3f}"
    N['gain_P0_to_C'] = f"{np.min([seedmean('P0R1', m) - seedmean('C_NOSPD', m) for m in M4]):.3f}–{np.max([seedmean('P0R1', m) - seedmean('C_NOSPD', m) for m in M4]):.3f}"
    N['nsig_P1_vs_B'] = str(sum(v < 0.05 for v in pBP.values()))
    pCP = {m: dmp('C_NOSPD', 'P0R1', m) for m in M4}; N['dm_C_vs_P0'] = ', '.join(f'{m} {pf(v)}' for m, v in pCP.items()); N['nsig_C_vs_P0'] = str(sum(v < 0.05 for v in pCP.values()))
# regime alternatives
if all(t in P and all(f'{m}_test' in P[t] for m in M4) for t in ['D_POST', 'E_RATIO']):
    rows = []
    for c, l in [('P1R0', 'No treatment (P1R0)'), ('P1R1', 'Regime indicator (P1R1)'), ('D_POST', 'Post-break training only'), ('E_RATIO', 'Ratio-adjusted pre-break target')]:
        rows.append('| ' + l + ' | ' + ' | '.join(f'{f3(seedmean(c, m))} ({sg(bias(c, m))})' for m in M4) + ' |')
    TB['T_regime'] = '| Treatment of the station breaks | MLR | XGBoost | TCN | BiLSTM |\n|---|---|---|---|---|\n' + '\n'.join(rows)
    for c in ['P1R0', 'P1R1', 'D_POST', 'E_RATIO']:
        for m in M4: N[f'{c}_{m}_rmse'] = f3(seedmean(c, m)); N[f'{c}_{m}_bias'] = sg(bias(c, m))
        N[f'{c}_mean_rmse'] = f3(np.mean([seedmean(c, m) for m in M4]))
    for c in ['D_POST', 'E_RATIO']:
        ps = {m: dmp(c, 'P1R1', m) for m in M4}; N[f'dm_{c}_vs_P1R1'] = ', '.join(f'{m} {pf(v)}' for m, v in ps.items()); N[f'nsig_{c}'] = str(sum(v < 0.05 for v in ps.values()))
    N['ratio_factor'] = f"{json.load(open(f'{X}/E_RATIO/cost.json'))['ratio_factor']:.3f}"; N['n_post'] = f"{json.load(open(f'{X}/D_POST/cost.json'))['n_train']:,}"
# final config table with pairwise CI
T5 = pd.read_csv(f'{B}/table_factorial.csv')
# homogeneity table
H = json.load(open(f'{R}/results/revision/homog2.json'))
hr = []
for ref, rl in [('ws_scalar', 'ERA5, hourly-mean speed'), ('era5_vec', 'ERA5, vector-mean speed'), ('imdaa', 'IMDAA')]:
    for per in ['2005-2020', '2005-2014 (training)']:
        h = H[f'{ref}|{per}']; mo = lambda s: pd.Timestamp(s).strftime('%b %Y')
        hr.append(f"| {rl} | {per.replace('-', '–').replace(' (training)', '')} | {h['n_months']} | after {mo(h['pettitt_after'])} | {pf(h['pettitt_p_ar1'])} | {mo(h['snht_at'])} | {h['snht_T']:.1f} | {pf(h['snht_p_white'])} | {pf(h['snht_p_ar1'])} | {h['step_pct']:+.0f}% |".replace('|-', '|−').replace(' -', ' −'))
    N[f'ratio_{ref}_before'] = f"{H[f'{ref}|ratio']['before']:.2f}"; N[f'ratio_{ref}_after'] = f"{H[f'{ref}|ratio']['after']:.2f}"; N[f'ratio_{ref}_chg'] = f"{H[f'{ref}|ratio']['change_pct']:.0f}"
    N[f'aug_{ref}'] = f"{H[f'{ref}|ratio']['aug2020']:.2f}"; N[f'aug_{ref}_rng'] = f"{H[f'{ref}|ratio']['aug_other_min']:.2f}–{H[f'{ref}|ratio']['aug_other_max']:.2f}"
TB['T_homog'] = '| Reference | Period | *n* | Pettitt change point | Pettitt *p* (AR(1)) | SNHT break | *T*~0~ | SNHT *p* (white noise) | SNHT *p* (AR(1)) | Step in ratio |\n|---|---|---|---|---|---|---|---|---|---|\n' + '\n'.join(hr)
N['phi_resid'] = f"{H['ws_scalar|2005-2020']['phi_resid']:.2f}"
N['homog_max_p_ar1'] = pf(max(max(H[k]['pettitt_p_ar1'], H[k]['snht_p_ar1']) for k in H if '|20' in k))
# conformal
CF = pd.read_csv(f'{R}/results/revision/conformal.csv', index_col=0)
for k, r in CF.iterrows():
    key = k.replace('Quantile ', 'Q').replace(' ', '_').replace('|', '_').replace('=', '').replace('.', 'p').replace('-', '_')
    for c in ['cov', 'width', 'cov_2018', 'cov_2019', 'cov_2020', 'cov_JulOct2020', 'roll90_min']: N[f'cf_{key}_{c}'] = f'{r[c]:.3f}' if 'cov' in c or 'roll' in c else f'{r[c]:.2f}'
yrs=[CF.loc[f'Quantile {n}|ACI gamma=0.01', f'cov_{y}'] for n in ('XGBoost','TCN') for y in (2018,2019,2020)]
N['aci_year_min']=f'{min(yrs):.3f}'; N['aci_year_max']=f'{max(yrs):.3f}'
# energy
EC = pd.read_csv(f'{R}/results/revision/energy_ci.csv').set_index('series'); EP = pd.read_csv(f'{R}/results/revision/energy_paired.csv').set_index('series')
N['quant_sd'] = f"{json.load(open(f'{R}/results/revision/quant.json'))['q_sd']:.2f}"; N['ratio_imdaa_chg_abs'] = f"{abs(H['imdaa|ratio']['change_pct']):.0f}"
TB['T_energy'] = EC.to_json(); json.dump(N, open(f'{R}/results/revision/numbers.json', 'w'), indent=1); json.dump(TB, open(f'{R}/results/revision/tables.json', 'w'), indent=1)
print(json.dumps(N, indent=0)[:6000]); [print(k, '\n', v) for k, v in TB.items() if k != 'T_energy']

# ---------- additional tables and response numbers
import os
T = pd.read_csv(f'{B}/table_factorial.csv')
fin = T[T.config == 'P1R1'].set_index('model'); base = T[T.config == 'Baseline'].set_index('model')
pw = PW[PW.ref == 'BiLSTM'].set_index('model')
rows = []
for lab, src in [('Raw ERA5 (hourly-mean speed)', ('b', 'Raw ERA5 (hourly-mean speed)')), ('ERA5 + linear correction', ('b', 'ERA5 hourly-mean speed + linear correction'))]:
    r = base.loc[src[1]]; rows.append(f"| {lab} | {r.RMSE:.3f} | {r.CI_lo:.2f}–{r.CI_hi:.2f} | {r.MAE:.3f} | {sg(r.Bias)} | {r.r:.3f} | {r.R2:.3f} | {r.SS_vs_hourly_ERA5:.3f} | – | – |")
for m in ['MLR', 'XGBoost', 'LSTM', 'BiLSTM', 'Attention-BiLSTM', 'TCN', 'Attention-TCN', 'Transformer']:
    r = fin.loc[m]; p = r.DM_p_vs_best_in_config
    dd = '–' if m == 'BiLSTM' else f"{sg(pw.loc[m,'diff'])} ({pw.loc[m,'lo']:+.3f} to {pw.loc[m,'hi']:+.3f})".replace('-', '−')
    ps = '–' if m == 'BiLSTM' else pf(p)
    rows.append(f"| {m} | {r.RMSE:.3f} | {r.CI_lo:.2f}–{r.CI_hi:.2f} | {r.MAE:.3f} | {sg(r.Bias)} | {r.r:.3f} | {r.R2:.3f} | {r.SS_vs_hourly_ERA5:.3f} | {ps} | {dd} |")
TB['T_final'] = '| Model | RMSE | 95% CI | MAE | Bias | r | R² | SS | p (vs BiLSTM) | ΔRMSE vs BiLSTM (95% CI) |\n|---|---|---|---|---|---|---|---|---|---|\n' + '\n'.join(rows)
# energy table
order = [('Station (observed)', 'Station (observed)'), ('Raw ERA5, vector-mean speed', 'Raw ERA5, vector-mean speed'), ('Raw ERA5, hourly-mean speed', 'Raw ERA5, hourly-mean speed'),
         ('ERA5 + linear correction', 'ERA5 + linear correction'), ('XGBoost, original set-up', 'XGBoost, original set-up'), ('TCN, original set-up', 'TCN, original set-up'),
         ('XGBoost, final set-up', 'XGBoost, final set-up'), ('XGBoost + variance inflation', 'XGBoost, final + variance inflation'), ('XGBoost + quantile mapping', 'XGBoost, final + quantile mapping'),
         ('TCN, final set-up', 'TCN, final set-up'), ('TCN + variance inflation', 'TCN, final + variance inflation'), ('TCN + quantile mapping', 'TCN, final + quantile mapping'),
         ('Quantile XGBoost, 50 sampled members', 'Quantile XGBoost, 50 sampled members'), ('Quantile TCN, 50 sampled members', 'Quantile TCN, 50 sampled members')]
rows = []
for k, lab in order:
    e = EC.loc[k]; rr = '–' if k.startswith('Station') else (f"{json.load(open(f'{B}/prob_metrics.json'))[k.split(',')[0]]['median_RMSE']:.3f}ᵐ" if 'members' in k else f"{EP.loc[k,'rmse']:.3f}")
    star = '' if k.startswith('Station') else ('*' if (EP.loc[k, 'dk_lo'] > 0 or EP.loc[k, 'dk_hi'] < 0) else '')
    stare = '' if k.startswith('Station') else ('*' if (EP.loc[k, 'depf_lo'] > 0 or EP.loc[k, 'depf_hi'] < 0) else '')
    v3 = '1.00' if k.startswith('Station') else f"{e.v3_ratio:.2f} ({e.v3_ratio_lo:.2f}–{e.v3_ratio_hi:.2f})"
    rows.append(f"| {lab} | {e['mean']:.2f} | {e.sd:.2f} | {e.p99:.2f} | {e.k:.2f}{star} ({e.k_lo:.2f}–{e.k_hi:.2f}) | {e.c:.2f} | {e.epf:.2f}{stare} ({e.epf_lo:.2f}–{e.epf_hi:.2f}) | {v3} | {rr} |")
TB['T_energy_md'] = '| Series | Mean | SD | P99 | Weibull k (95% CI) | Weibull c | EPF (95% CI) | Mean v³ ratio (95% CI) | RMSE |\n|---|---|---|---|---|---|---|---|---|\n' + '\n'.join(rows)
g = lambda k, c: EC.loc[k, c]
N.update(epf_obs=f"{g('Station (observed)','epf'):.2f}", epf_lc=f"{g('ERA5 + linear correction','epf'):.2f}", k_lc=f"{g('ERA5 + linear correction','k'):.2f}", k_obs=f"{g('Station (observed)','k'):.2f}",
         epf_xgb=f"{g('XGBoost, final set-up','epf'):.2f}", epf_qx=f"{g('Quantile XGBoost, 50 sampled members','epf'):.2f}", epf_qt=f"{g('Quantile TCN, 50 sampled members','epf'):.2f}",
         k_xgb_vi=f"{g('XGBoost + variance inflation','k'):.2f}", epf_xgb_vi=f"{g('XGBoost + variance inflation','epf'):.2f}", k_xgb=f"{g('XGBoost, final set-up','k'):.2f}",
         k_qx=f"{g('Quantile XGBoost, 50 sampled members','k'):.2f}", k_qt=f"{g('Quantile TCN, 50 sampled members','k'):.2f}", k_tcn=f"{g('TCN, final set-up','k'):.2f}",
         rmse_xgb=f"{EP.loc['XGBoost, final set-up','rmse']:.3f}", rmse_xgb_vi=f"{EP.loc['XGBoost + variance inflation','rmse']:.3f}", rmse_xgb_qm=f"{EP.loc['XGBoost + quantile mapping','rmse']:.3f}",
         rmse_tcn=f"{EP.loc['TCN, final set-up','rmse']:.3f}", rmse_tcn_vi=f"{EP.loc['TCN + variance inflation','rmse']:.3f}", rmse_tcn_qm=f"{EP.loc['TCN + quantile mapping','rmse']:.3f}",
         v3_lc=f"{g('ERA5 + linear correction','v3_ratio'):.2f}", v3_qx=f"{g('Quantile XGBoost, 50 sampled members','v3_ratio'):.2f}", v3_qt=f"{g('Quantile TCN, 50 sampled members','v3_ratio'):.2f}",
         v3_xgb=f"{g('XGBoost, final set-up','v3_ratio'):.2f}", v3_tcn=f"{g('TCN, final set-up','v3_ratio'):.2f}", v3_raw=f"{g('Raw ERA5, hourly-mean speed','v3_ratio'):.2f}",
         k_raw=f"{g('Raw ERA5, hourly-mean speed','k'):.2f}", epf_raw=f"{g('Raw ERA5, hourly-mean speed','epf'):.2f}", epf_tcn=f"{g('TCN, final set-up','epf'):.2f}",
         k_tcn_vi=f"{g('TCN + variance inflation','k'):.2f}", k_xgb_qm=f"{g('XGBoost + quantile mapping','k'):.2f}", k_obs_ci=f"{g('Station (observed)','k_lo'):.2f}–{g('Station (observed)','k_hi'):.2f}",
         epf_obs_ci=f"{g('Station (observed)','epf_lo'):.2f}–{g('Station (observed)','epf_hi'):.2f}")
TL = pd.read_csv(f'{R}/results/revision/tail_sensitivity.csv').set_index('case')
dk = max(abs(TL.loc[f'{n}|{m}', 'k'] - TL.loc[f'{n}|linear', 'k']) for n in ['Quantile XGBoost', 'Quantile TCN'] for m in ['none', 'half', 'double'])
dv = max(abs(TL.loc[f'{n}|{m}', 'v3_ratio'] - TL.loc[f'{n}|linear', 'v3_ratio']) for n in ['Quantile XGBoost', 'Quantile TCN'] for m in ['none', 'half', 'double'])
N['tail_dk'] = f'{dk:.2f}'; N['tail_dv3'] = f'{dv:.2f}'
N['anova_M_int'] = f"{float(N['anova_M_all']) - float(N['anova_M_main']):.1f}"
# probabilistic separate block
pz = [f'{X}/PROB_SEP/quantiles_{p}.npz' for p in ('xgb', 'tcn')]
if all(os.path.exists(p) for p in pz):
    Z0 = dict(np.load(f'{B}/prob/quantiles.npz', allow_pickle=True)); QS = Z0['QS']; yt = Z0['obs_test']
    tr = pd.read_csv(f'{R}/data/jaipur_daily_wind.csv', parse_dates=['date']); tr = tr[tr.date.dt.year.between(2005, 2014)].dropna()
    dts = pd.to_datetime(Z0['test_dates']); clim = np.array([np.quantile(tr.wind_speed_ms[tr.date.dt.month == mm], QS) for mm in range(1, 13)]); Qc = clim[dts.month.values - 1]
    crps = lambda Q, y_: np.mean([2 * np.mean(((y_ < Q[:, k]).astype(float) - a) * (Q[:, k] - y_)) for k, a in enumerate(QS)])
    SP = {}
    for p in pz:
        Z = dict(np.load(p, allow_pickle=True)); yc = Z['obs_cal']
        for name in ['Quantile XGBoost', 'Quantile TCN']:
            if f'{name}|test' not in Z: continue
            Qt, Qca = Z[f'{name}|test'], Z[f'{name}|cal']; s = np.maximum(Qca[:, 0] - yc, yc - Qca[:, -1]); q = np.quantile(s, min(1, 0.9 * (1 + 1 / len(s))))
            lo, hi = Qt[:, 0] - q, Qt[:, -1] + q; c = (yt >= lo) & (yt <= hi); med = Qt[:, list(np.round(QS, 2)).index(0.5)]
            SP[name] = dict(crpss=1 - crps(Qt, yt) / crps(Qc, yt), crps=crps(Qt, yt), cov_raw=np.mean((yt >= Qt[:, 0]) & (yt <= Qt[:, -1])), cov=c.mean(), width=np.mean(hi - lo),
                            cov2020=c[dts.year == 2020].mean(), covJO=c[(dts >= '2020-07-01') & (dts < '2020-11-01')].mean(), med_rmse=rm(med, yt))
    json.dump(SP, open(f'{R}/results/revision/prob_sep.json', 'w'), indent=1, default=float)
    for n, v in SP.items():
        key = 'qx' if 'XGB' in n else 'qt'
        for k, x in v.items(): N[f'sep_{key}_{k}'] = f'{x:.3f}' if k not in ('crpss', 'width') else f'{x:.2f}'
PR = json.load(open(f'{B}/prob_metrics.json'))
def cfv(name, lab, col): return CF.loc[f'{name}|{lab}', col]
if 'sep_qx_cov' in N and 'sep_qt_cov' in N:
    rr = [('CRPS (m s^−1^)', f"{PR['Quantile XGBoost']['CRPS']:.3f}", f"{PR['Quantile TCN']['CRPS']:.3f}"),
          ('CRPS of monthly climatology', f"{PR['Quantile XGBoost']['CRPS_clim']:.3f}", f"{PR['Quantile TCN']['CRPS_clim']:.3f}"),
          ('CRPS skill score', f"{PR['Quantile XGBoost']['CRPSS']:.2f}", f"{PR['Quantile TCN']['CRPSS']:.2f}"),
          ('Coverage of the 90% interval, before calibration', f"{PR['Quantile XGBoost']['cov90_raw']:.3f}", f"{PR['Quantile TCN']['cov90_raw']:.3f}"),
          ('Mean width, before calibration (m s^−1^)', f"{PR['Quantile XGBoost']['width90_raw']:.2f}", f"{PR['Quantile TCN']['width90_raw']:.2f}"),
          ('Coverage, CQR calibrated on 2015–2017', f"{PR['Quantile XGBoost']['cov90_conformal']:.3f}", f"{PR['Quantile TCN']['cov90_conformal']:.3f}"),
          ('Mean width, CQR on 2015–2017 (m s^−1^)', f"{PR['Quantile XGBoost']['width90_conformal']:.2f}", f"{PR['Quantile TCN']['width90_conformal']:.2f}"),
          ('⟦Coverage, separate block (tuned 2015–2016, calibrated 2017)⟧', N['sep_qx_cov'], N['sep_qt_cov']),
          ('⟦Mean width, separate block (m s^−1^)⟧', N['sep_qx_width'], N['sep_qt_width']),
          ('⟦CRPS skill score, separate block⟧', N['sep_qx_crpss'], N['sep_qt_crpss']),
          ('⟦Coverage, adaptive conformal (γ = 0.01)⟧', f"{cfv('Quantile XGBoost','ACI gamma=0.01','cov'):.3f}", f"{cfv('Quantile TCN','ACI gamma=0.01','cov'):.3f}"),
          ('⟦Mean width, adaptive conformal (m s^−1^)⟧', f"{cfv('Quantile XGBoost','ACI gamma=0.01','width'):.2f}", f"{cfv('Quantile TCN','ACI gamma=0.01','width'):.2f}"),
          ('⟦Coverage in Jul–Oct 2020: CQR / adaptive⟧', f"{cfv('Quantile XGBoost','CQR 2015-2017','cov_JulOct2020'):.2f} / {cfv('Quantile XGBoost','ACI gamma=0.01','cov_JulOct2020'):.2f}", f"{cfv('Quantile TCN','CQR 2015-2017','cov_JulOct2020'):.2f} / {cfv('Quantile TCN','ACI gamma=0.01','cov_JulOct2020'):.2f}"),
          ('Brier score, strong-wind day', f"{PR['Quantile XGBoost']['Brier']:.4f}", f"{PR['Quantile TCN']['Brier']:.4f}"),
          ('Brier score, climatology', f"{PR['Quantile XGBoost']['Brier_clim']:.4f}", f"{PR['Quantile TCN']['Brier_clim']:.4f}"),
          ('Brier skill score', f"{PR['Quantile XGBoost']['BSS']:.2f}", f"{PR['Quantile TCN']['BSS']:.2f}"),
          ('RMSE of the median (m s^−1^)', f"{PR['Quantile XGBoost']['median_RMSE']:.3f}", f"{PR['Quantile TCN']['median_RMSE']:.3f}")]
    TB['T_prob'] = '| Measure | Quantile XGBoost | Quantile TCN |\n|---|---|---|\n' + '\n'.join(f'| {a} | {b} | {c} |' for a, b, c in rr)
json.dump(N, open(f'{R}/results/revision/numbers.json', 'w'), indent=1); json.dump(TB, open(f'{R}/results/revision/tables.json', 'w'), indent=1)
print('extra ok', len(N), list(TB))

# ---------- Reviewer 1 additions: valid-time sampling, pre-2005 alternatives
M3 = ['MLR', 'XGBoost', 'TCN']
for t in ['H_VALID', 'F_PRE', 'G_PREADJ']:
    try: P[t] = load(f'{X}/{t}/preds.npz')
    except FileNotFoundError: pass
if all(t in P for t in ['H_VALID', 'F_PRE', 'G_PREADJ']):
    for t in ['H_VALID', 'F_PRE', 'G_PREADJ']:
        for m in M3:
            N[f'{t}_{m}_rmse'] = f3(seedmean(t, m)); N[f'{t}_{m}_bias'] = sg(bias(t, m)); N[f'{t}_{m}_p'] = pf(dmp(t, 'P1R1', m))
    N['H_pmin'] = f"{min(dmp('H_VALID', 'P1R1', m) for m in M3):.2f}"
    N['pre2005_factor'] = f"{json.load(open(f'{X}/G_PREADJ/cost.json'))['pre2005_factor']:.2f}"
    base_rows = TB['T_regime'].split('\n')
    extra = []
    for c, l in [('F_PRE', 'Keep 2000–2004, second regime indicator'), ('G_PREADJ', 'Keep 2000–2004, ratio-adjusted (×' + N['pre2005_factor'] + ')')]:
        extra.append('| ' + l + ' | ' + ' | '.join(f'{f3(seedmean(c, m))} ({sg(bias(c, m))})' for m in M3) + ' | – |')
    TB['T_regime'] = '\n'.join(base_rows + extra)
    # baselines with valid-time sampling
    E2 = pd.read_csv(f'{R}/data/era5_daily_ext_syn.csv', parse_dates=['date']).merge(pd.read_csv(f'{R}/data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'obs'}), on='date')
    tr2 = E2[E2.date.dt.year.between(2005, 2014)].dropna(subset=['obs']); te2 = E2.set_index('date').loc[pd.to_datetime(P['P1R1']['test_dates'])]
    bb, aa = np.polyfit(tr2.ws_valid, tr2.obs, 1)
    N['valid_raw_rmse'] = f3(rm(te2.ws_valid.values, y)); N['valid_raw_r'] = f3(np.corrcoef(te2.ws_valid, y)[0, 1]); N['valid_lc_rmse'] = f3(rm(aa + bb * te2.ws_valid.values, y))
json.dump(N, open(f'{R}/results/revision/numbers.json', 'w'), indent=1); json.dump(TB, open(f'{R}/results/revision/tables.json', 'w'), indent=1)

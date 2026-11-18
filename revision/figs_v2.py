"""Revised figures: 600 dpi, sized for a 17 cm text width, fonts >= 8 pt at print size."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(ROOT); _sys.path.insert(0, _os.path.join(ROOT, 'src'))
import json, sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from scipy import stats
R = ROOT; O = f'{R}/results/main'; F = f'{R}/figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 8.5, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333', 'axes.linewidth': 0.8, 'grid.color': '#dddddd', 'grid.linewidth': 0.6, 'savefig.dpi': 600})
C = ['#1f5fa8', '#d9541e', '#1a9a6c', '#c48a00', '#6b4fbb', '#555555']; INK = '#111111'; INK2 = '#444444'
W = 6.7
def save(fig, name):
    fig.savefig(f'{F}/{name}.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(f'{F}/{name}.tif', dpi=600, bbox_inches='tight', pad_inches=0.03, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig); print('saved', name)
FILL = {'data': ('#dbe8f8', '#1f5fa8'), 'prep': ('#f1f0ec', '#777777'), 'ml': ('#e6e2f8', '#6b4fbb'), 'dl': ('#d8f0e6', '#1a9a6c'),
        'prob': ('#fde3d8', '#d9541e'), 'eval': ('#fdf0cc', '#c48a00'), 'att': ('#fde3d8', '#d9541e')}
def box(ax, x, y, w, h, text, kind, fs=8, bold_first=True):
    f, e = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.003,rounding_size=0.01', fc=f, ec=e, lw=1.0))
    lines = text.split('\n')
    if bold_first:
        ax.text(x + w / 2, y + h / 2, r'$\bf{' + lines[0].replace(' ', r'\ ').replace('&', r'\&') + '}$' + ('\n' + '\n'.join(lines[1:]) if len(lines) > 1 else ''),
                ha='center', va='center', fontsize=fs, color=INK, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.25)
def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.0, shrinkA=0, shrinkB=0))

def fig_workflow():
    fig = plt.figure(figsize=(W, 8.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    for y, lab in [(0.905, 'Data'), (0.73, 'Preprocessing'), (0.555, 'Design'), (0.36, 'Models'), (0.18, 'Evaluation'), (0.04, 'Analyses')]:
        ax.text(0.025, y + 0.045, lab, ha='center', va='center', fontsize=8.5, fontweight='bold', color=INK2, rotation=90)
    box(ax, 0.06, 0.885, 0.29, 0.1, 'ERA5 reanalysis\nhourly, 0.25°, 2000–2020\n8 surface variables,\nnearest grid point', 'data', 7.8)
    box(ax, 0.365, 0.885, 0.29, 0.1, 'NOAA ISD, VIJP\nJaipur airport, 3-hourly\ndaily mean, ≥ 7 of 8 reports', 'data', 7.8)
    box(ax, 0.67, 0.885, 0.31, 0.1, 'IMDAA reanalysis\n12 km, daily 10 m speed\nsecond reference, extra input', 'data', 7.8)
    box(ax, 0.06, 0.70, 0.445, 0.155, 'Daily predictors from hourly ERA5\nOriginal set: daily means of 8 variables\nPhysics-based set: + hourly-mean speed,\nday/night speed, max and SD of speed,\nBLH max/min, t2m range, direction,\nsteadiness, day of year (21 variables)', 'prep', 7.6)
    box(ax, 0.52, 0.70, 0.46, 0.155, 'Homogeneity of the station record\nmonthly log ratio station / reanalysis\nPettitt and SNHT; AR(1) significance\nreferences: ERA5 and IMDAA;\nfull record and training years only\nregime indicator = 1 from Feb 2011', 'prep', 7.6)
    box(ax, 0.06, 0.53, 0.92, 0.13, 'Factorial design and protocol\n{original, physics-based} predictors × {without, with} regime indicator; intermediate predictor sets\ntrain 2005–2014 · validation 2015–2017 · test 2018 to 3 Dec 2020 · 7-day window\n30 random settings per model (validation RMSE), then 10 seeds of the selected setting', 'prep', 7.6)
    box(ax, 0.06, 0.335, 0.205, 0.155, 'Baselines\nraw ERA5\n(vector- and\nhourly-mean speed)\nlinear correction\nquantile mapping', 'prep', 7.4)
    box(ax, 0.28, 0.335, 0.19, 0.155, 'Statistical / ML\nmultiple linear\nregression\nXGBoost', 'ml', 7.4)
    box(ax, 0.485, 0.335, 0.235, 0.155, 'Deep learning\nLSTM, BiLSTM,\nattention-BiLSTM,\nTCN, attention-TCN,\nTransformer', 'dl', 7.4)
    box(ax, 0.735, 0.335, 0.245, 0.155, 'Probabilistic\nquantile XGBoost,\nquantile TCN\nconformal 90% interval\n(static and adaptive)', 'prob', 7.4)
    box(ax, 0.06, 0.155, 0.92, 0.135, 'Evaluation on the test period against the station\nRMSE, MAE, bias, r, R², skill score · block-bootstrap CIs · Diebold–Mariano tests\nANOVA of RMSE with interactions · CIs of pairwise RMSE differences\nCRPS, interval coverage, PIT, Brier score · Weibull k and c, mean v³, energy pattern factor', 'eval', 7.6)
    labs = ['Predictors, regime\nand architecture', 'Seasons, years,\nstrong-wind days', 'Predictor\nimportance', 'Probabilistic\nskill', 'Energy-relevant\ndistribution', 'IMDAA and\ncomputing cost']
    for i, l in enumerate(labs): box(ax, 0.06 + i * 0.1537, 0.025, 0.142, 0.09, l, 'prep', 7.4, bold_first=False)
    arrow(ax, 0.205, 0.885, 0.205, 0.855); arrow(ax, 0.51, 0.885, 0.70, 0.855); arrow(ax, 0.825, 0.885, 0.825, 0.855)
    arrow(ax, 0.28, 0.70, 0.28, 0.66); arrow(ax, 0.75, 0.70, 0.75, 0.66)
    for x in (0.16, 0.375, 0.60, 0.855): arrow(ax, x, 0.53, x, 0.49); arrow(ax, x, 0.335, x, 0.29)
    arrow(ax, 0.52, 0.155, 0.52, 0.115)
    save(fig, 'Figure1_workflow')

def fig_arch():
    fig = plt.figure(figsize=(W, 5.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 3); ax.set_ylim(0, 2); ax.axis('off')
    spec = {'(a) LSTM': [('Input: 7 days × F predictors', 'data'), ('Stacked LSTM layers', 'dl'), ('Hidden state of the last day', 'dl'), ('Dense (ReLU) + dropout', 'prep'), ('Output: wind speed, day t', 'att')],
            '(b) BiLSTM': [('Input: 7 days × F predictors', 'data'), ('Bidirectional LSTM layers', 'dl'), ('Last forward + first\nbackward state', 'dl'), ('Dense (ReLU) + dropout', 'prep'), ('Output: wind speed, day t', 'att')],
            '(c) Attention-BiLSTM': [('Input: 7 days × F predictors', 'data'), ('Bidirectional LSTM layers', 'dl'), ('Day-wise features h1 … h7', 'dl'), ('Additive attention → context', 'eval'), ('Dense (ReLU) + dropout', 'prep'), ('Output: wind speed, day t', 'att')],
            '(d) TCN': [('Input: 7 days × F predictors', 'data'), ('Residual blocks, causal\ndilated convolutions', 'ml'), ('Feature of the last day', 'ml'), ('Dense (ReLU) + dropout', 'prep'), ('Output: wind speed, day t\n(19 quantiles, quantile TCN)', 'att')],
            '(e) Attention-TCN': [('Input: 7 days × F predictors', 'data'), ('Residual blocks, causal\ndilated convolutions', 'ml'), ('Day-wise features h1 … h7', 'ml'), ('Additive attention → context', 'eval'), ('Dense (ReLU) + dropout', 'prep'), ('Output: wind speed, day t', 'att')],
            '(f) Transformer encoder': [('Input: 7 days × F predictors', 'data'), ('Linear embedding +\npositional encoding', 'eval'), ('Multi-head self-attention\n+ feed-forward layers', 'eval'), ('Representation of last day', 'eval'), ('Dense (ReLU) + dropout', 'prep'), ('Output: wind speed, day t', 'att')]}
    for i, (title, items) in enumerate(spec.items()):
        cx, top = (i % 3) + 0.5, 2 - (i // 3) * 1.0
        ax.add_patch(FancyBboxPatch((cx - 0.48, top - 0.975), 0.96, 0.95, boxstyle='round,pad=0,rounding_size=0.02', fc='#fafafa', ec='#cccccc', lw=0.8))
        ax.text(cx, top - 0.07, title, ha='center', va='center', fontsize=9, fontweight='bold', color=INK)
        n = len(items); step = 0.8 / n; h = step * 0.72
        for j, (t, k) in enumerate(items):
            y = top - 0.14 - j * step - h; f, e = FILL[k]
            ax.add_patch(FancyBboxPatch((cx - 0.42, y), 0.84, h, boxstyle='round,pad=0,rounding_size=0.015', fc=f, ec=e, lw=0.9))
            ax.text(cx, y + h / 2, t, ha='center', va='center', fontsize=7.6, color=INK, linespacing=1.1)
            if j < n - 1: arrow(ax, cx, y, cx, y - (step - h))
    save(fig, 'Figure3_architectures')

def fig_study():
    import json as js
    def polys(geom):
        if geom['type'] == 'Polygon': return [geom['coordinates'][0]]
        return [p[0] for p in geom['coordinates']]
    import urllib.request; os.makedirs(f'{R}/data/ne', exist_ok=True)
    for nm in ('ne_10m_admin_0_countries', 'ne_10m_admin_1_states_provinces'):
        if not os.path.exists(f'{R}/data/ne/{nm}.geojson'):
            urllib.request.urlretrieve(f'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{nm}.geojson', f'{R}/data/ne/{nm}.geojson')
    A0 = js.load(open(f'{R}/data/ne/ne_10m_admin_0_countries.geojson')); A1 = js.load(open(f'{R}/data/ne/ne_10m_admin_1_states_provinces.geojson'))
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw=dict(width_ratios=[1, 1.05]))
    a = ax[0]; a.set_facecolor('#e8f1fa')
    for f in A0['features']:
        if f['properties'].get('ADM0_A3') in ('IND',): continue
        for p in polys(f['geometry']):
            p = np.array(p)
            if p[:, 0].max() < 60 or p[:, 0].min() > 100 or p[:, 1].max() < 0 or p[:, 1].min() > 40: continue
            a.add_patch(Polygon(p, fc='#eeeeee', ec='#bbbbbb', lw=0.4))
    for f in A1['features']:
        if f['properties'].get('adm0_a3') != 'IND': continue
        raj = f['properties'].get('name') == 'Rajasthan'
        for p in polys(f['geometry']):
            a.add_patch(Polygon(np.array(p), fc='#fbd9c4' if raj else 'white', ec='#d9541e' if raj else '#aaaaaa', lw=1.0 if raj else 0.35, zorder=3 if raj else 2))
    a.plot(75.81, 26.82, marker='*', ms=11, color=INK, mec='white', zorder=5); a.text(77.3, 25.2, 'Jaipur', fontsize=8.5, zorder=5)
    a.text(69.5, 28.6, 'Rajasthan', color='#b23c0b', fontsize=8.5, fontweight='bold', zorder=5)
    a.set_xlim(66, 98); a.set_ylim(6, 37.5); a.set_aspect('equal'); a.set_xlabel('Longitude (°E)'); a.set_ylabel('Latitude (°N)')
    a.set_title('(a) Location of the study site', loc='left'); [s.set_visible(True) for s in a.spines.values()]
    b = ax[1]
    b.add_patch(Rectangle((75.625, 26.625), 0.25, 0.25, fc='#e3eefa', ec='#1f5fa8', lw=1.2))
    for v in (75.625, 75.875): b.axvline(v, color='#1f5fa8', lw=0.7, ls='--', alpha=0.6)
    for v in (26.625, 26.875): b.axhline(v, color='#1f5fa8', lw=0.7, ls='--', alpha=0.6)
    st = (75.812, 26.824); er = (75.75, 26.75); im = (75.72, 26.76)
    b.plot(*zip(er, st), ls=':', color='#1f5fa8', lw=1.3); b.plot(*zip(im, st), ls=':', color='#d9541e', lw=1.3)
    b.plot(*er, 's', ms=9, color='#1f5fa8', mec='white', label='ERA5 grid point (26.75°N, 75.75°E)')
    b.plot(*im, 'D', ms=8, color='#d9541e', mec='white', label='IMDAA grid point (26.76°N, 75.72°E)')
    b.plot(*st, '^', ms=9, color=INK, mec='white', label='VIJP station (26.82°N, 75.81°E)')
    b.text(75.79, 26.775, '10.3 km', color='#1f5fa8', fontsize=8.5); b.text(75.67, 26.81, '11.6 km', color='#d9541e', fontsize=8.5)
    b.text(75.635, 26.64, 'ERA5 grid cell (0.25°)', color='#1f5fa8', fontsize=8)
    dlon = 5 / (111.32 * np.cos(np.radians(26.6))); b.plot([75.52, 75.52 + dlon], [26.57, 26.57], color=INK, lw=2.5); b.text(75.52, 26.578, '5 km', fontsize=8)
    b.set_xlim(75.5, 76.0); b.set_ylim(26.55, 26.97); b.set_xlabel('Longitude (°E)'); b.set_ylabel('Latitude (°N)')
    b.legend(loc='upper left', fontsize=7.4, frameon=True, framealpha=0.95); b.set_title('(b) Grid points and station', loc='left'); [s.set_visible(True) for s in b.spines.values()]
    fig.text(0.01, -0.01, 'Boundaries: Natural Earth.', fontsize=7, color=INK2)
    plt.tight_layout(); save(fig, 'Figure2_study_area')

def fig_homog():
    E = pd.read_csv(f'{R}/data/era5_daily_ext.csv', parse_dates=['date'])
    S = pd.read_csv(f'{R}/data/jaipur_daily_wind.csv', parse_dates=['date']).rename(columns={'wind_speed_ms': 'obs'})
    D = E.merge(S, on='date', how='left').set_index('date')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9))
    for col, lab, c, mk in [('ws_scalar', 'Station / ERA5 (hourly-mean speed)', C[0], 'o'), ('imdaa_average_speed', 'Station / IMDAA', C[1], 's')]:
        cnt = D.obs.resample('YS').count(); Y = D[['obs', col]].resample('YS').mean(); Y = Y[cnt >= 200]
        q = (Y.obs / Y[col]).reindex(pd.date_range('2000', '2020', freq='YS'))
        ax[0].plot(q.index.year, q.values, marker=mk, ms=4.5, color=c, lw=1.4, label=lab)
        cm = D.obs.resample('MS').count(); M = D[['obs', col]].resample('MS').mean(); M = M[cm >= 20]
        qm = (M.obs / M[col]).loc['2005':].reindex(pd.date_range('2005-01', '2020-12', freq='MS'))
        ax[1].plot(qm.index, qm.values, color=c, lw=1.0, label=lab)
    ax[0].axvline(2004.5, color=INK2, ls='--', lw=1); ax[0].axvline(2011.08, color='#b2182b', ls='--', lw=1)
    ax[0].text(2000.2, 0.66, 'level shift\nbefore 2005', fontsize=7.8, color=INK2); ax[0].text(2011.3, 1.17, 'break,\nFeb 2011', fontsize=7.8, color='#b2182b')
    ax[0].set_xticks(range(2000, 2021, 4)); ax[0].set_xlim(1999.5, 2020.5); ax[0].set_ylabel('Annual mean ratio'); ax[0].set_xlabel('Year')
    ax[0].set_title('(a) Annual ratio', loc='left'); ax[0].grid(alpha=1); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07))
    ax[1].axvline(pd.Timestamp('2011-02-01'), color='#b2182b', ls='--', lw=1); ax[1].axvspan(pd.Timestamp('2020-07-01'), pd.Timestamp('2020-10-31'), color='#f4a582', alpha=0.5, lw=0)
    ax[1].set_ylabel('Monthly mean ratio'); ax[1].set_xlabel('Year'); ax[1].set_title('(b) Monthly ratio from 2005', loc='left'); ax[1].grid(alpha=1)
    import matplotlib.dates as md; ax[1].xaxis.set_major_locator(md.YearLocator(3, month=1, day=1)); ax[1].set_xticks([pd.Timestamp(f'{y}-01-01') for y in range(2005, 2021, 3)]); ax[1].xaxis.set_major_formatter(md.DateFormatter('%Y'))
    plt.tight_layout(); save(fig, 'Figure4_homogeneity')

def fig_importance():
    I = pd.read_csv(f'{O}/importance.csv'); G = I[I.kind == 'group']
    order = G[G.model == 'TCN'].sort_values('mean').predictor.tolist()
    fig, ax = plt.subplots(figsize=(W, 3.4)); y = np.arange(len(order))
    for j, m in enumerate(['XGBoost', 'TCN', 'BiLSTM']):
        s = G[G.model == m].set_index('predictor').reindex(order)
        ax.barh(y + (j - 1) * 0.26, s['mean'], height=0.25, xerr=s['std'], color=C[j], label=m, error_kw=dict(lw=0.7, ecolor='#333333', capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8); ax.set_xlabel('Increase in test RMSE when shuffled (m s$^{-1}$)')
    ax.grid(axis='x', alpha=1); ax.legend(frameon=False, loc='lower right')
    plt.tight_layout(); save(fig, 'Figure6_importance')


def fig_factorial():
    T = pd.read_csv(f'{O}/table_factorial.csv'); X = f'{R}/results/revision/exp'
    order = ['MLR', 'XGBoost', 'LSTM', 'BiLSTM', 'Attention-BiLSTM', 'TCN', 'Attention-TCN', 'Transformer']
    cfg = [('P0R0', 'Original predictors', 'o'), ('P0R1', 'Original + regime indicator', 's'), ('P1R0', 'Physics-based predictors', '^'), ('P1R1', 'Physics-based + regime indicator', 'D')]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]; yy = np.arange(len(order))[::-1]
    for j, (c, lab, mk) in enumerate(cfg):
        s = T[T.config == c].set_index('model').reindex(order)
        a.errorbar(s.RMSE, yy + (1.5 - j) * 0.18, xerr=s.RMSE_sd, fmt=mk, ms=5, color=C[j], elinewidth=0.9, capsize=0, mec='white', mew=0.6, label=lab)
    for k, ls in [('Raw ERA5 (hourly-mean speed)', '--'), ('ERA5 hourly-mean speed + linear correction', ':')]:
        a.axvline(T[(T.config == 'Baseline') & (T.model == k)].RMSE.iloc[0], color=INK2, lw=1, ls=ls)
    a.set_yticks(yy); a.set_yticklabels(order); a.set_xlabel('Test RMSE (m s$^{-1}$), mean ± SD over 10 seeds'); a.grid(axis='x', alpha=1)
    a.set_title('(a) Factorial design', loc='left'); a.set_ylim(-1.3, len(order) - 0.4); a.text(0.781, -1.2, 'raw ERA5', fontsize=7.5, color=INK2, ha='right', va='bottom'); a.text(0.707, -1.2, 'ERA5 + LC', fontsize=7.5, color=INK2, ha='right', va='bottom')
    h, l = a.get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    b = ax[1]; steps = ['P0R1', 'A_WS', 'B_SPD', 'C_NOSPD', 'P1R1']; labs = ['Original', 'A: + WS', 'B: + 5 speeds', 'C: + non-speed', 'Physics-based']
    def rmse_of(c, m):
        p = f'{O}/{c}/preds.npz' if c.startswith('P') else f'{X}/{c}/preds.npz'; d = np.load(p, allow_pickle=True); yv = d['obs_test']
        return np.mean([np.sqrt(np.mean((q - yv) ** 2)) for q in d[f'{m}_test']])
    for j, m in enumerate(['MLR', 'XGBoost', 'TCN', 'BiLSTM']):
        b.plot(np.arange(5) + (j - 1.5) * 0.08, [rmse_of(c, m) for c in steps], marker='osD^'[j], color=[C[5], C[0], C[2], C[4]][j], lw=0, ms=6, mec='white', label=m)
    b.set_xticks(range(5)); b.set_xticklabels(labs, rotation=30, ha='right', fontsize=8); b.set_ylabel('Test RMSE (m s$^{-1}$)')
    b.set_title('(b) Intermediate predictor sets\n(all with regime indicator)', loc='left'); b.grid(alpha=1); b.legend(frameon=False, fontsize=7.8)
    plt.tight_layout(); save(fig, 'Figure5_factorial')

def fig_prob():
    Z = dict(np.load(f'{O}/prob/quantiles.npz', allow_pickle=True)); obs = Z['obs_test']; thr = json.load(open(f'{O}/meta.json'))['strong_threshold']
    dt = pd.to_datetime(Z['test_dates']); yv = Z['obs_val']
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for j, name in enumerate(['Quantile XGBoost', 'Quantile TCN']):
        pit = np.load(f'{O}/prob/pit_{name.replace(" ", "_")}.npy'); h, e = np.histogram(pit, bins=10, range=(0, 1)); w = 0.04
        ax[0].bar(e[:-1] + 0.05 + (j - 0.5) * w, h / len(pit) * 10, width=w - 0.004, color=C[j], label=name.replace('Quantile ', 'Q-'))
        pex = np.load(f'{O}/prob/pex_{name.replace(" ", "_")}.npy'); sw = obs > thr
        bins = np.array([0, 0.05, 0.15, 0.3, 0.5, 1.0]); idx = np.digitize(pex, bins) - 1; xs, ys = [], []
        for k in range(len(bins) - 1):
            m = idx == k
            if m.sum() >= 5: xs.append(pex[m].mean()); ys.append(sw[m].mean())
        ax[1].plot(xs, ys, marker='os'[j], color=C[j], lw=1.5, ms=5, mec='white', label=name.replace('Quantile ', 'Q-'))
        Qv, Qt = Z[f'{name}|val'], Z[f'{name}|test']; sv = np.maximum(Qv[:, 0] - yv, yv - Qv[:, -1]); st = np.maximum(Qt[:, 0] - obs, obs - Qt[:, -1])
        q = np.quantile(sv, 0.9 * (1 + 1 / len(sv))); cs = ((obs >= Qt[:, 0] - q) & (obs <= Qt[:, -1] + q)).astype(float)
        a_ = 0.1; sc = list(sv); ca = np.zeros(len(obs))
        for t in range(len(obs)):
            lev = 1 - a_; qq = np.quantile(sc, lev) if lev < 1 else max(sc) + 1
            ca[t] = float(Qt[t, 0] - qq <= obs[t] <= Qt[t, -1] + qq); a_ += 0.01 * (0.1 - (1 - ca[t])); sc.append(st[t])
        r1 = pd.Series(cs, index=dt).rolling(90).mean(); r2 = pd.Series(ca, index=dt).rolling(90).mean()
        ax[2].plot(r1.index, r1, color=C[j], lw=1.0, ls='--'); ax[2].plot(r2.index, r2, color=C[j], lw=1.3)
    ax[0].axhline(1, color=INK2, lw=1, ls='--'); ax[0].set_xlabel('PIT'); ax[0].set_ylabel('Relative frequency'); ax[0].set_title('(a) PIT histogram', loc='left'); ax[0].legend(frameon=False, fontsize=7.5)
    ax[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls='--'); ax[1].set_xlim(0, 0.8); ax[1].set_ylim(0, 0.8); ax[1].set_xlabel('Forecast probability'); ax[1].set_ylabel('Observed frequency')
    ax[1].set_title('(b) Reliability, strong wind', loc='left')
    ax[2].axhline(0.9, color=INK2, lw=0.8); ax[2].set_ylim(0.4, 1.02); ax[2].set_ylabel('90-day coverage'); ax[2].set_title('(c) 90% interval coverage', loc='left')
    from matplotlib.lines import Line2D
    ax[2].legend([Line2D([], [], color=INK2, ls='--'), Line2D([], [], color=INK2)], ['CQR (static)', 'ACI, γ = 0.01'], frameon=False, fontsize=7.5, loc='lower left')
    import matplotlib.dates as md; ax[2].xaxis.set_major_locator(md.YearLocator()); ax[2].xaxis.set_major_formatter(md.DateFormatter('%Y'))
    for a in ax: a.grid(alpha=1)
    plt.tight_layout(); save(fig, 'Figure7_probabilistic')

def fig_energy():
    EC = pd.read_csv(f'{R}/results/revision/energy_ci.csv').set_index('series')
    fig = plt.figure(figsize=(W, 5.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.15, 1])
    ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    v = np.linspace(0, 7, 300)
    for k, lab, col, ls in [('Station (observed)', 'Station', INK, '-'), ('Raw ERA5, hourly-mean speed', 'Raw ERA5', C[1], '--'),
                            ('XGBoost, final set-up', 'XGBoost', C[0], '-'), ('Quantile XGBoost, 50 sampled members', 'Q-XGBoost, sampled', C[2], '-')]:
        ax[0].plot(v, stats.weibull_min.pdf(v, EC.loc[k, 'k'], scale=EC.loc[k, 'c']), color=col, ls=ls, lw=1.6, label=f"{lab} (k = {EC.loc[k, 'k']:.2f})")
    ax[0].set_xlabel('Daily mean wind speed (m s$^{-1}$)'); ax[0].set_ylabel('Weibull density'); ax[0].legend(frameon=False, fontsize=8, loc='upper right'); ax[0].set_ylim(0, 0.5); ax[0].set_xlim(0, 7)
    ax[0].set_title('(a) Fitted Weibull distributions', loc='left')
    sel = [('Raw ERA5, hourly-mean speed', 'ERA5, hourly-mean', C[1]), ('ERA5 + linear correction', 'ERA5 + lin. corr.', C[1]), ('XGBoost, final set-up', 'XGBoost', C[0]), ('TCN, final set-up', 'TCN', C[0]),
           ('XGBoost + variance inflation', 'XGBoost + var. infl.', C[4]), ('TCN + variance inflation', 'TCN + var. infl.', C[4]),
           ('Quantile XGBoost, 50 sampled members', 'Q-XGBoost, sampled', C[2]), ('Quantile TCN, 50 sampled members', 'Q-TCN, sampled', C[2])]
    yy = np.arange(len(sel))[::-1]
    for i, (k, l, c) in enumerate(sel):
        for a, col, ref in [(ax[1], 'v3_ratio', 1.0), (ax[2], 'epf', EC.loc['Station (observed)', 'epf'])]:
            a.errorbar(EC.loc[k, col], yy[i], xerr=[[EC.loc[k, col] - EC.loc[k, col + '_lo']], [EC.loc[k, col + '_hi'] - EC.loc[k, col]]], fmt='o', color=c, ms=4.5, elinewidth=1, capsize=2)
    ax[1].axvline(1, color=INK, lw=1); ax[1].set_yticks(yy); ax[1].set_yticklabels([s[1] for s in sel], fontsize=7.8); ax[1].set_xlabel('Mean $v^3$ / observed'); ax[1].set_title('(b) Mean cubed speed', loc='left')
    eo = EC.loc['Station (observed)']; ax[2].axvspan(eo.epf_lo, eo.epf_hi, color='#dddddd', lw=0); ax[2].axvline(eo.epf, color=INK, lw=1)
    ax[2].set_yticks(yy); ax[2].set_yticklabels([]); ax[2].set_xlabel('Energy pattern factor'); ax[2].set_title('(c) Distribution shape', loc='left')
    for a in ax[1:]: a.grid(axis='x', alpha=1)
    plt.tight_layout(); save(fig, 'Figure8_energy')

if __name__ == '__main__':
    for n in sys.argv[1:]: globals()['fig_' + n]()

import numpy as np
from scipy import stats


def dm_test(loss_a, loss_b, h=1):
    """Diebold-Mariano test (two-sided) on loss differential d = loss_a - loss_b,
    Newey-West long-run variance (Bartlett, lag = floor(n^(1/3))) and the
    Harvey-Leybourne-Newbold small-sample correction. Negative stat: A better."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    lag = int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


def baseline_preds(df, train_years=(2000, 2014)):
    """E1 baselines as functions of ERA5 speed, fitted on the given training years."""
    tr = df[df.date.dt.year.between(*train_years)]
    x, y = tr.era5_speed.values, tr.imdaa_average_speed.values
    b, a = np.polyfit(x, y, 1)
    q = np.linspace(0.01, 0.99, 99)
    qx, qy = np.quantile(x, q), np.quantile(y, q)
    e = df.era5_speed.values
    return {'Interpolated ERA5': e, 'Linear bias correction': a + b * e,
            'Quantile mapping': np.interp(e, qx, qy)}

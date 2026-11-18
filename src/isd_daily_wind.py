"""
Daily average wind speed from NOAA ISD Global Hourly CSV files.

Usage:
    python isd_daily_wind.py <folder_or_files...> -o jaipur_daily_wind.csv

Output: two columns -> date, wind_speed_ms
- Uses the 8 three-hourly times (00, 03, ..., 21 UTC).
- Skips missing (9999) and bad-quality speeds. Calm = 0 m/s.
- A day is kept only if it has at least MIN_HOURS of the 8 times; otherwise blank.
"""
import argparse, glob, os
import pandas as pd

MIN_HOURS = 7
GOOD_QC = {'0', '1', '4', '5', '9'}


def read_one(path):
    df = pd.read_csv(path, usecols=['DATE', 'REPORT_TYPE', 'WND'], dtype=str)
    w = df['WND'].str.split(',', expand=True)
    df['spd'] = pd.to_numeric(w[3], errors='coerce') / 10.0
    df['t'] = pd.to_datetime(df['DATE'])
    df = df[(df['spd'] < 999) & w[4].isin(GOOD_QC)]
    df = df[(df['t'].dt.minute == 0) & (df['t'].dt.hour % 3 == 0)]
    # one value per time: prefer SYNOP (FM-12) over METAR
    df = df.assign(pref=(df['REPORT_TYPE'].str.strip() != 'FM-12').astype(int))
    df = df.sort_values(['t', 'pref']).drop_duplicates('t')
    return df[['t', 'spd']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--out', default='jaipur_daily_wind.csv')
    a = ap.parse_args()

    files = []
    for p in a.inputs:
        files += sorted(glob.glob(os.path.join(p, '*.csv'))) if os.path.isdir(p) else [p]

    obs = pd.concat([read_one(f) for f in files])
    g = obs.groupby(obs['t'].dt.floor('D'))['spd']
    daily = g.mean().where(g.count() >= MIN_HOURS)

    days = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    out = daily.reindex(days).round(2).rename('wind_speed_ms')
    out.index.name = 'date'
    out.index = out.index.date
    out.to_csv(a.out, index_label='date')

    print(f'Days: {len(out)}  with value: {out.notna().sum()}  blank: {out.isna().sum()}')


if __name__ == '__main__':
    main()

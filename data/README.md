# Data

All inputs are public. Only the derived daily station series is stored here.

| File | Source | How to obtain |
|---|---|---|
| `jaipur_daily_wind.csv` (included) | NOAA Integrated Surface Database, station 42348099999 (Jaipur, VIJP) | Built with `src/isd_daily_wind.py` from `raw/isd/*.csv`: mean of the eight synoptic reports (00–21 UTC), at least 7 valid reports per day, calm = 0 m/s, failed quality flags dropped |
| `raw/isd/*.csv` | NOAA NCEI Global Hourly | `bash data/download_isd.sh` (2000–2020) |
| ERA5 hourly CSV | Copernicus Climate Data Store, ERA5 hourly data on single levels (https://doi.org/10.24381/cds.adbb2d47) | Nearest grid point 26.75°N, 75.75°E, 2000-01-01 00 UTC to 2020-12-03 23 UTC; variables u10, v10, d2m, t2m, blh, msl, sp, tp. Columns expected by `src/prep.py`: `valid_time, u10, v10, d2m, t2m, blh, msl, sp, tp, imdaa_average_speed` |
| IMDAA daily 10 m wind speed | NCMRWF (https://rds.ncmrwf.gov.in) | Nearest grid point 26.76°N, 75.72°E; merged into the ERA5 CSV as `imdaa_average_speed` (one daily value per UTC day) |

`python src/prep.py <era5_hourly.csv>` writes `data/era5_daily_ext.csv`, the daily predictor file used by all experiments.

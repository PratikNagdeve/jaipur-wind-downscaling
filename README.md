# Jaipur wind speed downscaling

Code, derived data and results for the article

> Nagdeve P, Kumari J and Wadhvani R. *Predictor construction and station homogeneity matter more than model architecture in reanalysis-to-station wind speed downscaling: a probabilistic case study for wind energy at Jaipur, India.* Wind Engineering (under review, WIE-26-0353).

Daily ERA5 predictors are downscaled to observed wind speed at Jaipur airport (NOAA ISD station 42348099999) with eight models: MLR, XGBoost, LSTM, BiLSTM, attention-BiLSTM, TCN, attention-TCN and a Transformer encoder. A factorial design crosses the predictor set with the treatment of a 2011 break in the station record. Every model gets the same tuning budget (30 random settings) and ten random seeds. Quantile XGBoost and a quantile TCN with conformal calibration provide probabilistic output, which is evaluated on energy-relevant measures.

## Repository layout

```
data/        derived station series, data instructions, ISD download script
src/         main pipeline (predictors, factorial design, homogeneity, probabilistic models, evaluation)
revision/    analyses added in the revision (intermediate predictor sets, break treatments,
             ANOVA, pairwise CIs, homogeneity robustness, adaptive conformal, energy CIs, figures)
results/     main/      outputs of the factorial design and probabilistic models (predictions, tables)
             revision/  outputs of the revision analyses
figures/     the eight manuscript figures (PNG, 600 dpi)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root. Training uses CPUs only (one thread per process).

## Reproducing the results

1. **Data.** Follow `data/README.md`. The derived station series `data/jaipur_daily_wind.csv` is included; to rebuild it, run `bash data/download_isd.sh` and `python src/isd_daily_wind.py data/raw/isd -o data/jaipur_daily_wind.csv`.
2. **Daily predictors.** `python src/prep.py <era5_hourly.csv>` writes `data/era5_daily_ext.csv`. This file contains the hourly-mean speed and the day/night, maximum and SD speeds, the BLH range, the t2m range, direction and steadiness.
3. **Factorial design** (4 configurations × 8 models): `bash src/run_all.sh` writes `results/main/P{0,1}R{0,1}/`.
4. **Homogeneity, probabilistic models, evaluation:**
   ```bash
   python src/homog.py
   PART=xgb python src/prob.py && PART=tcn python src/prob.py
   python src/analysis.py
   python src/importance.py
   ```
5. **Revision analyses:**
   ```bash
   for t in A_WS B_SPD C_NOSPD D_POST E_RATIO; do python revision/run_config2.py $t; done   # intermediate sets, break treatments
   python revision/prob2.py                                    # separate calibration year
   python revision/anova.py; python revision/pairwise.py       # ANOVA with interactions; pairwise RMSE-difference CIs
   python revision/homog2.py                                   # second reference, training years, AR(1) significance
   python revision/isd_check.py; python revision/isd_calm.py  # ISD report types, units, calm reports (needs data/raw/isd)
   python revision/conformal.py                                # 2017-only CQR and adaptive conformal inference
   python revision/era5_series.py; python revision/energy.py; python revision/energy_paired.py
   python revision/collect.py                                  # all manuscript numbers in one file
   python revision/figs_v2.py workflow study arch homog factorial importance prob energy
   ```
   Steps 3–5 can be skipped. `results/` already holds the saved predictions, so the revision analyses and figures can be rerun from them directly.

Set-up names: P0 = original 8 daily-mean variables; P1 = physics-based set (21 variables); R1 = regime indicator (1 from February 2011). A, B and C are the intermediate predictor sets; D is training on post-break years only; E is the ratio-adjusted pre-break target.

Results can differ slightly between hardware and library versions; the saved predictions in `results/` are the ones reported in the article.

## Data licences

ERA5: Copernicus licence (Hersbach et al., 2020). IMDAA: NCMRWF data policy. ISD: NOAA NCEI. The code is released under the MIT licence.

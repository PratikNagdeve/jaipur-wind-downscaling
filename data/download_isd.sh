#!/bin/bash
# Download NOAA ISD (Global Hourly) for Jaipur VIJP, station 42348099999, 2000-2020, into data/raw/isd/
cd "$(dirname "$0")"; mkdir -p raw/isd
for y in $(seq 2000 2020); do
  curl -sSf -o raw/isd/42348099999_$y.csv "https://www.ncei.noaa.gov/data/global-hourly/access/$y/42348099999.csv" || echo "no file for $y"
done

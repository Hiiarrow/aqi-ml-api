"""
Audit cached raw CPCB 2017-2023 Parquets (raw_cpcb/cpcb_6cities_*.parquet)
to inspect every station's PM2.5 coverage, units/anomalies, cap behavior,
and 15-min vs hourly cadence before exporting OpenCity CSVs.
"""
import glob
import os

import numpy as np
import pandas as pd

files = sorted(glob.glob(os.path.join("raw_cpcb", "cpcb_6cities_*.parquet")))
assert files, "No cpcb_6cities_*.parquet files found"
dfs = []
for f in files:
    d = pd.read_parquet(f, columns=["City", "Station Name", "Timestamp", "PM2.5 (µg/m³)", "PM10 (µg/m³)"])
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

# Strip fake UTC label -> naive IST wall-clock datetime (empirically verified via solar radiation)
ts = pd.to_datetime(df["Timestamp"])
if ts.dt.tz is not None:
    ts = ts.dt.tz_localize(None)
df["ts"] = ts
df["hour"] = ts.dt.floor("h")

print(f"Total 15-min rows (2017-2023, 6 cities): {len(df):,}")
pm = df["PM2.5 (µg/m³)"]
print("Raw 15-min PM2.5 sanity:",
      f"nonnull={pm.notna().sum():,}",
      f"<=0={(pm<=0).sum()}",
      f"<0={(pm<0).sum()}",
      f"==500={(pm==500).sum()}",
      f">500={(pm>500).sum()}",
      f">=985={(pm>=985).sum()}",
      f"==1000={(pm==1000).sum()}",
      f"max={pm.max():.2f}")

# Check negative values if any
if (pm < 0).sum() > 0:
    print("WARNING negative 15-min values sample:", df[pm < 0][["City", "Station Name", "ts", "PM2.5 (µg/m³)"]].head(10))

# Filter physical non-negative for hourly rollup audit
valid_15 = df[pm.notna() & (pm >= 0)].copy()
hr = valid_15.groupby(["City", "Station Name", "hour"], as_index=False).agg(
    pm25=("PM2.5 (µg/m³)", "mean"),
    n_sub=("PM2.5 (µg/m³)", "count"),
)

rows = []
for (city, st), g in hr.groupby(["City", "Station Name"]):
    x = g["pm25"].values
    t0, t1 = g["hour"].min(), g["hour"].max()
    span_y = (t1 - t0).days / 365.25
    rows.append(dict(
        city=city, station=st,
        start=t0.strftime("%Y-%m-%d"), end=t1.strftime("%Y-%m-%d"),
        span_y=round(span_y, 2),
        hours=len(x),
        med_sub=round(float(g["n_sub"].median()), 1),
        mean=round(float(x.mean()), 1),
        median=round(float(np.median(x)), 1),
        p99=round(float(np.percentile(x, 99)), 1),
        max=round(float(x.max()), 1),
        gt500=int((x > 500).sum()),
        eq500=int((x == 500).sum()),
        ge985=int((x >= 985).sum()),
    ))

rep = pd.DataFrame(rows).sort_values(["city", "hours"], ascending=[True, False])
os.makedirs("fetch", exist_ok=True)
rep.to_csv("fetch/cpcb_station_audit_2017_2023.csv", index=False)
print("\nSaved fetch/cpcb_station_audit_2017_2023.csv")
print(rep.to_string(index=False))

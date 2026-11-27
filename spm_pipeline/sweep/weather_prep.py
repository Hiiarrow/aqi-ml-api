"""
Prepare weather datasets (strictly <= time t, zero future leakage):
1. Open-Meteo ERA5 hourly weather (2017-2023 IST wall-clock) for all 6 city centroids -> raw_cpcb/weather_openmeteo_2017_2023.parquet
2. CPCB co-located hourly weather (AT, RH, WS, WD_sin, WD_cos, SR) QC'd for all 24 stations -> raw_cpcb/weather_cpcb_24stations_2017_2023.parquet
"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fetch"))

from export_opencity import DEFAULT_STATIONS

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
}

OM_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "surface_pressure", "shortwave_radiation", "precipitation", "boundary_layer_height",
]


def fetch_openmeteo_all(out_path):
    if os.path.exists(out_path):
        print(f"Open-Meteo cache already exists: {out_path}", flush=True)
        return pd.read_parquet(out_path)
    frames = []
    for city, (lat, lon) in CITY_COORDS.items():
        t0 = time.time()
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date=2017-01-01&end_date=2023-12-31"
            f"&hourly={','.join(OM_VARS)}&timezone=Asia%2FKolkata"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "python-spm-weather"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        h = d["hourly"]
        df = pd.DataFrame({"city": city, "hour": pd.to_datetime(h["time"])})
        for v in OM_VARS:
            df[v] = pd.to_numeric(h[v], errors="coerce").astype(np.float32)
        frames.append(df)
        print(f"  Fetched Open-Meteo {city}: {len(df):,} hours in {time.time()-t0:.1f}s", flush=True)
        time.sleep(0.5)
    all_df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(all_df):,} rows)", flush=True)
    return all_df


def build_cpcb_weather(out_path):
    if os.path.exists(out_path):
        print(f"CPCB weather cache already exists: {out_path}", flush=True)
        return pd.read_parquet(out_path)
    target_map = {full: (city, short) for city, m in DEFAULT_STATIONS.items() for full, short in m.items()}
    wcols = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "SR (W/mt2)"]
    files = sorted(glob.glob(os.path.join(ROOT, "raw_cpcb", "cpcb_6cities_*.parquet")))
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["City", "Station Name", "Timestamp"] + wcols)
        d = d[d["Station Name"].isin(target_map)].copy()
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(raw["Timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    raw["hour"] = ts.dt.floor("h")
    raw["station"] = raw["Station Name"].map(lambda s: target_map[s][1])
    raw["city"] = raw["Station Name"].map(lambda s: target_map[s][0])

    # Physical range QC on CPCB met sensors before hourly mean
    raw.loc[(raw["AT (°C)"] < -5) | (raw["AT (°C)"] > 55), "AT (°C)"] = np.nan
    raw.loc[(raw["RH (%)"] < 0) | (raw["RH (%)"] > 100), "RH (%)"] = np.nan
    raw.loc[(raw["WS (m/s)"] < 0) | (raw["WS (m/s)"] > 40), "WS (m/s)"] = np.nan
    raw.loc[(raw["WD (deg)"] < 0) | (raw["WD (deg)"] > 360), "WD (deg)"] = np.nan
    raw.loc[(raw["SR (W/mt2)"] < 0) | (raw["SR (W/mt2)"] > 1500), "SR (W/mt2)"] = np.nan

    # Vector circular average for wind direction via sin/cos
    rad = np.deg2rad(raw["WD (deg)"])
    raw["wd_sin"] = np.sin(rad)
    raw["wd_cos"] = np.cos(rad)

    hr = raw.groupby(["city", "station", "hour"], as_index=False).agg(
        at=("AT (°C)", "mean"),
        rh=("RH (%)", "mean"),
        ws=("WS (m/s)", "mean"),
        wd_sin=("wd_sin", "mean"),
        wd_cos=("wd_cos", "mean"),
        sr=("SR (W/mt2)", "mean"),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hr.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(hr):,} station-hours)", flush=True)
    return hr


if __name__ == "__main__":
    os.chdir(ROOT)
    fetch_openmeteo_all(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet"))
    build_cpcb_weather(os.path.join(ROOT, "raw_cpcb", "weather_cpcb_24stations_2017_2023.parquet"))

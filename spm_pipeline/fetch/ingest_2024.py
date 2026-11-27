"""
Ingest Vonter 2024 CPCB 15-minute Parquet + Open-Meteo 2024 ERA5 weather,
build 2017-2024 OpenCity matrices in data_2024/ for the 20 study stations (excl. Hyderabad),
and build frozen QC v3 caches in results_final/ without modifying earlier folders or data/.
"""
import json
import os
import pickle
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))

import qc  # noqa: E402
from download_cpcb import fetch_year_sparse  # noqa: E402
from weather_prep import CITY_COORDS, OM_VARS  # noqa: E402


STUDY_CITIES = {"Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"}


def ensure_2024_cpcb_parquet():
    dst = os.path.join(ROOT, "raw_cpcb", "cpcb_6cities_2024.parquet")
    if not os.path.exists(dst):
        fetch_year_sparse(2024, dst)
    else:
        print(f"2024 CPCB parquet already cached: {dst}", flush=True)


def ensure_openmeteo_2017_2024():
    out_p = os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2024.parquet")
    if os.path.exists(out_p):
        df = pd.read_parquet(out_p)
        print(f"Open-Meteo 2017-2024 already cached: {out_p} ({len(df):,} rows)", flush=True)
        return df

    base = pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet"))
    frames = [base]
    for city, (lat, lon) in CITY_COORDS.items():
        t0 = time.time()
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date=2024-01-01&end_date=2024-12-31"
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
        print(f"  Fetched Open-Meteo 2024 {city}: {len(df):,} hours, nans={int(df[OM_VARS].isna().sum().sum())} ({time.time()-t0:.1f}s)", flush=True)
        time.sleep(0.4)

    all_df = pd.concat(frames, ignore_index=True).sort_values(["city", "hour"]).reset_index(drop=True)
    exp_idx = pd.date_range("2017-01-01 00:00", "2024-12-31 23:00", freq="h")
    non_blh = [v for v in OM_VARS if v != "boundary_layer_height"]
    for city, g in all_df.groupby("city"):
        assert len(g) == len(exp_idx), f"{city}: {len(g)} != {len(exp_idx)}"
        assert g["hour"].is_monotonic_increasing and g["hour"].nunique() == len(exp_idx)
        assert int(g[non_blh].isna().sum().sum()) == 0, f"NaNs in core Open-Meteo vars for {city}"
        pre24 = g[g["hour"] < pd.Timestamp("2024-01-01 00:00")]
        assert int(pre24["boundary_layer_height"].isna().sum()) == 0, f"Pre-2024 BLH NaNs for {city}"
        blh_na_24 = int(g["boundary_layer_height"].isna().sum())
        if blh_na_24:
            print(f"  [note] {city}: boundary_layer_height has {blh_na_24} upstream Open-Meteo NaNs in 2024 (standardized->0.0 at extract_features)", flush=True)
    all_df.to_parquet(out_p, index=False)
    print(f"Saved Open-Meteo 2017-2024 -> {out_p} ({len(all_df):,} rows, 0 NaNs)", flush=True)
    return all_df


def ensure_data_2024_csvs():
    out_dir = os.path.join(ROOT, "data_2024")
    if os.path.isdir(out_dir):
        existing = qc.list_station_files(out_dir)
        if len(existing) == 20:
            print(f"data_2024/ already populated with 20 stations.", flush=True)
            return
    cmd = [
        sys.executable,
        os.path.join(ROOT, "fetch", "export_opencity.py"),
        "--raw-dir", os.path.join(ROOT, "raw_cpcb"),
        "--out-dir", out_dir,
        "--stations", os.path.join(ROOT, "fetch", "study20_stations.json"),
    ]
    subprocess.run(cmd, check=True)


def build_qc3_cache(data_dir, cache_path, report_path, label):
    if os.path.exists(cache_path) and os.path.exists(report_path):
        with open(cache_path, "rb") as f:
            clean = pickle.load(f)
        print(f"{label} cache already exists: {cache_path}", flush=True)
        return clean

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    clean, rows = {}, []
    for city, st, p in qc.list_station_files(data_dir):
        if city not in STUDY_CITIES:
            continue
        st_obj, rep = qc.qc_station_v3(p)
        rows.append(dict(city=city, station=st, **rep))
        if st_obj is not None:
            clean.setdefault(city, {})[st] = st_obj
    pd.DataFrame(rows).to_csv(report_path, index=False)
    assert len(clean) == 5 and sum(len(v) for v in clean.values()) == 20, f"{label}: expected 5 cities / 20 stations"
    with open(cache_path, "wb") as f:
        pickle.dump(clean, f)
    print(f"Saved {label} QC v3 cache -> {cache_path}", flush=True)
    return clean


def main():
    os.chdir(ROOT)
    ensure_2024_cpcb_parquet()
    ensure_openmeteo_2017_2024()
    ensure_data_2024_csvs()

    out_dir = os.path.join(ROOT, "results_final")
    os.makedirs(out_dir, exist_ok=True)
    c23 = build_qc3_cache(
        os.path.join(ROOT, "data"),
        os.path.join(out_dir, "clean_20_2017_2023_v3.pkl"),
        os.path.join(out_dir, "qc3_2023_station_report.csv"),
        "2017-2023 (data/)",
    )
    c24 = build_qc3_cache(
        os.path.join(ROOT, "data_2024"),
        os.path.join(out_dir, "clean_20_2017_2024_v3.pkl"),
        os.path.join(out_dir, "qc3_2024_station_report.csv"),
        "2017-2024 (data_2024/)",
    )

    print("\n=== 2024 POST-QC v3 AVAILABILITY PER STATION ===")
    c2024_cut = np.datetime64("2024-01-01T00", "h")
    for city in sorted(c24):
        for st in sorted(c24[city]):
            obj24 = c24[city][st]
            m24 = (obj24["times"] >= c2024_cut) & np.isfinite(obj24["value"])
            obj23 = c23[city][st]
            pre_m = obj23["times"] < np.datetime64("2023-12-31T00", "h")
            sub23 = obj23["value"][pre_m]
            # Compare interior 2017-2023 values away from Dec 31 boundary
            idx24_pre = obj24["times"] < np.datetime64("2023-12-31T00", "h")
            sub24 = obj24["value"][idx24_pre]
            # align start timestamps since 2017-2024 might have identical range through 2023-12-30
            t23 = pd.Series(sub23, index=obj23["times"][pre_m])
            t24 = pd.Series(sub24, index=obj24["times"][idx24_pre]).reindex(t23.index)
            diff = np.nanmax(np.abs(t23.values - t24.values)) if np.isfinite(t23.values).any() else 0.0
            assert diff < 1e-6, f"2017-2023 mismatch for {city}/{st}: max diff={diff}"
            valid_tot = int(np.isfinite(obj24["value"]).sum())
            print(f"  {city:10s} | {st:18s} | total_valid_17_24={valid_tot:6d} | valid_2024={int(m24.sum()):5d}")


if __name__ == "__main__":
    main()

"""Fetch and cache Open-Meteo ERA5 hourly weather (2022-2024) for all candidate external cities
and report each city's surface_pressure range vs the 5-city training range.

Writes strictly into results_ext_cities/. Does not modify any frozen files.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sweep"))

from weather_prep import OM_VARS  # noqa: E402

EXT_DIR = os.path.join(HERE, "results_ext_cities")
CACHE_DIR = os.path.join(EXT_DIR, "cache")

# State hints to verify Indian geocoding results cleanly
EXPECTED_ADMIN1 = {
    "Prayagraj": "Uttar Pradesh",
    "Chandigarh": "Chandigarh",
    "Muzaffarpur": "Bihar",
    "Gurugram": "Haryana",
}


def geocode_indian_city(city, expected_state):
    query = city
    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode({"name": query, "count": 10, "language": "en", "format": "json"})
    )
    req = urllib.request.Request(url, headers={"User-Agent": "python-spm-ext-weather"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    results = [x for x in data.get("results", []) if x.get("country_code") == "IN"]
    assert results, f"No Indian geocoding result for {city}"
    exact = [
        x for x in results
        if str(x.get("admin1", "")).lower() == str(expected_state).lower()
    ]
    best = exact[0] if exact else results[0]
    return float(best["latitude"]), float(best["longitude"]), float(best.get("elevation", np.nan)), best.get("admin1", "")


def main():
    ca = pd.read_csv(os.path.join(EXT_DIR, "step1_city_audit_2022_2024.csv"))
    cands = ca[ca["eligible_ge3_stations"]].copy()

    coords_path = os.path.join(CACHE_DIR, "ext_city_coords.csv")
    wx_path = os.path.join(CACHE_DIR, "weather_openmeteo_ext_cities_2022_2024.parquet")

    if os.path.exists(coords_path) and os.path.exists(wx_path):
        coords_df = pd.read_csv(coords_path)
        wx_df = pd.read_parquet(wx_path)
        print(f"Loaded cached ext weather: {wx_path} ({len(wx_df):,} rows)")
    else:
        coord_rows, frames = [], []
        for _, r in cands.iterrows():
            city, state = r["city"], r["state"]
            lat, lon, elev, adm1 = geocode_indian_city(city, state)
            coord_rows.append(dict(
                city=city, state=state, eligible_all_stations=bool(r["eligible_all_stations"]),
                lat=lat, lon=lon, elev_m=elev, geocoded_admin1=adm1,
            ))
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
                f"&start_date=2022-01-01&end_date=2024-12-31"
                f"&hourly={','.join(OM_VARS)}&timezone=Asia%2FKolkata"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "python-spm-ext-weather"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.loads(resp.read())
            h = d["hourly"]
            df = pd.DataFrame({"city": city, "hour": pd.to_datetime(h["time"])})
            for v in OM_VARS:
                df[v] = pd.to_numeric(h[v], errors="coerce").astype(np.float32)
            frames.append(df)
            print(f"  Fetched {city:14s} ({lat:.4f}, {lon:.4f}, elev={elev:.0f}m, {adm1}): {len(df):,}h", flush=True)
            time.sleep(0.35)

        coords_df = pd.DataFrame(coord_rows)
        coords_df.to_csv(coords_path, index=False)
        wx_df = pd.concat(frames, ignore_index=True).sort_values(["city", "hour"]).reset_index(drop=True)
        wx_df.to_parquet(wx_path, index=False)
        print(f"Saved {wx_path} ({len(wx_df):,} rows)")

    # Check surface_pressure range against the 5-city training ranges (2017-2022 and 2017-2023)
    tr_wx = pd.read_parquet(os.path.join(HERE, "raw_cpcb", "weather_openmeteo_2017_2024.parquet"))
    tr5 = tr_wx[tr_wx["city"].isin({"Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"})].copy()
    tr22 = tr5[tr5["hour"] < pd.Timestamp("2023-01-01")]["surface_pressure"]
    tr23 = tr5[tr5["hour"] < pd.Timestamp("2024-01-01")]["surface_pressure"]
    p22_lo, p22_hi = float(tr22.min()), float(tr22.max())
    p23_lo, p23_hi = float(tr23.min()), float(tr23.max())
    print(f"\n5-city training surface_pressure ranges:")
    print(f"  2017-2022 (P2_2023 train): [{p22_lo:.2f}, {p22_hi:.2f}] hPa")
    print(f"  2017-2023 (P2_2024 train): [{p23_lo:.2f}, {p23_hi:.2f}] hPa")

    prows = []
    for _, cr in coords_df.iterrows():
        c = cr["city"]
        g = wx_df[wx_df["city"] == c]
        g23 = g[g["hour"].dt.year == 2023]["surface_pressure"]
        g24 = g[g["hour"].dt.year == 2024]["surface_pressure"]
        out23 = (float(g23.min()) < p22_lo) or (float(g23.max()) > p22_hi)
        out24 = (float(g24.min()) < p23_lo) or (float(g24.max()) > p23_hi)
        prows.append(dict(
            city=c, state=cr["state"], eligible_all_stations=cr["eligible_all_stations"],
            elev_m=cr["elev_m"],
            sp_2023_min=round(float(g23.min()), 2), sp_2023_max=round(float(g23.max()), 2), outside_2017_2022=out23,
            sp_2024_min=round(float(g24.min()), 2), sp_2024_max=round(float(g24.max()), 2), outside_2017_2023=out24,
        ))
    pdf = pd.DataFrame(prows)
    pdf.to_csv(os.path.join(EXT_DIR, "step1_surface_pressure_check.csv"), index=False)
    print("\n=== CANDIDATE CITIES SURFACE_PRESSURE RANGE VS 5-CITY TRAINING ===")
    print(pdf.to_string(index=False))


if __name__ == "__main__":
    main()

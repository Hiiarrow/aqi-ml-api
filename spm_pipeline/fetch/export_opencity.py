"""
Convert cached CPCB 6-city Parquets (raw_cpcb/cpcb_6cities_*.parquet) into
verified OpenCity matrix CSVs under data/<City>/<Station>.csv.

Usage:
    python fetch/export_opencity.py --stations fetch/selected_stations.json --out data
"""
import argparse
import calendar
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import qc  # noqa: E402


# Default 24 canonical stations (4 per city across 6 cities; Chembur excluded)
DEFAULT_STATIONS = {
    "Bengaluru": {
        "BTM Layout, Bengaluru - CPCB": "BTM Layout",
        "Hebbal, Bengaluru - KSPCB": "Hebbal",
        "Hombegowda Nagar, Bengaluru - KSPCB": "Hombegowda Nagar",
        "Peenya, Bengaluru - CPCB": "Peenya",
    },
    "Chennai": {
        "Kodungaiyur, Chennai - TNPCB": "Kodungaiyur",
        "Manali, Chennai - CPCB": "Manali CPCB",
        "Perungudi, Chennai - TNPCB": "Perungudi",
        "Royapuram, Chennai - TNPCB": "Royapuram",
    },
    "Delhi": {
        "Anand Vihar, Delhi - DPCC": "Anand Vihar",
        "Mandir Marg, Delhi - DPCC": "Mandir Marg",
        "Punjabi Bagh, Delhi - DPCC": "Punjabi Bagh",
        "R K Puram, Delhi - DPCC": "R.K. Puram",
    },
    "Hyderabad": {
        "Bollaram Industrial Area, Hyderabad - TSPCB": "Bollaram",
        "ICRISAT Patancheru, Hyderabad - TSPCB": "ICRISAT",
        "Sanathnagar, Hyderabad - TSPCB": "Sanathnagar",
        "Zoo Park, Hyderabad - TSPCB": "Zoo Park",
    },
    "Kolkata": {
        "Ballygunge, Kolkata - WBPCB": "Ballygunge",
        "Jadavpur, Kolkata - WBPCB": "Jadavpur",
        "Rabindra Bharati University, Kolkata - WBPCB": "Rabindra Bharati",
        "Victoria, Kolkata - WBPCB": "Victoria",
    },
    "Mumbai": {
        "Bandra Kurla Complex, Mumbai - IITM": "BKC",
        "Borivali East, Mumbai - MPCB": "Borivali East",
        "Powai, Mumbai - MPCB": "Powai",
        "Sion, Mumbai - MPCB": "Sion MPCB",
    },
}


def write_opencity_matrix(hourly_series: pd.Series, out_path: str):
    """Fast vectorized writer matching qc.parse_opencity exact format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if hourly_series.empty:
        raise ValueError(f"Empty series for {out_path}")

    t0 = hourly_series.index.min().replace(day=1, hour=0)
    t1_ts = hourly_series.index.max()
    last_day = calendar.monthrange(t1_ts.year, t1_ts.month)[1]
    t1 = t1_ts.replace(day=last_day, hour=23)

    full_idx = pd.date_range(t0, t1, freq="h")
    s = hourly_series.reindex(full_idx)

    # Reshape into table indexed by (year, month, day) with columns 0..23
    df = pd.DataFrame({"v": s.values}, index=full_idx)
    df["y"] = full_idx.year
    df["m"] = full_idx.month
    df["d"] = full_idx.day
    df["h"] = full_idx.hour
    piv = df.pivot(index=["y", "m", "d"], columns="h", values="v")

    lines = []
    cur_y = None
    first_hdr = True
    for (y, m), grp in piv.groupby(level=[0, 1], sort=True):
        if y != cur_y:
            cur_y = y
            lines.append(f"Year,{y}")
        lines.append(f"{calendar.month_name[m]}-{y}," + ",".join(f"{h:02d}:00:00" for h in range(24)))
        if first_hdr:
            lines.append('"\n"')
            first_hdr = False
        vals = grp.to_numpy()
        days = grp.index.get_level_values(2)
        for day, row in zip(days, vals):
            toks = ["" if np.isnan(x) else f"{x:.2f}" for x in row]
            lines.append(f"{day}," + ",".join(toks))

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw_cpcb")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--stations", default=None, help="Optional JSON mapping {City: {CPCB Name: short_name}}")
    ap.add_argument("--col", default="PM2.5 (µg/m³)")
    args = ap.parse_args()

    station_map = DEFAULT_STATIONS
    if args.stations:
        with open(args.stations, "r", encoding="utf-8") as f:
            station_map = json.load(f)

    target_full_names = {full for cmap in station_map.values() for full in cmap}
    files = sorted(glob.glob(os.path.join(args.raw_dir, "cpcb_6cities_*.parquet")))
    assert files, f"No cpcb_6cities_*.parquet files found in {args.raw_dir}"

    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["City", "Station Name", "Timestamp", args.col])
        d = d[d["Station Name"].isin(target_full_names)]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    # Strip fake UTC label (verified IST wall-clock)
    ts = pd.to_datetime(df["Timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    df["hour"] = ts.dt.floor("h")

    # Drop null & physically impossible negative readings before averaging
    n_neg = int((df[args.col] < 0).sum())
    df = df[df[args.col].notna() & (df[args.col] >= 0)].copy()
    print(f"Loaded {len(df):,} valid non-negative 15-min rows (dropped {n_neg} <0 rows) for {len(target_full_names)} target stations.")

    hr = df.groupby(["City", "Station Name", "hour"])[args.col].mean()

    for city, cmap in station_map.items():
        for full_name, short_name in cmap.items():
            sub = hr.loc[(city, full_name)] if (city, full_name) in hr.index else pd.Series(dtype=float)
            if sub.empty:
                raise RuntimeError(f"MISSING DATA for {city} / {full_name}")
            out_path = os.path.join(args.out_dir, city, f"{short_name}.csv")
            write_opencity_matrix(sub, out_path)
            # Round-trip verification against qc.parse_opencity
            idx, vals, slots, dups = qc.parse_opencity(out_path)
            assert dups == 0, f"Duplicate timestamps in {out_path}"
            n_fin = int(np.isfinite(vals).sum())
            print(f"  {city:10s} | {short_name:18s} | {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d} | "
                  f"valid={n_fin:,}/{slots:,}h | mean={np.nanmean(vals):.2f} max={np.nanmax(vals):.2f}")


if __name__ == "__main__":
    main()

"""Step 1 external validation audit: sparse-download Vonter 2022-2024 PM2.5 across all cities,
run frozen QC v3 per station over 2022-2024, report all non-study/non-Hyderabad cities,
and determine the eligible city list (>=3 stations, >=85% valid hours, no ceiling clipping).

Writes exclusively to results_ext_cities/. Does not modify any frozen directories.
"""
import io
import json
import os
import re
import struct
import sys
import time
from bisect import bisect_right

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fetch"))

import qc  # noqa: E402
from export_opencity import write_opencity_matrix  # noqa: E402

EXT_DIR = os.path.join(HERE, "results_ext_cities")
CACHE_DIR = os.path.join(EXT_DIR, "cache")
OC_DIR = os.path.join(EXT_DIR, "opencity_2022_2024")

EXCLUDED_CITIES = {"Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai", "Hyderabad"}
KEEP_COLS = ["State", "City", "Station Name", "Timestamp", "PM2.5 (µg/m³)"]
FULL_22_24 = pd.date_range("2022-01-01 00:00", "2024-12-31 23:00", freq="h")
TOTAL_HOURS_22_24 = len(FULL_22_24)  # 26,304 hours


def _safe_name(s):
    return re.sub(r'[<>:"/\\|?*]+', "_", str(s)).strip()


def _http_get_range(sess, url, start, end, retries=5):
    for attempt in range(retries):
        try:
            r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} range {start}-{end} after {wait}s: {e}", flush=True)
            time.sleep(wait)


def fetch_all_cities_pm25_year(yr, out_path):
    """Sparse HTTP Range fetch of ['State', 'City', 'Station Name', 'Timestamp', 'PM2.5 (µg/m³)'] for all cities."""
    if os.path.exists(out_path):
        print(f"[{yr}] SKIP download (cached at {out_path})", flush=True)
        return out_path

    t0 = time.time()
    url = f"https://github.com/Vonter/india-cpcb-aqi/releases/download/{yr}/cpcb-air-quality-{yr}.parquet"
    sess = requests.Session()
    sess.headers["User-Agent"] = "python-cpcb-ext-audit"

    r0 = sess.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=30)
    r0.raise_for_status()
    final_url = r0.url
    total_len = int(r0.headers["Content-Range"].split("/")[-1])

    tail8 = _http_get_range(sess, final_url, total_len - 8, total_len - 1)
    footer_len, magic = struct.unpack("<I4s", tail8)
    assert magic == b"PAR1", f"Invalid magic {magic} for {yr}"
    meta_start = total_len - 8 - footer_len
    meta_bytes = _http_get_range(sess, final_url, meta_start, total_len - 1)

    class MetaOnlyFile(io.RawIOBase):
        def __init__(self):
            self.pos = 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.pos
        def seek(self, off, w=io.SEEK_SET):
            if w == io.SEEK_SET: self.pos = off
            elif w == io.SEEK_CUR: self.pos += off
            elif w == io.SEEK_END: self.pos = total_len + off
            return self.pos
        def read(self, n=-1):
            if n == 0 or self.pos >= total_len: return b""
            o = self.pos - meta_start
            d = meta_bytes[o:] if n < 0 else meta_bytes[o:o + n]
            self.pos += len(d)
            return d
        def readinto(self, b):
            d = self.read(len(b))
            n = len(d)
            b[:n] = d
            return n

    pf = pq.ParquetFile(MetaOnlyFile())
    names = pf.schema.names
    col_idxs = [names.index(c) for c in KEEP_COLS]

    spans = []
    num_rgs = pf.metadata.num_row_groups
    for i in range(num_rgs):
        rg = pf.metadata.row_group(i)
        for ci in col_idxs:
            cm = rg.column(ci)
            offs = [x for x in (cm.dictionary_page_offset, cm.data_page_offset) if x is not None and x >= 0]
            s = min(offs)
            e = min(total_len - 1, s + cm.total_compressed_size + 128)
            spans.append((s, e))

    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 524288:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    dl_mb = (sum(e - s + 1 for s, e in merged) + len(meta_bytes)) / 1e6
    print(f"[{yr}] {pf.metadata.num_rows:,} rows | fetching {dl_mb:.1f}MB / {total_len/1e6:.1f}MB in {len(merged)} chunks", flush=True)

    chunks = []
    for k, (s, e) in enumerate(merged, 1):
        buf = _http_get_range(sess, final_url, s, e)
        chunks.append((s, buf))
        if k % 15 == 0 or k == len(merged):
            print(f"[{yr}]   chunk {k}/{len(merged)} ({time.time()-t0:.1f}s)", flush=True)
    chunks.append((meta_start, meta_bytes))
    chunks.sort(key=lambda x: x[0])
    starts = [c[0] for c in chunks]

    class SparseMemFile(io.RawIOBase):
        def __init__(self):
            self.pos = 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.pos
        def seek(self, off, w=io.SEEK_SET):
            if w == io.SEEK_SET: self.pos = off
            elif w == io.SEEK_CUR: self.pos += off
            elif w == io.SEEK_END: self.pos = total_len + off
            return self.pos
        def read(self, n=-1):
            if n == 0 or self.pos >= total_len: return b""
            j = bisect_right(starts, self.pos) - 1
            if j >= 0:
                cs, buf = chunks[j]
                if cs <= self.pos < cs + len(buf):
                    o = self.pos - cs
                    d = buf[o:] if n < 0 else buf[o:o + n]
                    self.pos += len(d)
                    return d
            end = min(total_len - 1, self.pos + (1048576 if n < 0 else n) - 1)
            d = _http_get_range(sess, final_url, self.pos, end)
            self.pos += len(d)
            return d
        def readinto(self, b):
            d = self.read(len(b))
            n = len(d)
            b[:n] = d
            return n

    pf2 = pq.ParquetFile(SparseMemFile())
    df = pf2.read_row_groups(list(range(num_rgs)), columns=KEEP_COLS).to_pandas()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{yr}] Saved {len(df):,} rows -> {out_path} ({time.time()-t0:.1f}s)", flush=True)
    return out_path


def _ceiling_check(times, vals):
    """Inspect max value, exact-at-max count/share, within-1 share, and per-year ceiling flag."""
    fin = np.isfinite(vals)
    n_fin = int(fin.sum())
    if n_fin == 0:
        return dict(max_val=np.nan, p99=np.nan, max_over_p99=np.nan,
                    n_at_max=0, share_at_max_pct=np.nan, share_within_1_pct=np.nan,
                    ceiling_flag=False, ceiling_years="")
    vf = vals[fin]
    tf = pd.DatetimeIndex(times[fin])
    mx = float(np.max(vf))
    p99 = float(np.percentile(vf, 99))
    ratio = mx / p99 if p99 > 0 else np.nan
    n_at_max = int(np.sum(np.isclose(vf, mx, atol=1e-6)))
    n_w1 = int(np.sum(vf >= mx - 1.0))
    sh_max = 100.0 * n_at_max / n_fin
    sh_w1 = 100.0 * n_w1 / n_fin

    # Same annual ceiling rule established earlier (share_within_1 > 0.3% and max/p99 < 1.1)
    # plus direct overall spike pileup at max (share_at_max > 0.1% with n_at_max >= 5)
    c_years = []
    for yr in (2022, 2023, 2024):
        vy = vf[tf.year == yr]
        if len(vy) < 100:
            continue
        ymx = float(np.max(vy))
        yp99 = float(np.percentile(vy, 99))
        yw1 = 100.0 * float(np.sum(vy >= ymx - 1.0)) / len(vy)
        yshmax = 100.0 * float(np.sum(np.isclose(vy, ymx, atol=1e-6))) / len(vy)
        yrat = ymx / yp99 if yp99 > 0 else np.nan
        if (yw1 > 0.3 and np.isfinite(yrat) and yrat < 1.1) or (yshmax > 0.1 and np.sum(np.isclose(vy, ymx, atol=1e-6)) >= 5):
            c_years.append(f"{yr}(max={ymx:.2f},n={int(np.sum(np.isclose(vy, ymx, atol=1e-6)))},w1={yw1:.2f}%,rat={yrat:.3f})")

    overall_flag = bool(c_years) or (sh_w1 > 0.3 and np.isfinite(ratio) and ratio < 1.1) or (sh_max > 0.1 and n_at_max >= 5)
    return dict(
        max_val=round(mx, 2),
        p99=round(p99, 2),
        max_over_p99=round(ratio, 3) if np.isfinite(ratio) else np.nan,
        n_at_max=n_at_max,
        share_at_max_pct=round(sh_max, 4),
        share_within_1_pct=round(sh_w1, 4),
        ceiling_flag=overall_flag,
        ceiling_years=";".join(c_years),
    )


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OC_DIR, exist_ok=True)

    parts = []
    for yr in (2022, 2023, 2024):
        p = fetch_all_cities_pm25_year(yr, os.path.join(CACHE_DIR, f"vonter_pm25_allcities_{yr}.parquet"))
        d = pd.read_parquet(p, columns=["State", "City", "Station Name", "Timestamp", "PM2.5 (µg/m³)"])
        d = d[~d["City"].isin(EXCLUDED_CITIES)].copy()
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    print(f"\nLoaded non-excluded 15-min rows (2022-2024): {len(df):,}, cities={df['City'].nunique()}, stations={df.groupby(['City', 'Station Name']).ngroups}", flush=True)

    all_meta = (
        df.groupby(["City", "Station Name"], as_index=False)["State"]
        .first()
    )
    st_states = {(r["City"], r["Station Name"]): r["State"] for _, r in all_meta.iterrows()}

    ts = pd.to_datetime(df["Timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    df["hour"] = ts.dt.floor("h")
    df_pos = df[df["PM2.5 (µg/m³)"].notna() & (df["PM2.5 (µg/m³)"] >= 0)].copy()

    hr = df_pos.groupby(["City", "Station Name", "hour"], as_index=False)["PM2.5 (µg/m³)"].mean()
    grp_map = {(c, s): g for (c, s), g in hr.groupby(["City", "Station Name"], sort=True)}

    station_rows = []
    manifest = {}  # City -> list of dicts
    for city, st_name in sorted(st_states):
        if (city, st_name) not in grp_map:
            row = dict(
                state=st_states[(city, st_name)], city=city, station=st_name,
                csv_rel="", valid_hours_qc3=0, total_hours=TOTAL_HOURS_22_24,
                valid_frac=0.0, valid_pct=0.0, qc_status="empty_2022_2024",
                max_val=np.nan, p99=np.nan, max_over_p99=np.nan, n_at_max=0,
                share_at_max_pct=np.nan, share_within_1_pct=np.nan,
                ceiling_flag=False, ceiling_years="",
            )
            station_rows.append(row)
            manifest.setdefault(city, []).append(row)
            continue
        g = grp_map[(city, st_name)]
        s = pd.Series(g["PM2.5 (µg/m³)"].values, index=g["hour"].values).sort_index()
        s = s[(s.index >= FULL_22_24[0]) & (s.index <= FULL_22_24[-1])]
        if s.empty:
            row = dict(
                state=st_states.get((city, st_name), ""), city=city, station=st_name,
                csv_rel="", valid_hours_qc3=0, total_hours=TOTAL_HOURS_22_24,
                valid_frac=0.0, valid_pct=0.0, qc_status="empty_2022_2024",
                max_val=np.nan, p99=np.nan, max_over_p99=np.nan, n_at_max=0,
                share_at_max_pct=np.nan, share_within_1_pct=np.nan,
                ceiling_flag=False, ceiling_years="",
            )
            station_rows.append(row)
            manifest.setdefault(city, []).append(row)
            continue

        rel = os.path.join(_safe_name(city), f"{_safe_name(st_name)}.csv")
        out_csv = os.path.join(OC_DIR, rel)
        write_opencity_matrix(s, out_csv)

        st_obj, rep = qc.qc_station_v3(out_csv)
        if st_obj is not None:
            t_arr = st_obj["times"]
            v_arr = st_obj["value"]
            m22 = (t_arr >= np.datetime64("2022-01-01T00", "h")) & (t_arr <= np.datetime64("2024-12-31T23", "h"))
            t_sub = t_arr[m22]
            v_sub = v_arr[m22]
            n_val = int(np.isfinite(v_sub).sum())
            cc = _ceiling_check(t_sub, v_sub)
        else:
            # QC v3 returned None due to span < 2.0y; still report valid hours & raw post-QC count
            n_val = int(rep["valid_hours_out"])
            cc = _ceiling_check(s.index.values, s.values.astype(float))

        vfrac = n_val / float(TOTAL_HOURS_22_24)
        row = dict(
            state=st_states.get((city, st_name), ""),
            city=city,
            station=st_name,
            csv_rel=rel,
            valid_hours_qc3=n_val,
            total_hours=TOTAL_HOURS_22_24,
            valid_frac=round(vfrac, 4),
            valid_pct=round(100.0 * vfrac, 2),
            qc_status=rep["status"],
            **cc,
        )
        station_rows.append(row)
        manifest.setdefault(city, []).append(row)

    df_st = pd.DataFrame(station_rows).sort_values(["city", "valid_frac"], ascending=[True, False])
    df_st.to_csv(os.path.join(EXT_DIR, "step1_station_audit_2022_2024.csv"), index=False)

    city_rows = []
    for city, srows in sorted(manifest.items()):
        srows_sorted = sorted(srows, key=lambda r: (-r["valid_frac"], r["station"]))
        n_tot = len(srows_sorted)
        good = [r for r in srows_sorted if (r["valid_frac"] >= 0.85 and not r["ceiling_flag"] and r["qc_status"] == "retained")]
        bad_cov = [r for r in srows_sorted if r["valid_frac"] < 0.85]
        bad_ceil = [r for r in srows_sorted if r["ceiling_flag"]]

        all_pass = (n_tot >= 3 and len(good) == n_tot)
        ge3_pass = (len(good) >= 3)

        reasons = []
        if n_tot < 3:
            reasons.append(f"<3 stations ({n_tot})")
        if bad_cov:
            reasons.append(f"{len(bad_cov)}/{n_tot} station(s) <85% valid")
        if bad_ceil:
            reasons.append(f"{len(bad_ceil)}/{n_tot} station(s) ceiling-flagged")
        reason_str = "eligible (all stations pass)" if all_pass else "; ".join(reasons)

        st_detail = " | ".join(
            f"{r['station']}={r['valid_frac']:.3f}(max={r['max_val']},@max={r['share_at_max_pct']:.3f}%{' CEIL' if r['ceiling_flag'] else ''})"
            for r in srows_sorted
        )
        city_rows.append(dict(
            state=srows_sorted[0]["state"],
            city=city,
            n_stations=n_tot,
            n_clean_ge85=len(good),
            min_valid_frac=min(r["valid_frac"] for r in srows_sorted),
            max_valid_frac=max(r["valid_frac"] for r in srows_sorted),
            any_ceiling_flag=bool(bad_ceil),
            eligible_all_stations=all_pass,
            eligible_ge3_stations=ge3_pass,
            exclusion_reason=reason_str,
            stations_detail=st_detail,
        ))

    df_city = pd.DataFrame(city_rows).sort_values(["eligible_all_stations", "eligible_ge3_stations", "n_stations", "city"], ascending=[False, False, False, True])
    df_city.to_csv(os.path.join(EXT_DIR, "step1_city_audit_2022_2024.csv"), index=False)

    df_elig = df_city[df_city["eligible_all_stations"]].copy()
    df_elig.to_csv(os.path.join(EXT_DIR, "step1_eligible_cities.csv"), index=False)

    print(f"\nTotal non-excluded cities: {len(df_city)}")
    print(f"Eligible cities (all stations pass >=85% & no ceiling): {len(df_elig)}")
    print(f"Cities with >=3 clean >=85% stations (subset): {int(df_city['eligible_ge3_stations'].sum())}")
    print("\n=== CANDIDATE CITIES (>=3 STATIONS) SUMMARY ===")
    sub_c = df_city[df_city["n_stations"] >= 3]
    print(sub_c[["state", "city", "n_stations", "n_clean_ge85", "min_valid_frac", "max_valid_frac", "any_ceiling_flag", "eligible_all_stations", "eligible_ge3_stations", "exclusion_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()

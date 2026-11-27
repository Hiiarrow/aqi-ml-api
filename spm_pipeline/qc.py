"""Parse OpenCity matrix CSVs and apply the single frozen QC spec."""
import datetime as dt
import hashlib
import os
import re

import numpy as np
import pandas as pd

import config

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_opencity(path):
    """Return (times[np.datetime64[h]], values[float], n_slots, n_duplicates)."""
    month = year = None
    seen = {}
    dups = 0
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip().strip('"').strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            m = re.match(r"^([A-Za-z]+)-(\d{4})$", parts[0])
            if m and m.group(1).lower() in MONTHS:
                month, year = MONTHS[m.group(1).lower()], int(m.group(2))
                continue
            if parts[0].isdigit() and month is not None:
                day = int(parts[0])
                for hr, tok in enumerate(parts[1:25]):
                    try:
                        ts = dt.datetime(year, month, day, hr)
                    except ValueError:
                        break
                    try:
                        val = float(tok)
                    except ValueError:
                        val = np.nan
                    if ts in seen:
                        dups += 1
                        continue
                    seen[ts] = val
    if not seen:
        raise ValueError(f"No data parsed from {path}")
    idx = pd.DatetimeIndex(sorted(seen))
    vals = np.array([seen[t] for t in idx.to_pydatetime()], dtype=float)
    return idx, vals, len(seen), dups


def _stuck_flags(v, eligible, max_run):
    n = len(v)
    flags = np.zeros(n, bool)
    i = 0
    while i < n:
        if not eligible[i]:
            i += 1
            continue
        j = i + 1
        while j < n and eligible[j] and v[j] == v[i]:
            j += 1
        if j - i > max_run:
            flags[i:j] = True
        i = j
    return flags


def _fill_short_gaps(v, max_gap):
    v = v.copy()
    interp = np.zeros(len(v), bool)
    isn = np.isnan(v)
    n = len(v)
    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < n:
            v[i:j] = np.linspace(v[i - 1], v[j], j - i + 2)[1:-1]
            interp[i:j] = True
        i = j
    return v, interp


def qc_station(path):
    """Apply the QC spec. Returns (station dict or None, report dict, over_cap list)."""
    idx, raw, n_slots, dups = parse_opencity(path)
    full = pd.date_range(idx[0], idx[-1], freq="h")
    s = pd.Series(raw, index=idx).reindex(full)
    v = s.values.astype(float)
    rep = dict(raw_slots=n_slots, nonnull_raw=int(np.isfinite(raw).sum()),
               duplicate_timestamps=dups)

    over = np.where(v > config.CAP)[0]
    over_list = [(str(full[i]), float(v[i])) for i in over]
    v[over] = np.nan
    rep["above_cap_removed"] = len(over)

    at_cap = (v == config.CAP)
    rep["hours_at_cap"] = int(at_cap.sum())

    eligible = np.isfinite(v)
    stuck = _stuck_flags(v, eligible, config.MAX_STUCK)
    v[stuck] = np.nan
    rep["stuck_hours_removed"] = int(stuck.sum())

    v, interp = _fill_short_gaps(v, config.MAX_GAP)
    rep["interpolated_hours"] = int(interp.sum())
    rep["valid_hours_out"] = int(np.isfinite(v).sum())

    valid_idx = np.where(np.isfinite(v))[0]
    span_years = (full[valid_idx[-1]] - full[valid_idx[0]]).days / 365.25 if len(valid_idx) else 0.0
    rep["span_years"] = round(span_years, 2)
    if span_years < config.MIN_SPAN_YEARS:
        rep["status"] = "dropped (span)"
        return None, rep, over_list
    rep["status"] = "retained"
    st = dict(times=full.values.astype("datetime64[h]"), value=v, interp=interp)
    return st, rep, over_list


def _one_hour_spike_flags(v, min_val=200.0, both_ratio=3.0, isol_gt=500.0, one_nan_ratio=5.0):
    """Flag single-hour sensor spikes using immediate neighbour ratios (evaluated read-only on v).
    - Both neighbours finite: x >= min_val and min(x/max(prev,1), x/max(next,1)) >= both_ratio
    - One neighbour NaN:       x > isol_gt and x/max(finite_nbr, 1) >= one_nan_ratio
    - Both neighbours NaN:     x > isol_gt
    """
    n = len(v)
    flags = np.zeros(n, bool)
    for i in range(n):
        x = v[i]
        if not np.isfinite(x):
            continue
        pl = v[i - 1] if i > 0 else np.nan
        pr = v[i + 1] if i + 1 < n else np.nan
        if np.isfinite(pl) and np.isfinite(pr):
            if x >= min_val and min(x / max(pl, 1.0), x / max(pr, 1.0)) >= both_ratio:
                flags[i] = True
        elif (not np.isfinite(pl)) and (not np.isfinite(pr)):
            if x > isol_gt:
                flags[i] = True
        else:
            nb = pl if np.isfinite(pl) else pr
            if x > isol_gt and (x / max(nb, 1.0)) >= one_nan_ratio:
                flags[i] = True
    return flags


def qc_station_v2(path, upper=980.0, fault_vals=(985.0, 999.99), fault_tol=0.01):
    """QC v2: exact/near faults {985.0, 999.99}, bounds [0, 980], stuck >6h, neighbour-ratio 1h spikes, short-gap fill."""
    idx, raw, n_slots, dups = parse_opencity(path)
    full = pd.date_range(idx[0], idx[-1], freq="h")
    s = pd.Series(raw, index=idx).reindex(full)
    v = s.values.astype(float)
    rep = dict(raw_slots=n_slots, nonnull_raw=int(np.isfinite(raw).sum()), duplicate_timestamps=dups)

    fault_985 = np.isfinite(v) & np.isclose(v, 985.0, atol=fault_tol)
    fault_999 = np.isfinite(v) & np.isclose(v, 999.99, atol=fault_tol)
    v[fault_985 | fault_999] = np.nan
    rep["fault_985_removed"] = int(fault_985.sum())
    rep["fault_999_removed"] = int(fault_999.sum())
    rep["fault_985_999_removed"] = int((fault_985 | fault_999).sum())

    neg = np.isfinite(v) & (v < 0)
    over = np.isfinite(v) & (v > upper)
    v[neg | over] = np.nan
    rep["out_of_range_removed"] = int((neg | over).sum())

    stuck = _stuck_flags(v, np.isfinite(v), config.MAX_STUCK)
    v[stuck] = np.nan
    rep["stuck_hours_removed"] = int(stuck.sum())

    spikes = _one_hour_spike_flags(v)
    v[spikes] = np.nan
    rep["one_hour_spike_removed"] = int(spikes.sum())

    rep["total_removed"] = int(rep["out_of_range_removed"] + rep["fault_985_999_removed"]
                               + rep["stuck_hours_removed"] + rep["one_hour_spike_removed"])

    v, interp = _fill_short_gaps(v, config.MAX_GAP)
    rep["interpolated_hours"] = int(interp.sum())
    rep["valid_hours_out"] = int(np.isfinite(v).sum())

    valid_idx = np.where(np.isfinite(v))[0]
    span_years = (full[valid_idx[-1]] - full[valid_idx[0]]).days / 365.25 if len(valid_idx) else 0.0
    rep["span_years"] = round(span_years, 2)
    if span_years < config.MIN_SPAN_YEARS:
        rep["status"] = "dropped (span)"
        return None, rep
    rep["status"] = "retained"
    st = dict(times=full.values.astype("datetime64[h]"), value=v, interp=interp)
    return st, rep


def _spike_v3b_flags(v, max_len=2, half_win=6, ratio=4.0, min_val=200.0, flank_ratio=0.25):
    """QC v3b: flag runs of 1..max_len finite hours where every value > min_val, both immediate
    flanks (s-1 and e) are finite and < flank_ratio * min(run), and every value in the run
    exceeds ratio * median(finite values in [s-half_win:s) U [e:e+half_win)).
    """
    n = len(v)
    flags = np.zeros(n, bool)
    for k in range(1, max_len + 1):
        for s in range(1, n - k):
            e = s + k
            run = v[s:e]
            if not np.all(np.isfinite(run)) or np.any(run <= min_val):
                continue
            left, right = v[s - 1], v[e]
            if not (np.isfinite(left) and np.isfinite(right)):
                continue
            rmin = float(np.min(run))
            if not (left < flank_ratio * rmin and right < flank_ratio * rmin):
                continue
            ctx = np.concatenate([v[max(0, s - half_win):s], v[e:min(n, e + half_win)]])
            fin = ctx[np.isfinite(ctx)]
            if len(fin) == 0:
                continue
            med = float(np.median(fin))
            if np.all(run > ratio * med):
                flags[s:e] = True
    return flags


def qc_station_v3(path, upper=980.0, fault_tol=0.01, halo_hours=3):
    """Frozen QC v3:
      1) Exact/near faults {985.0, 999.99} plus +-3h halo around each fault run -> NaN
      2) Bounds [0, 980] -> NaN
      3) Stuck runs > 6h (any value) and >= 3h (values > 200) -> NaN
      4) Spikes: v2 neighbour-ratio 1h spikes and v3b 1-2h rolling-median spikes -> NaN
      5) Short-gap linear interpolation (<= 6h) & MIN_SPAN_YEARS retention
    """
    idx, raw, n_slots, dups = parse_opencity(path)
    full = pd.date_range(idx[0], idx[-1], freq="h")
    s = pd.Series(raw, index=idx).reindex(full)
    v = s.values.astype(float)
    n = len(v)
    rep = dict(raw_slots=n_slots, nonnull_raw=int(np.isfinite(raw).sum()), duplicate_timestamps=dups)

    fault = np.isfinite(v) & (np.isclose(v, 985.0, atol=fault_tol) | np.isclose(v, 999.99, atol=fault_tol))
    halo_mask = np.zeros(n, bool)
    for fi in np.where(fault)[0]:
        lo, hi = max(0, fi - halo_hours), min(n, fi + halo_hours + 1)
        halo_mask[lo:hi] = True
    halo_only = halo_mask & (~fault) & np.isfinite(v)
    v[fault | halo_only] = np.nan
    rep["fault_985_999_removed"] = int(fault.sum())
    rep["fault_halo_3h_removed"] = int(halo_only.sum())

    oor = np.isfinite(v) & ((v < 0) | (v > upper))
    v[oor] = np.nan
    rep["out_of_range_removed"] = int(oor.sum())

    stuck6 = _stuck_flags(v, np.isfinite(v), config.MAX_STUCK)
    v[stuck6] = np.nan
    rep["stuck_gt6_removed"] = int(stuck6.sum())

    stuck_hi = _stuck_flags(v, np.isfinite(v) & (v > 200.0), 2)
    v[stuck_hi] = np.nan
    rep["stuck_ge3_gt200_removed"] = int(stuck_hi.sum())

    sp1 = _one_hour_spike_flags(v)
    v[sp1] = np.nan
    rep["spike_1h_removed"] = int(sp1.sum())

    sp3b = _spike_v3b_flags(v, max_len=2, half_win=6, ratio=4.0, min_val=200.0, flank_ratio=0.25)
    v[sp3b] = np.nan
    rep["spike_v3b_removed"] = int(sp3b.sum())

    rep["total_removed"] = int(
        rep["fault_985_999_removed"] + rep["fault_halo_3h_removed"]
        + rep["out_of_range_removed"] + rep["stuck_gt6_removed"]
        + rep["stuck_ge3_gt200_removed"] + rep["spike_1h_removed"]
        + rep["spike_v3b_removed"]
    )

    v, interp = _fill_short_gaps(v, config.MAX_GAP)
    rep["interpolated_hours"] = int(interp.sum())
    rep["valid_hours_out"] = int(np.isfinite(v).sum())

    valid_idx = np.where(np.isfinite(v))[0]
    span_years = (full[valid_idx[-1]] - full[valid_idx[0]]).days / 365.25 if len(valid_idx) else 0.0
    rep["span_years"] = round(span_years, 2)
    if span_years < config.MIN_SPAN_YEARS:
        rep["status"] = "dropped (span)"
        return None, rep
    rep["status"] = "retained"
    st = dict(times=full.values.astype("datetime64[h]"), value=v, interp=interp)
    return st, rep


def list_station_files(data_dir):
    """data/<City>/<Station>.csv only. Refuses to run if train/test duplicate folders exist."""
    files = []
    bad = []
    for root, _, names in os.walk(data_dir):
        for n in names:
            if n.lower().endswith(".csv"):
                p = os.path.join(root, n)
                rel = os.path.relpath(p, data_dir).split(os.sep)
                if len(rel) != 2:
                    bad.append(p)
                elif rel[0].lower() in ("train", "test"):
                    bad.append(p)
                else:
                    files.append((rel[0], os.path.splitext(n)[0], p))
    if bad:
        raise SystemExit("Unexpected CSV layout (expected data/<City>/<Station>.csv only):\n  " + "\n  ".join(bad))
    return sorted(files)

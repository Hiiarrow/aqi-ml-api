"""
Leak-free fold builders for P1 (LOCO) and P2 (2023 temporal holdout),
with optional time-<=t weather channels (openmeteo or cpcb) and direct/residual scaling.
"""
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config


def _time_feats(times):
    ti = pd.DatetimeIndex(times)
    hr = ti.hour.values.astype(np.float32)
    mo = (ti.month.values - 1).astype(np.float32)
    return np.stack([
        np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
        np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12),
    ], axis=1).astype(np.float32)


def _align_weather(city, name, times, weather_mode, weather_df):
    """Return (W[n, Cw], cols, has_miss_mask) aligned to st['times'] at <=t only."""
    if weather_mode == "none" or weather_df is None:
        return None, [], False
    t_idx = pd.to_datetime(times)
    if weather_mode == "openmeteo":
        wcols = [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "surface_pressure", "shortwave_radiation",
            "precipitation", "boundary_layer_height",
        ]
        sub = weather_df[weather_df["city"] == city].drop_duplicates("hour").set_index("hour").reindex(t_idx)
        rad = np.deg2rad(sub["wind_direction_10m"].values.astype(np.float32))
        arr = np.column_stack([
            sub["temperature_2m"].values,
            sub["relative_humidity_2m"].values,
            sub["wind_speed_10m"].values,
            np.sin(rad),
            np.cos(rad),
            sub["surface_pressure"].values,
            sub["shortwave_radiation"].values,
            sub["precipitation"].values,
            sub["boundary_layer_height"].values,
        ]).astype(np.float32)
        names = ["temp", "rh", "ws", "wd_sin", "wd_cos", "pres", "rad", "precip", "pblh"]
        return arr, names, False
    if weather_mode == "cpcb":
        wcols = ["at", "rh", "ws", "wd_sin", "wd_cos", "sr"]
        sub = (weather_df[(weather_df["city"] == city) & (weather_df["station"] == name)]
               .drop_duplicates("hour").set_index("hour").reindex(t_idx))
        # Interpolate short <=6h weather gaps before windowing (matches PM2.5 QC max_gap)
        vals = sub[wcols].interpolate(method="linear", limit=config.MAX_GAP, limit_area="inside").values.astype(np.float32)
        miss = np.isnan(vals[:, [0, 1, 2, 5]]).astype(np.float32)  # missing flags for at, rh, ws, sr
        return np.concatenate([vals, miss], axis=1).astype(np.float32), wcols + ["at_na", "rh_na", "ws_na", "sr_na"], True
    raise ValueError(weather_mode)


def station_windows(key, city, name, st, L, h, weather_mode="none", weather_df=None):
    v, t = st["value"], st["times"]
    n = len(v)
    valid = np.isfinite(v)
    cs = np.concatenate([[0], np.cumsum(valid)])
    idx = np.arange(L - 1, n - h)
    ok = (cs[idx + 1] - cs[idx + 1 - L] == L) & valid[idx + h] & ~st["interp"][idx + h]
    idx = idx[ok]
    if len(idx) == 0:
        return None
    tf = _time_feats(t)
    lag = sliding_window_view(v, L)[idx - L + 1].astype(np.float32)
    tfs = sliding_window_view(tf, L, axis=0)[idx - L + 1].transpose(0, 2, 1)  # N, L, 4
    tt = pd.DatetimeIndex(t[idx + h])

    out = dict(
        lag=lag,
        tf_seq=np.ascontiguousarray(tfs),
        tf_tgt=tf[idx + h],
        y=v[idx + h].astype(np.float32),
        y_curr=v[idx].astype(np.float32),
        rise3=(v[idx] - v[idx - 3]).astype(np.float32),
        month=tt.month.values,
        times=t[idx + h],
        block=np.array([f"{key}|{d}" for d in tt.strftime("%Y-%m-%d")]),
        station=np.array([key] * len(idx)),
    )
    W, _, _ = _align_weather(city, name, t, weather_mode, weather_df)
    if W is not None:
        # Strictly <= t: window [idx - L + 1 .. idx] ONLY
        w_seq = sliding_window_view(W, L, axis=0)[idx - L + 1].transpose(0, 2, 1)  # N, L, Cw
        out["w_seq"] = np.ascontiguousarray(w_seq, dtype=np.float32)
    return out


def _cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _slice_mask(w, mask):
    return {k: a[mask] for k, a in w.items()}


def build_loco_fold(clean, heldout, L=24, h=1, weather_mode="none", weather_df=None, gap_buffer=True):
    tr, va, te = [], [], []
    gap = np.timedelta64(h + L, "h") if gap_buffer else np.timedelta64(0, "h")
    for city, stations in sorted(clean.items()):
        for name, st in sorted(stations.items()):
            key = f"{city}/{name}"
            w = station_windows(key, city, name, st, L, h, weather_mode, weather_df)
            if w is None:
                continue
            if city == heldout:
                te.append(w)
            else:
                n = len(w["y"])
                nv = int(n * config.VAL_FRACTION)
                val_start = w["times"][n - nv]
                m_va = w["times"] >= val_start
                m_tr = w["times"] <= (val_start - gap)
                if m_tr.sum() > 0:
                    tr.append(_slice_mask(w, m_tr))
                if m_va.sum() > 0:
                    va.append(_slice_mask(w, m_va))
    tr, va, te = _cat(tr), _cat(va), _cat(te)

    tr_st, va_st, te_st = set(tr["station"]), set(va["station"]), set(te["station"])
    assert tr_st.isdisjoint(te_st) and va_st.isdisjoint(te_st), "LEAKAGE: test stations in train/val"
    assert all(s.startswith(heldout + "/") for s in te_st), "test partition has other cities"
    assert not any(s.startswith(heldout + "/") for s in tr_st | va_st), "held-out city in train/val"
    return _finalize_fold(tr, va, te, L, h, weather_mode)


def build_p2_2023_fold(clean, L=24, h=1, weather_mode="none", weather_df=None):
    """P2: Train/val on 2017-2022 across all 24 stations, strict h+24 gap before val and before 2023 test."""
    cutoff_test = np.datetime64("2023-01-01T00", "h")
    gap = np.timedelta64(h + L, "h")
    tr, va, te = [], [], []
    for city, stations in sorted(clean.items()):
        for name, st in sorted(stations.items()):
            key = f"{city}/{name}"
            w = station_windows(key, city, name, st, L, h, weather_mode, weather_df)
            if w is None:
                continue
            m_te = w["times"] >= cutoff_test
            m_pre = w["times"] <= (cutoff_test - gap)
            if m_te.sum() == 0 or m_pre.sum() < 200:
                continue
            pre = _slice_mask(w, m_pre)
            m = len(pre["y"])
            nv = max(1, int(m * config.VAL_FRACTION))
            val_start = pre["times"][m - nv]
            m_va = pre["times"] >= val_start
            m_tr = pre["times"] <= (val_start - gap)
            assert pre["times"][m_tr].max() + gap <= pre["times"][m_va].min(), f"P2 train/val gap violation on {key}"
            assert pre["times"][m_va].max() + gap <= w["times"][m_te].min(), f"P2 val/test gap violation on {key}"
            tr.append(_slice_mask(pre, m_tr))
            va.append(_slice_mask(pre, m_va))
            te.append(_slice_mask(w, m_te))
    tr, va, te = _cat(tr), _cat(va), _cat(te)
    assert tr["times"].max() + gap <= te["times"].min()
    assert va["times"].max() + gap <= te["times"].min()
    assert te["times"].min() >= cutoff_test
    return _finalize_fold(tr, va, te, L, h, weather_mode)


def _finalize_fold(tr, va, te, L, h, weather_mode):
    mu = float(tr["lag"].mean(dtype=np.float64))
    sd = float(tr["lag"].std(dtype=np.float64))
    tr_d = (tr["y"] - tr["y_curr"]).astype(np.float64)
    d_mu = float(tr_d.mean())
    d_sd = float(tr_d.std())

    w_stats = None
    if "w_seq" in tr:
        w = tr["w_seq"]  # N, L, Cw
        w_mu = np.nanmean(w, axis=(0, 1), dtype=np.float64, keepdims=True).astype(np.float32)
        w_sd = np.nanstd(w, axis=(0, 1), dtype=np.float64, keepdims=True).astype(np.float32)
        w_sd = np.where(w_sd < 1e-6, 1.0, w_sd).astype(np.float32)
        if weather_mode == "cpcb":
            # Do not shift/scale the last 4 binary *_na columns
            w_mu[..., -4:] = 0.0
            w_sd[..., -4:] = 1.0
        w_stats = (w_mu, w_sd)

    return dict(train=tr, val=va, test=te, mu=mu, sd=sd, d_mu=d_mu, d_sd=d_sd,
                L=L, h=h, weather_mode=weather_mode, w_stats=w_stats)


def extract_features(part, fold, target_mode="direct"):
    mu, sd = fold["mu"], fold["sd"]
    pm = ((part["lag"] - mu) / sd).astype(np.float32)  # N, L
    flat = np.concatenate([pm, part["tf_tgt"]], axis=1)
    seq_parts = [pm[:, :, None], part["tf_seq"]]

    if "w_seq" in part:
        w_mu, w_sd = fold["w_stats"]
        wn = ((part["w_seq"] - w_mu) / w_sd).astype(np.float32)
        wn = np.nan_to_num(wn, nan=0.0)
        seq_parts.append(wn)
        # Flat summary weather features strictly <= t: value at t, 24h mean, 6h change
        w_t = wn[:, -1, :]
        w_m24 = wn.mean(axis=1)
        w_d6 = wn[:, -1, :] - wn[:, -7, :]
        flat = np.concatenate([flat, w_t, w_m24, w_d6], axis=1)

    seq = np.concatenate(seq_parts, axis=2).astype(np.float32)
    if target_mode == "direct":
        y_sc = ((part["y"] - mu) / sd).astype(np.float32)
        inv = lambda p: p * sd + mu  # noqa: E731
    elif target_mode == "residual":
        d_mu, d_sd = fold["d_mu"], fold["d_sd"]
        y_sc = (((part["y"] - part["y_curr"]) - d_mu) / d_sd).astype(np.float32)
        inv = lambda p: part["y_curr"] + (p * d_sd + d_mu)  # noqa: E731
    else:
        raise ValueError(target_mode)
    return flat, seq, y_sc, inv

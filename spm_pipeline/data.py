"""Windowing and leave-one-city-out split, with leakage assertions."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import config


def _time_feats(times):
    ti = pd.DatetimeIndex(times)
    hr = ti.hour.values.astype(float)
    mo = (ti.month.values - 1).astype(float)
    return np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                     np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)], axis=1).astype(np.float32)


def station_windows(key, st, L, h):
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
    tfs = sliding_window_view(tf, L, axis=0)[idx - L + 1].transpose(0, 2, 1)  # N,L,4
    tt = pd.DatetimeIndex(t[idx + h])
    return dict(
        lag=lag, tf_seq=np.ascontiguousarray(tfs), tf_tgt=tf[idx + h],
        y=v[idx + h].astype(np.float32), y_curr=v[idx].astype(np.float32),
        rise3=(v[idx] - v[idx - 3]).astype(np.float32),
        month=tt.month.values, times=t[idx + h],
        block=np.array([f"{key}|{d}" for d in tt.strftime("%Y-%m-%d")]),
        station=np.array([key] * len(idx)),
    )


def _cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def build_fold(clean, heldout, L=None, h=None):
    L = L or config.LOOKBACK
    h = h or config.HORIZON
    tr, va, te = [], [], []
    for city, stations in clean.items():
        for name, st in stations.items():
            key = f"{city}/{name}"
            w = station_windows(key, st, L, h)
            if w is None:
                continue
            if city == heldout:
                te.append(w)
            else:
                n = len(w["y"])
                nv = int(n * config.VAL_FRACTION)
                tr.append({k: a[: n - nv] for k, a in w.items()})
                va.append({k: a[n - nv:] for k, a in w.items()})
    tr, va, te = _cat(tr), _cat(va), _cat(te)

    tr_st, va_st, te_st = set(tr["station"]), set(va["station"]), set(te["station"])
    assert tr_st.isdisjoint(te_st) and va_st.isdisjoint(te_st), "LEAKAGE: test stations in train/val"
    assert all(s.startswith(heldout + "/") for s in te_st), "test partition has other cities"
    assert not any(s.startswith(heldout + "/") for s in tr_st | va_st), "held-out city in train/val"

    mu, sd = float(tr["lag"].mean()), float(tr["lag"].std())
    return dict(train=tr, val=va, test=te, mu=mu, sd=sd, L=L, h=h)


def features(part, mu, sd):
    pm = ((part["lag"] - mu) / sd).astype(np.float32)
    flat = np.concatenate([pm, part["tf_tgt"]], axis=1)
    seq = np.concatenate([pm[:, :, None], part["tf_seq"]], axis=2)
    y = ((part["y"] - mu) / sd).astype(np.float32)
    return flat, seq, y

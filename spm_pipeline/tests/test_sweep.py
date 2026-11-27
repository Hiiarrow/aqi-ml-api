"""Leakage, residual scaling, weather alignment, and CNN-LSTM tests for sweep/."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))

from models_sweep import build_torch_model
from splits import build_loco_fold, build_p2_2023_fold, extract_features


def _mock_clean_and_weather():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-01 00:00", "2023-06-30 23:00", freq="h")
    n = len(idx)
    clean = {}
    om_rows, cpcb_rows = [], []
    for city, mu in [("Delhi", 100.0), ("Mumbai", 40.0), ("Chennai", 25.0)]:
        clean[city] = {}
        om_rows.append(pd.DataFrame({
            "city": city, "hour": idx,
            "temperature_2m": 25 + rng.normal(0, 3, n),
            "relative_humidity_2m": 65 + rng.normal(0, 10, n),
            "wind_speed_10m": np.abs(rng.normal(2, 1, n)),
            "wind_direction_10m": rng.uniform(0, 360, n),
            "surface_pressure": 1005 + rng.normal(0, 3, n),
            "shortwave_radiation": np.maximum(0, rng.normal(200, 100, n)),
            "precipitation": np.zeros(n),
            "boundary_layer_height": 500 + rng.normal(0, 100, n),
        }))
        for st in ["S1", "S2"]:
            clean[city][st] = {
                "times": idx.values.astype("datetime64[h]"),
                "value": np.maximum(5.0, mu + rng.normal(0, 10, n)).astype(float),
                "interp": np.zeros(n, bool),
            }
            cpcb_rows.append(pd.DataFrame({
                "city": city, "station": st, "hour": idx,
                "at": np.nan if city == "Chennai" else (25 + rng.normal(0, 3, n)),
                "rh": 60 + rng.normal(0, 10, n),
                "ws": np.abs(rng.normal(2, 1, n)),
                "wd_sin": rng.normal(0, 0.5, n),
                "wd_cos": rng.normal(0, 0.5, n),
                "sr": np.maximum(0, rng.normal(150, 80, n)),
            }))
    return clean, pd.concat(om_rows, ignore_index=True), pd.concat(cpcb_rows, ignore_index=True)


def test_p2_temporal_gap_and_residual_roundtrip():
    clean, om_df, cpcb_df = _mock_clean_and_weather()
    for h in (1, 24):
        fold = build_p2_2023_fold(clean, L=24, h=h, weather_mode="openmeteo", weather_df=om_df)
        gap = np.timedelta64(h + 24, "h")
        assert fold["val"]["times"].max() + gap <= fold["test"]["times"].min()
        assert fold["test"]["times"].min() >= np.datetime64("2023-01-01T00", "h")
        Xtr_f, Xtr_s, ytr_sc, inv_tr = extract_features(fold["train"], fold, "residual")
        assert np.all(np.isfinite(Xtr_f)) and np.all(np.isfinite(Xtr_s))
        assert np.allclose(inv_tr(ytr_sc), fold["train"]["y"], atol=1e-4)


def test_cpcb_weather_missing_city_handling_and_cnn_lstm():
    import torch
    clean, _, cpcb_df = _mock_clean_and_weather()
    fold = build_loco_fold(clean, heldout="Delhi", L=24, h=1, weather_mode="cpcb", weather_df=cpcb_df)
    Xtr_f, Xtr_s, ytr_sc, inv_tr = extract_features(fold["train"], fold, "direct")
    assert np.all(np.isfinite(Xtr_f)) and np.all(np.isfinite(Xtr_s))
    assert np.allclose(inv_tr(ytr_sc), fold["train"]["y"], atol=1e-4)
    # Verify Chennai at_na column == 1.0 even after normalization
    chennai_mask = np.array([s.startswith("Chennai/") for s in fold["train"]["station"]])
    at_na_chan = Xtr_s[chennai_mask, :, -4]
    assert np.all(at_na_chan == 1.0), "Missingness indicator broken for Chennai!"

    m = build_torch_model("cnn_lstm", Xtr_s.shape[2], hidden=64)
    y = m(torch.from_numpy(Xtr_s[:16]))
    assert y.shape == (16,)


if __name__ == "__main__":
    test_p2_temporal_gap_and_residual_roundtrip()
    test_cpcb_weather_missing_city_handling_and_cnn_lstm()
    print("ALL SWEEP TESTS PASSED")

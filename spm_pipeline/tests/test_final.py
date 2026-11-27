"""Unit tests for final_study.py: DATE blocks, baselines, P2_2024 BKC handling, and zero-leakage."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluate as E  # noqa: E402
import final_study as FS  # noqa: E402


def _make_mock_clean(include_2024=True, drop_bkc_2024=True):
    rng = np.random.default_rng(42)
    end = "2024-12-31 23:00" if include_2024 else "2023-12-31 23:00"
    idx_full = pd.date_range("2022-01-01 00:00", end, freq="h")
    idx_23 = pd.date_range("2022-01-01 00:00", "2023-12-28 23:00", freq="h")
    clean, om_rows = {}, []
    for city, base in [("Delhi", 120.0), ("Mumbai", 45.0), ("Chennai", 30.0)]:
        clean[city] = {}
        om_rows.append(pd.DataFrame({
            "city": city, "hour": idx_full,
            "temperature_2m": 25.0 + rng.normal(0, 3, len(idx_full)),
            "relative_humidity_2m": 60.0 + rng.normal(0, 10, len(idx_full)),
            "wind_speed_10m": np.abs(rng.normal(2.0, 1.0, len(idx_full))),
            "wind_direction_10m": rng.uniform(0, 360, len(idx_full)),
            "surface_pressure": 1005.0 + rng.normal(0, 2, len(idx_full)),
            "shortwave_radiation": np.maximum(0.0, rng.normal(200, 80, len(idx_full))),
            "precipitation": np.zeros(len(idx_full)),
            "boundary_layer_height": np.where(
                (idx_full >= pd.Timestamp("2024-01-01")) & (idx_full < pd.Timestamp("2024-07-01")),
                np.nan, 400.0 + rng.normal(0, 50, len(idx_full)),
            ) if include_2024 else 400.0 + rng.normal(0, 50, len(idx_full)),
        }))
        sts = ["BKC", "Powai"] if city == "Mumbai" else ["S1", "S2"]
        for st in sts:
            ix = idx_23 if (include_2024 and drop_bkc_2024 and city == "Mumbai" and st == "BKC") else idx_full
            n = len(ix)
            hrs = ix.hour.values
            sig = base + 15.0 * np.sin(2 * np.pi * hrs / 24.0) + rng.normal(0, 5, n)
            clean[city][st] = {
                "times": ix.values.astype("datetime64[h]"),
                "value": np.maximum(5.0, sig).astype(float),
                "interp": np.zeros(n, bool),
            }
    return clean, pd.concat(om_rows, ignore_index=True)


def test_date_blocks_and_same_hour_yesterday():
    clean, om = _make_mock_clean(include_2024=False)
    for h in (1, 6, 12, 24):
        f_none = FS.build_loco_fold_final(clean, "Delhi", L=24, h=h, weather_mode="none")
        f_om = FS.build_loco_fold_final(clean, "Delhi", L=24, h=h, weather_mode="openmeteo", weather_df=om)
        # Verify identical alignment between none & openmeteo
        assert np.array_equal(f_none["test"]["y"], f_om["test"]["y"])
        assert np.array_equal(f_none["test"]["times"], f_om["test"]["times"])
        assert np.array_equal(f_none["test"]["station"], f_om["test"]["station"])
        assert np.array_equal(f_none["test"]["block_date"], f_om["test"]["block_date"])

        # DATE block check: n_blocks == unique calendar dates across ALL stations combined (no station|date)
        te_dates = pd.DatetimeIndex(f_none["test"]["times"]).strftime("%Y-%m-%d")
        assert np.array_equal(f_none["test"]["block_date"], te_dates)
        assert len(np.unique(f_none["test"]["block_date"])) == len(np.unique(te_dates))

        if h >= 6:
            assert "same_hour_yesterday" in f_none["test"]
            shy = f_none["test"]["same_hour_yesterday"]
            if h == 24:
                assert np.allclose(shy, f_none["test"]["y_curr"])
            else:
                # Verify exact value 24h before target time on first station (Delhi/S1)
                st0 = clean["Delhi"]["S1"]
                s_map = pd.Series(st0["value"], index=st0["times"])
                m0 = f_none["test"]["station"] == "Delhi/S1"
                t_tgt = f_none["test"]["times"][m0]
                expected = s_map.reindex(t_tgt - np.timedelta64(24, "h")).values
                assert np.allclose(shy[m0], expected), f"same_hour_yesterday mismatch at h={h}"
        else:
            assert "same_hour_yesterday" not in f_none["test"]


def test_climatology_train_only():
    clean, _ = _make_mock_clean(include_2024=False)
    fold = FS.build_loco_fold_final(clean, "Delhi", L=24, h=1)
    # Shift Delhi test y hugely; climatology must remain invariant (strictly train-only)
    fold_orig = fold
    va1, te1, m1 = FS.compute_hour_climatology(fold_orig)
    fold["val"]["y"] = fold["val"]["y"] + 999.0
    fold["test"]["y"] = fold["test"]["y"] + 999.0
    va2, te2, m2 = FS.compute_hour_climatology(fold)
    assert np.allclose(m1, m2) and np.allclose(va1, va2) and np.allclose(te1, te2)


def test_p2_2024_bkc_and_purge_and_om_blh():
    clean, om = _make_mock_clean(include_2024=True, drop_bkc_2024=True)
    fold = FS.build_p2_temporal_fold(clean, test_year=2024, L=24, h=24, weather_mode="openmeteo", weather_df=om)
    gap = np.timedelta64(24 + 24, "h")
    assert fold["train"]["times"].max() + gap <= fold["test"]["times"].min()
    assert fold["val"]["times"].max() + gap <= fold["test"]["times"].min()
    assert fold["test"]["times"].min() >= np.datetime64("2024-01-01T00", "h")
    assert "Mumbai/BKC" in set(fold["train"]["station"])
    assert "Mumbai/BKC" not in set(fold["test"]["station"])
    assert "Mumbai/Powai" in set(fold["test"]["station"])
    # Check finite features even with 2024 BLH NaNs (standardized -> 0.0)
    from sweep.splits import extract_features
    Xtr_f, Xtr_s, _, _ = extract_features(fold["train"], fold, "direct")
    Xte_f, Xte_s, _, _ = extract_features(fold["test"], fold, "direct")
    assert np.all(np.isfinite(Xtr_f)) and np.all(np.isfinite(Xtr_s))
    assert np.all(np.isfinite(Xte_f)) and np.all(np.isfinite(Xte_s))


if __name__ == "__main__":
    test_date_blocks_and_same_hour_yesterday()
    test_climatology_train_only()
    test_p2_2024_bkc_and_purge_and_om_blh()
    print("ALL FINAL STUDY UNIT TESTS PASSED")

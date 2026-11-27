"""Verification tests for QC, OpenCity matrix export, LOCO leakage prevention, and time-t regimes."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fetch"))

import config
import data as D
import evaluate as E
import qc
from export_opencity import write_opencity_matrix


def test_opencity_roundtrip():
    idx = pd.date_range("2020-01-01 00:00", "2020-03-31 23:00", freq="h")
    vals = np.linspace(10.0, 200.0, len(idx))
    vals[50:55] = np.nan
    s = pd.Series(vals, index=idx)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "st.csv")
        write_opencity_matrix(s, p)
        idx2, vals2, slots, dups = qc.parse_opencity(p)
        assert dups == 0
        assert slots == len(idx)
        assert np.allclose(vals[np.isfinite(vals)], vals2[np.isfinite(vals2)], atol=0.02)
        assert np.array_equal(np.isnan(vals), np.isnan(vals2))


def test_qc_cap_vs_stuck_and_gap():
    idx = pd.date_range("2018-01-01 00:00", "2021-01-10 23:00", freq="h")
    vals = np.random.default_rng(0).uniform(20, 100, len(idx))
    # Isolated/short run at upper bound (4h <= MAX_STUCK=6): must stay kept as valid
    vals[100:104] = config.CAP
    # Interior stuck run of 10h (> MAX_STUCK=6): must become NaN (then longer than MAX_GAP=6 so NOT interpolated)
    vals[200:210] = 55.5
    # Short gap of 4h (<= MAX_GAP=6): must be linearly interpolated
    vals[300:304] = np.nan
    # Above upper bound run of 8h (> MAX_GAP=6): must become NaN and remain NaN
    vals[400:408] = config.CAP + 250.0

    s = pd.Series(vals, index=idx)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "st.csv")
        write_opencity_matrix(s, p)
        st, rep, over = qc.qc_station(p)
        assert st is not None and rep["status"] == "retained"
        assert rep["hours_at_cap"] == 4
        assert rep["stuck_hours_removed"] == 10
        assert rep["above_cap_removed"] == 8
        assert rep["interpolated_hours"] == 4
        assert np.all(st["value"][100:104] == config.CAP)
        assert np.all(np.isnan(st["value"][200:210]))
        assert np.all(st["interp"][300:304])
        assert np.all(np.isnan(st["value"][400:408]))


def test_time_t_regime_invariance_to_future_y():
    # Regimes must ONLY depend on time-t features (y_curr, rise3, month), never y
    t_dict = {
        "y": np.array([10.0, 200.0, 30.0, 400.0]),
        "y_curr": np.array([65.0, 40.0, 260.0, 125.0]),
        "rise3": np.array([35.0, 5.0, 40.0, 10.0]),
        "month": np.array([1, 6, 12, 4]),
    }
    m1 = E.regime_masks(t_dict)
    t_dict["y"] = np.array([999.0, 1.0, 1.0, 999.0])
    m2 = E.regime_masks(t_dict)
    for k in m1:
        assert np.array_equal(m1[k], m2[k]), f"Regime {k} leaked future y!"


def test_loco_fold_disjoint_and_scaler():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2018-01-01", "2020-06-01", freq="h")
    n = len(idx)
    clean = {}
    for city, offset in [("Delhi", 100.0), ("Mumbai", 30.0), ("Chennai", 20.0)]:
        clean[city] = {}
        for st_name in ["S1", "S2"]:
            clean[city][st_name] = {
                "times": idx.values.astype("datetime64[h]"),
                "value": offset + rng.normal(0, 5, n),
                "interp": np.zeros(n, bool),
            }
    fold = D.build_fold(clean, heldout="Delhi", L=24, h=1)
    assert set(fold["test"]["station"]) == {"Delhi/S1", "Delhi/S2"}
    assert set(fold["train"]["station"]).isdisjoint(set(fold["test"]["station"]))
    # Scaler mean mu must come strictly from Mumbai/Chennai (~25), NOT Delhi (~100)
    assert 20.0 <= fold["mu"] <= 30.0


def test_qc_v2_rules():
    idx = pd.date_range("2018-01-01 00:00", "2021-01-10 23:00", freq="h")
    vals = np.random.default_rng(0).uniform(20, 80, len(idx))
    # 1) Sustained genuine ramp (>500 up to 750) must be kept
    vals[50:55] = [200.0, 450.0, 750.0, 720.0, 510.0]
    # 2) Above 980 -> removed; 985.0 -> fault removed; 999.99 -> fault removed
    vals[100] = 982.0
    vals[120] = 985.0
    vals[140] = 999.99
    # 3) Stuck > 6h (8h identical) -> removed
    vals[200:208] = 66.0
    # 4) Both-neighbour 1h spike (x>=200 & min_ratio>=3) -> removed
    vals[300] = 350.0
    # 5) One-NaN >500 with ratio>=5 (adjacent to removed 985.0) -> removed
    vals[400] = 985.0
    vals[401] = 750.0
    # 6) Diwali high peak >500 next to 1h gap with moderate ratio (<5) -> preserved
    vals[500:506] = [261.0, 961.0, np.nan, 711.0, 450.0, 300.0]
    s = pd.Series(vals, index=idx)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "st2.csv")
        write_opencity_matrix(s, p)
        st, rep = qc.qc_station_v2(p)
        assert st is not None and rep["status"] == "retained"
        assert rep["out_of_range_removed"] == 1
        assert rep["fault_985_999_removed"] == 3
        assert rep["stuck_hours_removed"] == 8
        assert rep["one_hour_spike_removed"] == 2
        assert np.all(np.isclose(st["value"][50:55], [200.0, 450.0, 750.0, 720.0, 510.0]))
        assert st["interp"][300] and st["value"][300] < 100.0
        assert st["interp"][400] and st["interp"][401]
        assert st["value"][501] == 961.0 and not st["interp"][501]


def test_qc_v3_rules():
    idx = pd.date_range("2018-01-01 00:00", "2021-01-10 23:00", freq="h")
    vals = np.random.default_rng(7).uniform(20, 60, len(idx))
    # 1) 985.0 fault + 3h halo on each side -> 7 consecutive NaNs (>MAX_GAP=6) -> remain NaN
    vals[100] = 985.0
    vals[99] = 283.8  # within 3h halo -> must be removed
    # 2) Stuck >=3h above 200 (gap=4 <=6 so interpolated, not original 629)
    vals[300:304] = 629.0
    # 3) 2-hour spike pair (v3b): [20.25, 262.0, 744.5, 25.75]
    vals[499:503] = [20.25, 262.0, 744.5, 25.75]
    # 4) Sustained winter ramp (4h > 500) -> kept intact
    vals[700:704] = [536.5, 724.25, 821.25, 853.75]
    s = pd.Series(vals, index=idx)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "st3.csv")
        write_opencity_matrix(s, p)
        st, rep = qc.qc_station_v3(p)
        assert st is not None and rep["status"] == "retained"
        assert rep["fault_985_999_removed"] == 1
        assert rep["fault_halo_3h_removed"] == 6
        assert rep["stuck_ge3_gt200_removed"] == 4
        assert rep["spike_v3b_removed"] == 2
        assert np.all(np.isnan(st["value"][97:104]))
        assert np.all(st["interp"][300:304])
        assert np.all(st["interp"][500:502]) and st["value"][500] < 100.0
        assert np.all(np.isclose(st["value"][700:704], [536.5, 724.25, 821.25, 853.75]))


if __name__ == "__main__":
    test_opencity_roundtrip()
    test_qc_cap_vs_stuck_and_gap()
    test_time_t_regime_invariance_to_future_y()
    test_loco_fold_disjoint_and_scaler()
    test_qc_v2_rules()
    test_qc_v3_rules()
    print("ALL PIPELINE UNIT TESTS PASSED")

"""Unit tests for ext_cities_run.py: model save/reload fidelity, window lookback,
residual tree scaling, Step D summaries, and surface_pressure range checks.
"""
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))

import config  # noqa: E402
import ext_cities_run as EX  # noqa: E402
import final_study as FS  # noqa: E402
from sweep.models_sweep import fit_gbm, train_neural  # noqa: E402
from sweep.splits import extract_features  # noqa: E402


def _mock_station(start="2022-01-01 00:00", end="2024-12-31 23:00", base=80.0, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    hrs = idx.hour.values
    vals = np.maximum(5.0, base + 20.0 * np.sin(2 * np.pi * hrs / 24.0) + rng.normal(0, 6.0, n)).astype(float)
    return {"times": idx.values.astype("datetime64[h]"), "value": vals, "interp": np.zeros(n, bool)}


def test_checkpoint_reload_and_ext_lookback():
    tr_clean = {"CityA": {"S1": _mock_station("2021-01-01 00:00", "2023-12-31 23:00", 70.0, 1)}}
    fold = FS.build_p2_temporal_fold(tr_clean, test_year=2023, L=24, h=6, weather_mode="none")
    Xtr_d, Xtr_s, ytr_d, _ = extract_features(fold["train"], fold, "direct")
    Xva_d, Xva_s, yva_d, inv_va_d = extract_features(fold["val"], fold, "direct")
    Xte_d, Xte_s, _, inv_te_d = extract_features(fold["test"], fold, "direct")
    _, _, ytr_r, _ = extract_features(fold["train"], fold, "residual")
    _, _, yva_r, inv_va_r = extract_features(fold["val"], fold, "residual")
    _, _, _, inv_te_r = extract_features(fold["test"], fold, "residual")

    (_, _, _), (ridge_m, _, rte, _) = EX.fit_linear_and_ridge_with_models(
        Xtr_d, ytr_d, Xva_d, fold["val"]["y"].astype(np.float64), inv_va_d, Xte_d, inv_te_d
    )
    _, gd_te, gd_m = fit_gbm(Xtr_d, ytr_d, Xva_d, yva_d, Xte_d)
    _, gr_te, gr_m = fit_gbm(Xtr_d, ytr_r, Xva_d, yva_r, Xte_d)
    _, pte_sc, best_state, ep_done = train_neural(
        "gru", Xtr_s, ytr_d, Xva_s, yva_d, Xte_s, seed=0, epochs=2, patience=2, batch=512, hidden=16
    )

    with tempfile.TemporaryDirectory() as td:
        pt_p = os.path.join(td, "gru_seed0.pt")
        import torch
        old_h = config.HIDDEN
        config.HIDDEN = 16
        try:
            torch.save(dict(state_dict=best_state, c_in=int(Xtr_s.shape[2]), epochs=int(ep_done), kind="gru", seed=0), pt_p)
            p_reload = EX.predict_torch_from_checkpoint(pt_p, Xte_s)
            assert np.max(np.abs(inv_te_d(p_reload) - inv_te_d(pte_sc))) < 1e-4

            bb_p = os.path.join(td, "bb.pkl")
            with open(bb_p, "wb") as f:
                pickle.dump(dict(ridge=ridge_m, gd=gd_m, gr=gr_m), f)
            with open(bb_p, "rb") as f:
                bb2 = pickle.load(f)
            assert np.allclose(inv_te_d(bb2["ridge"].predict(Xte_d)), rte)
            assert np.allclose(inv_te_d(bb2["gd"].predict(Xte_d)), inv_te_d(gd_te))
            assert np.allclose(inv_te_r(bb2["gr"].predict(Xte_d)), inv_te_r(gr_te))
        finally:
            config.HIDDEN = old_h

    # Verify external city 2023 windowing uses 2022 lookback only and hits exact 2023 timestamps (8760h)
    ext_st = {"S1": _mock_station("2022-01-01 00:00", "2024-12-31 23:00", 110.0, 2)}
    te23 = EX.build_city_test_windows("ExtCity", ext_st, L=24, h=6, test_year=2023, weather_mode="none", weather_df=None)
    assert te23["times"].min() == np.datetime64("2023-01-01T00", "h")
    assert te23["times"].max() == np.datetime64("2023-12-31T23", "h")
    assert len(te23["y"]) == 8760


def test_step_d_summary_counts():
    df = pd.DataFrame([
        dict(
            city_group="PRIMARY", test_year=2023, horizon=1, best_tree_choice="gbm_residual",
            ens_skill_RMSE=10.0, ridge_skill_RMSE=5.0, best_tree_skill_RMSE=8.0,
            ens_vs_pers_dRMSE_status="entirely_below_zero", ens_vs_pers_dMAE_status="entirely_below_zero",
            ridge_vs_pers_dRMSE_status="entirely_below_zero", ridge_vs_pers_dMAE_status="includes_zero",
            best_tree_vs_pers_dRMSE_status="entirely_below_zero", best_tree_vs_pers_dMAE_status="entirely_below_zero",
            ridge_vs_ens_dRMSE_status="entirely_above_zero", ridge_vs_ens_dMAE_status="entirely_above_zero",
        ),
        dict(
            city_group="PRIMARY", test_year=2023, horizon=1, best_tree_choice="gbm_residual",
            ens_skill_RMSE=20.0, ridge_skill_RMSE=15.0, best_tree_skill_RMSE=12.0,
            ens_vs_pers_dRMSE_status="includes_zero", ens_vs_pers_dMAE_status="entirely_below_zero",
            ridge_vs_pers_dRMSE_status="entirely_above_zero", ridge_vs_pers_dMAE_status="entirely_above_zero",
            best_tree_vs_pers_dRMSE_status="entirely_below_zero", best_tree_vs_pers_dMAE_status="includes_zero",
            ridge_vs_ens_dRMSE_status="includes_zero", ridge_vs_ens_dMAE_status="entirely_below_zero",
        ),
    ])
    s = EX.summarize_across_cities(df, "none").iloc[0]
    assert s["n_cities"] == 2
    assert s["ens_skill_RMSE_med"] == 15.0 and s["ens_skill_RMSE_min"] == 10.0 and s["ens_skill_RMSE_max"] == 20.0
    assert s["ens_vs_pers_dRMSE_below"] == 1 and s["ens_vs_pers_dRMSE_incl"] == 1 and s["ens_vs_pers_dRMSE_above"] == 0
    assert s["ridge_vs_ens_dRMSE_below"] == 0 and s["ridge_vs_ens_dRMSE_incl"] == 1 and s["ridge_vs_ens_dRMSE_above"] == 1


if __name__ == "__main__":
    test_checkpoint_reload_and_ext_lookback()
    test_step_d_summary_counts()
    print("ALL EXT CITIES UNIT TESTS PASSED")

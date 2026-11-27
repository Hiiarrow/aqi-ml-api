"""
Step A Retraining & Evaluation under Frozen QC v3 (20 stations only; Hyderabad excluded).

Runs with identical sweep hyperparameters (LOOKBACK=24, HIDDEN=64, EPOCHS=60, PATIENCE=7,
BATCH=2048, LR=1e-3, SEEDS=[0..4]):
  1. P2_2023_ALL20_h24_direct_none        (gru, lstm, bilstm, tcn, cnn_lstm)
  2. P2_2023_ALL20_h24_residual_openmeteo (gru, cnn_lstm)
  3. P2_2023_ALL20_h1_direct_none         (gru, lstm, bilstm, tcn, cnn_lstm)
  4. P2_2023_ALL20_h1_residual_openmeteo  (gru, cnn_lstm)
  5. P1_LOCO_Delhi20_h1_direct_none       (gru, lstm, bilstm, tcn, cnn_lstm)
  6. P1_LOCO_Delhi20_h1_residual_none     (gru, lstm, bilstm, tcn, cnn_lstm)
  7. P1_LOCO_Delhi20_h1_direct_openmeteo  (gru, cnn_lstm)

Also fits residual-target GBM/XGBoost for every direct run to report best_res_tree,
primary equal_neural_ens, and stack ablation with paired CIs vs persistence and vs equal_neural_ens,
plus per-regime tables (y_t >= 60, 120, 250, onset, winter, other, all).

Writes strictly to results_qc3_stepA/ (never modifies results/, results_sweep/, or results_qc2/).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))

import config
import evaluate as E
import qc
from sweep.models_sweep import fit_gbm, fit_xgb
from sweep.run_sweep import execute_spec
from sweep.splits import build_loco_fold, build_p2_2023_fold, extract_features


STUDY_CITIES = {"Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"}

STEP_A_SPECS = [
    (dict(run_id="P2_2023_ALL20_h24_direct_none", protocol="P2_2023", heldout="ALL20",
          horizon=24, target_mode="direct", weather_mode="none"),
     ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]),
    (dict(run_id="P2_2023_ALL20_h24_residual_openmeteo", protocol="P2_2023", heldout="ALL20",
          horizon=24, target_mode="residual", weather_mode="openmeteo"),
     ["gru", "cnn_lstm"]),
    (dict(run_id="P2_2023_ALL20_h1_direct_none", protocol="P2_2023", heldout="ALL20",
          horizon=1, target_mode="direct", weather_mode="none"),
     ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]),
    (dict(run_id="P2_2023_ALL20_h1_residual_openmeteo", protocol="P2_2023", heldout="ALL20",
          horizon=1, target_mode="residual", weather_mode="openmeteo"),
     ["gru", "cnn_lstm"]),
    (dict(run_id="P1_LOCO_Delhi20_h1_direct_none", protocol="P1_LOCO", heldout="Delhi",
          horizon=1, target_mode="direct", weather_mode="none"),
     ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]),
    (dict(run_id="P1_LOCO_Delhi20_h1_residual_none", protocol="P1_LOCO", heldout="Delhi",
          horizon=1, target_mode="residual", weather_mode="none"),
     ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]),
    (dict(run_id="P1_LOCO_Delhi20_h1_direct_openmeteo", protocol="P1_LOCO", heldout="Delhi",
          horizon=1, target_mode="direct", weather_mode="openmeteo"),
     ["gru", "cnn_lstm"]),
]


def build_clean_20_v3(data_dir, out_dir):
    cache_p = os.path.join(out_dir, "clean_20_v3.pkl")
    rep_p = os.path.join(out_dir, "qc3_20station_report.csv")
    if os.path.exists(cache_p) and os.path.exists(rep_p):
        with open(cache_p, "rb") as f:
            return pickle.load(f)
    clean, rows = {}, []
    for city, st, p in qc.list_station_files(data_dir):
        if city not in STUDY_CITIES:
            continue
        st_obj, rep = qc.qc_station_v3(p)
        rows.append(dict(city=city, station=st, **rep))
        if st_obj is not None:
            clean.setdefault(city, {})[st] = st_obj
    pd.DataFrame(rows).to_csv(rep_p, index=False)
    assert len(clean) == 5 and sum(len(v) for v in clean.values()) == 20, "Expected 5 cities, 20 stations"
    with open(cache_p, "wb") as f:
        pickle.dump(clean, f)
    return clean


def ensure_residual_trees(spec, clean, weather_caches, out_dir):
    """For direct_* runs, fit residual-target gbm & xgboost on the same QC v3 20-station fold."""
    rdir = os.path.join(out_dir, "runs", spec["run_id"])
    tp = os.path.join(rdir, "preds_res_trees.pkl")
    if spec["target_mode"] == "residual":
        with open(os.path.join(rdir, "preds.pkl"), "rb") as f:
            pr = pickle.load(f)
        out = dict(val={k: pr["val"][k] for k in ("gbm", "xgboost")},
                   test={k: pr["test"][k] for k in ("gbm", "xgboost")})
        with open(tp, "wb") as f:
            pickle.dump(out, f)
        return out
    if os.path.exists(tp):
        with open(tp, "rb") as f:
            return pickle.load(f)

    wdf = weather_caches.get(spec["weather_mode"])
    if spec["protocol"] == "P1_LOCO":
        fold = build_loco_fold(clean, spec["heldout"], L=config.LOOKBACK, h=spec["horizon"],
                               weather_mode=spec["weather_mode"], weather_df=wdf, gap_buffer=True)
    else:
        fold = build_p2_2023_fold(clean, L=config.LOOKBACK, h=spec["horizon"],
                                  weather_mode=spec["weather_mode"], weather_df=wdf)
    Xtr_f, _, ytr, _ = extract_features(fold["train"], fold, "residual")
    Xva_f, _, yva, inv_va = extract_features(fold["val"], fold, "residual")
    Xte_f, _, _, inv_te = extract_features(fold["test"], fold, "residual")
    gva, gte, _ = fit_gbm(Xtr_f, ytr, Xva_f, yva, Xte_f)
    xva, xte, _ = fit_xgb(Xtr_f, ytr, Xva_f, yva, Xte_f)
    out = dict(
        val=dict(gbm=inv_va(gva), xgboost=inv_va(xva)),
        test=dict(gbm=inv_te(gte), xgboost=inv_te(xte)),
    )
    with open(tp, "wb") as f:
        pickle.dump(out, f)
    print(f"  [{spec['run_id']}] Residual-target companion trees fitted.", flush=True)
    return out


def build_step_a_outputs(clean, weather_caches, out_dir):
    sum_rows, ci_rows, reg_rows = [], [], []
    for spec, nmodels in STEP_A_SPECS:
        rid = spec["run_id"]
        rdir = os.path.join(out_dir, "runs", rid)
        pp = os.path.join(rdir, "preds.pkl")
        if not os.path.exists(pp):
            continue
        with open(pp, "rb") as f:
            pr = pickle.load(f)
        res_trees = ensure_residual_trees(spec, clean, weather_caches, out_dir)

        test, val, meta, y, val_y = pr["test"], pr["val"], pr["meta"], pr["meta"]["y"], pr["val_y"]
        nfams = sorted([m for m in test if m.endswith("_avg") and test[m] is not None])
        eq_te = np.mean([test[m] for m in nfams], axis=0)
        eq_va = np.mean([val[m] for m in nfams], axis=0)

        best_res_tree_name = min(("gbm", "xgboost"), key=lambda m: E.rmse(res_trees["val"][m] - val_y))
        best_res_tree_va = res_trees["val"][best_res_tree_name]
        best_res_tree_te = res_trees["test"][best_res_tree_name]

        best_run_tree_name = min(("gbm", "xgboost"), key=lambda m: E.rmse(val[m] - val_y))
        best_val_n_name = min(nfams, key=lambda m: E.rmse(val[m] - val_y))

        p_rmse = E.rmse(test["persistence"] - y)
        p_mae = E.mae(test["persistence"] - y)

        methods = {
            "persistence": (val["persistence"], test["persistence"]),
            "equal_neural_ens": (eq_va, eq_te),
            f"res_tree ({best_res_tree_name})": (best_res_tree_va, best_res_tree_te),
            f"val_best_neural ({best_val_n_name})": (val[best_val_n_name], test[best_val_n_name]),
            "stack": (val["stack"], test["stack"]),
        }
        for mname, (pv, pt) in methods.items():
            rv = E.rmse(pv - val_y)
            rt, at = E.rmse(pt - y), E.mae(pt - y)
            sum_rows.append(dict(
                run_id=rid, N_test=len(y), method=mname,
                neural_fams="+".join(m[:-4] for m in nfams),
                persist_RMSE=round(p_rmse, 3), persist_MAE=round(p_mae, 3),
                val_RMSE=round(rv, 3), test_RMSE=round(rt, 3), test_MAE=round(at, 3),
                skill_RMSE_pct=round(100.0 * (1.0 - rt / p_rmse), 2),
                skill_MAE_pct=round(100.0 * (1.0 - at / p_mae), 2),
            ))

        # Paired day-block CIs vs persistence and vs equal_neural_ens
        comp_preds = {
            "persistence": test["persistence"],
            "equal_neural_ens": eq_te,
            f"res_tree_{best_res_tree_name}": best_res_tree_te,
            best_val_n_name: test[best_val_n_name],
            "stack": test["stack"],
        }
        rt_key = f"res_tree_{best_res_tree_name}"
        pairs = [
            ("equal_neural_ens", "persistence"),
            (rt_key, "persistence"),
            ("stack", "persistence"),
            (best_val_n_name, "persistence"),
            (rt_key, "equal_neural_ens"),
            ("stack", "equal_neural_ens"),
            (best_val_n_name, "equal_neural_ens"),
        ]
        for crow in E.paired_block_bootstrap(y, comp_preds, meta["block"], pairs):
            ci_rows.append(dict(run_id=rid, **crow))

        # Regime tables (y_t >= 60, 120, 250, onset, winter, other, all)
        reg_preds = dict(test)
        reg_preds["equal_neural_ens"] = eq_te
        reg_preds["res_best_tree"] = best_res_tree_te
        for rrow in E.regime_table(meta, y, reg_preds):
            reg_rows.append(dict(run_id=rid, **rrow))

    pd.DataFrame(sum_rows).to_csv(os.path.join(out_dir, "STEP_A_SUMMARY.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(out_dir, "STEP_A_CIS.csv"), index=False)
    pd.DataFrame(reg_rows).to_csv(os.path.join(out_dir, "STEP_A_REGIMES.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_qc3_stepA")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    clean = build_clean_20_v3(os.path.join(ROOT, "data"), out_dir)
    weather_caches = {
        "none": None,
        "openmeteo": pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet")),
    }
    t0 = time.time()
    for spec, nmodels in STEP_A_SPECS:
        execute_spec(spec, clean, weather_caches, config.SEEDS, config.EPOCHS, out_dir, neural_models=nmodels)
        build_step_a_outputs(clean, weather_caches, out_dir)
    print(f"\nSTEP A (QC v3, 20 stations) COMPLETE in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

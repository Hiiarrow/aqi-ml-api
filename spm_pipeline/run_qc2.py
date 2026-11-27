"""
Retrain selected sweep benchmarks end-to-end under QC v2 (both train & test):
  1. P2_2023_ALL_h24_direct_none        (5 neural families x 5 seeds)
  2. P2_2023_ALL_h1_direct_none         (5 neural families x 5 seeds)
  3. P1_LOCO_Delhi_h1_direct_none       (5 neural families x 5 seeds)
  4. P2_2023_ALL_h24_residual_openmeteo (gru + cnn_lstm x 5 seeds, val headline)
  5. P2_2023_ALL_h1_direct_openmeteo    (gru + cnn_lstm x 5 seeds, val headline)
  6. P1_LOCO_Delhi_h1_direct_openmeteo  (gru + cnn_lstm x 5 seeds, weather headline)

Writes strictly to results_qc2/ (never modifies results/ or results_sweep/).
"""
import argparse
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
from sweep.run_sweep import execute_spec


RUN_SPECS = [
    (dict(run_id="P2_2023_ALL_h24_direct_none", protocol="P2_2023", heldout="ALL",
          horizon=24, target_mode="direct", weather_mode="none"),
     ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]),
    (dict(run_id="P2_2023_ALL_h1_direct_none", protocol="P2_2023", heldout="ALL",
          horizon=1, target_mode="direct", weather_mode="none"),
     ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]),
    (dict(run_id="P1_LOCO_Delhi_h1_direct_none", protocol="P1_LOCO", heldout="Delhi",
          horizon=1, target_mode="direct", weather_mode="none"),
     ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]),
    (dict(run_id="P2_2023_ALL_h24_residual_openmeteo", protocol="P2_2023", heldout="ALL",
          horizon=24, target_mode="residual", weather_mode="openmeteo"),
     ["gru", "cnn_lstm"]),
    (dict(run_id="P2_2023_ALL_h1_direct_openmeteo", protocol="P2_2023", heldout="ALL",
          horizon=1, target_mode="direct", weather_mode="openmeteo"),
     ["gru", "cnn_lstm"]),
    (dict(run_id="P1_LOCO_Delhi_h1_direct_openmeteo", protocol="P1_LOCO", heldout="Delhi",
          horizon=1, target_mode="direct", weather_mode="openmeteo"),
     ["gru", "cnn_lstm"]),
]


def build_clean_v2(data_dir, out_dir):
    cache_p = os.path.join(out_dir, "clean_v2.pkl")
    rep_p = os.path.join(out_dir, "qc2_station_report.csv")
    if os.path.exists(cache_p) and os.path.exists(rep_p):
        with open(cache_p, "rb") as f:
            return pickle.load(f)
    clean, rows = {}, []
    for city, st, p in qc.list_station_files(data_dir):
        st_obj, rep = qc.qc_station_v2(p)
        rows.append(dict(city=city, station=st, **rep))
        if st_obj is not None:
            clean.setdefault(city, {})[st] = st_obj
    pd.DataFrame(rows).to_csv(rep_p, index=False)
    with open(cache_p, "wb") as f:
        pickle.dump(clean, f)
    return clean


def summarize_qc1_vs_qc2(v1_dir="results_sweep", v2_dir="results_qc2"):
    comp_rows, ci_rows = [], []
    for spec, _ in RUN_SPECS:
        rid = spec["run_id"]
        p2_path = os.path.join(ROOT, v2_dir, "runs", rid, "preds.pkl")
        p1_path = os.path.join(ROOT, v1_dir, "runs", rid, "preds.pkl")
        if not (os.path.exists(p2_path) and os.path.exists(p1_path)):
            continue
        for tag, pp in [("QC_v1", p1_path), ("QC_v2", p2_path)]:
            with open(pp, "rb") as f:
                pr = pickle.load(f)
            test, val, meta, y = pr["test"], pr["val"], pr["meta"], pr["meta"]["y"]
            val_y = pr["val_y"]
            nfams = sorted([m for m in test if m.endswith("_avg") and test[m] is not None])
            eq_te = np.mean([test[m] for m in nfams], axis=0)
            eq_va = np.mean([val[m] for m in nfams], axis=0)
            pers_rmse = E.rmse(test["persistence"] - y)
            best_n = min(nfams, key=lambda m: E.rmse(val[m] - val_y))
            stack_rmse = E.rmse(test["stack"] - y)
            eq_rmse = E.rmse(eq_te - y)
            bn_rmse = E.rmse(test[best_n] - y)
            comp_rows.append(dict(
                run_id=rid, qc_version=tag, N=len(y),
                persist_RMSE=round(pers_rmse, 3),
                best_val_neural=best_n,
                best_val_neural_RMSE=round(bn_rmse, 3),
                best_val_neural_skill_pct=round(100.0 * (1.0 - bn_rmse / pers_rmse), 2),
                equal_neural_ens_RMSE=round(eq_rmse, 3),
                equal_neural_ens_skill_pct=round(100.0 * (1.0 - eq_rmse / pers_rmse), 2),
                stack_RMSE=round(stack_rmse, 3),
                stack_skill_pct=round(100.0 * (1.0 - stack_rmse / pers_rmse), 2),
            ))
            comps = dict(stack=test["stack"], neural_eq=eq_te, best_n=test[best_n])
            for c_row in E.paired_block_bootstrap(y, comps, meta["block"], [("stack", "neural_eq"), ("stack", "best_n")]):
                ci_rows.append(dict(run_id=rid, qc_version=tag,
                                    A=c_row["A"],
                                    B=("equal_neural_ens" if c_row["B"] == "neural_eq" else best_n),
                                    n_blocks=c_row["n_blocks"],
                                    dRMSE_mean=round(c_row["dRMSE_mean"], 3),
                                    dRMSE_lo=round(c_row["dRMSE_lo"], 3),
                                    dRMSE_hi=round(c_row["dRMSE_hi"], 3),
                                    dMAE_mean=round(c_row["dMAE_mean"], 3),
                                    dMAE_lo=round(c_row["dMAE_lo"], 3),
                                    dMAE_hi=round(c_row["dMAE_hi"], 3)))
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(os.path.join(ROOT, v2_dir, "QC1_VS_QC2_SUMMARY.csv"), index=False)
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(os.path.join(ROOT, v2_dir, "STACK_VS_EQ_ENS_CIS.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_qc2")
    ap.add_argument("--core-only", action="store_true", help="Only run the 3 direct_none 5-family benchmarks")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    clean = build_clean_v2(os.path.join(ROOT, "data"), out_dir)
    weather_caches = {
        "none": None,
        "openmeteo": pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet")),
    }
    queue = RUN_SPECS[:3] if args.core_only else RUN_SPECS
    t0 = time.time()
    for spec, nmodels in queue:
        execute_spec(spec, clean, weather_caches, config.SEEDS, config.EPOCHS, out_dir, neural_models=nmodels)
        summarize_qc1_vs_qc2("results_sweep", args.out)
    print(f"\nQC v2 RETRAINING COMPLETE in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

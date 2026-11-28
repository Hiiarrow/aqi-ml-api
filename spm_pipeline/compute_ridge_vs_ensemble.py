"""Compute supplementary Ridge vs equal_neural_ens DATE-block CIs (Table 6) from saved preds.pkl.

Reuses existing results_final/runs/<run_id>/preds.pkl only; never retrains or alters
frozen settings, models, QC, or earlier results.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import compute_r2 as CR2  # noqa: E402
import evaluate as E  # noqa: E402
from final_report import OUT_DIR  # noqa: E402

DELHI_H24_PAIRS = [
    ("ridge", "equal_neural_ens"),
    ("ridge", "persistence"),
    ("ridge", "best_tree"),
]
ENS_ONLY_PAIR = [("ridge", "equal_neural_ens")]

TABLE6_RUN_SPECS = [
    ("P1_LOCO_Delhi_h24_direct_none", DELHI_H24_PAIRS),
    ("P1_LOCO_Delhi_h24_direct_openmeteo", DELHI_H24_PAIRS),
    ("P2_2023_ALL20_h6_direct_none", ENS_ONLY_PAIR),
    ("P2_2023_ALL20_h6_direct_openmeteo", ENS_ONLY_PAIR),
    ("P2_2023_ALL20_h12_direct_none", ENS_ONLY_PAIR),
    ("P2_2023_ALL20_h12_direct_openmeteo", ENS_ONLY_PAIR),
    ("P2_2023_ALL20_h24_direct_none", ENS_ONLY_PAIR),
    ("P2_2023_ALL20_h24_direct_openmeteo", ENS_ONLY_PAIR),
    ("P2_2024_ALL19_h24_direct_none", ENS_ONLY_PAIR),
    ("P2_2024_ALL19_h24_direct_openmeteo", ENS_ONLY_PAIR),
]


def _model_stats(y, pred, persist_rmse, persist_mae):
    e = pred - y
    r, a = E.rmse(e), E.mae(e)
    return dict(
        RMSE=round(r, 3),
        MAE=round(a, 3),
        skill_RMSE=round(100.0 * (1.0 - r / persist_rmse), 2),
        skill_MAE=round(100.0 * (1.0 - a / persist_mae), 2),
        R2=round(CR2.calc_r2(y, pred), 4),
    )


def compute_table6_ridge_vs_ens(out_dir=None):
    """Build TABLE6_RIDGE_VS_ENSEMBLE.csv from saved predictions only."""
    out_dir = out_dir or OUT_DIR
    runs_dir = os.path.join(out_dir, "runs")

    missing = [
        rid for rid, _ in TABLE6_RUN_SPECS
        if not os.path.exists(os.path.join(runs_dir, rid, "preds.pkl"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Saved predictions missing for {len(missing)} run(s); refusing to retrain: {missing}"
        )

    rows = []
    for rid, pairs in TABLE6_RUN_SPECS:
        rdir = os.path.join(runs_dir, rid)
        with open(os.path.join(rdir, "preds.pkl"), "rb") as f:
            preds_bundle = pickle.load(f)
        with open(os.path.join(rdir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        y = preds_bundle["meta"]["y"].astype(np.float64)
        blocks = preds_bundle["meta"]["block_date"]
        needed_models = ("persistence", "ridge", "best_tree", "equal_neural_ens")
        tp = {k: preds_bundle["test"][k].astype(np.float64) for k in needed_models}

        p_rmse = E.rmse(tp["persistence"] - y)
        p_mae = E.mae(tp["persistence"] - y)
        ridge_st = _model_stats(y, tp["ridge"], p_rmse, p_mae)
        ens_st = _model_stats(y, tp["equal_neural_ens"], p_rmse, p_mae)

        boot_rows = E.paired_block_bootstrap(y, tp, blocks, pairs)
        for b in boot_rows:
            rows.append(dict(
                run_id=rid,
                protocol=meta["protocol"],
                heldout=meta["heldout"],
                horizon=int(meta["horizon"]),
                weather_mode=meta["weather_mode"],
                N_test=int(len(y)),
                n_date_blocks=int(b["n_blocks"]),
                ridge_RMSE=ridge_st["RMSE"],
                ridge_MAE=ridge_st["MAE"],
                ridge_skill_RMSE=ridge_st["skill_RMSE"],
                ridge_skill_MAE=ridge_st["skill_MAE"],
                ridge_R2=ridge_st["R2"],
                ens_RMSE=ens_st["RMSE"],
                ens_MAE=ens_st["MAE"],
                ens_skill_RMSE=ens_st["skill_RMSE"],
                ens_skill_MAE=ens_st["skill_MAE"],
                ens_R2=ens_st["R2"],
                A=b["A"],
                B=b["B"],
                dRMSE_mean=round(float(b["dRMSE_mean"]), 3),
                dRMSE_lo=round(float(b["dRMSE_lo"]), 3),
                dRMSE_hi=round(float(b["dRMSE_hi"]), 3),
                dMAE_mean=round(float(b["dMAE_mean"]), 3),
                dMAE_lo=round(float(b["dMAE_lo"]), 3),
                dMAE_hi=round(float(b["dMAE_hi"]), 3),
            ))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "TABLE6_RIDGE_VS_ENSEMBLE.csv")
    df.to_csv(out_csv, index=False)
    return df, out_csv


def main():
    df, out_csv = compute_table6_ridge_vs_ens()
    print(f"Saved {out_csv} ({len(df)} rows)")
    print(df.to_csv(index=False).rstrip())


if __name__ == "__main__":
    main()

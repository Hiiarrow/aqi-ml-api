"""Compute supplementary R^2 metrics (Table 5) from saved predictions in results_final/.

Reuses existing results_final/runs/<run_id>/preds.pkl only; never retrains or alters
frozen settings, models, QC, or earlier result tables.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from final_report import OUT_DIR, RUNS_ORDER  # noqa: E402

TABLE5_MODELS = (
    "persistence",
    "same_hour_yesterday",
    "ridge",
    "best_tree",
    "equal_neural_ens",
    "stack",
)

SUPPLEMENTARY_NOTE = (
    "Note: R^2 is reported as a supplementary metric only; "
    "RMSE, MAE, and skill vs persistence remain the primary evaluation metrics."
)


def calc_r2(y, pred):
    """Compute test-set R^2 = 1 - SSE / SST using the pooled test target mean."""
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(pred, dtype=np.float64)
    yc = y64 - np.mean(y64)
    sst = float(np.sum(yc * yc))
    if sst <= 0.0:
        return float("nan")
    e = p64 - y64
    sse = float(np.sum(e * e))
    return float(1.0 - sse / sst)


def paired_block_bootstrap_r2(y, pred_a, pred_b, blocks, B=None, seed=None):
    """Paired block-bootstrap mean and 95% CI for R^2(pred_a) - R^2(pred_b).

    Uses the exact calendar DATE blocks and multinomial RNG draw as evaluate.paired_block_bootstrap,
    computing pooled SST on each resampled block union via shift-invariant centered sums.
    """
    B = B or config.BOOT_RESAMPLES
    rng = np.random.default_rng(config.BOOT_SEED if seed is None else seed)
    y64 = np.asarray(y, dtype=np.float64)
    pa = np.asarray(pred_a, dtype=np.float64)
    pb = np.asarray(pred_b, dtype=np.float64)

    uniq, inv = np.unique(blocks, return_inverse=True)
    nb = len(uniq)
    cnt = np.bincount(inv, minlength=nb).astype(np.float64)

    yc = y64 - np.mean(y64)
    sum_zc = np.bincount(inv, weights=yc, minlength=nb)
    sum_zc2 = np.bincount(inv, weights=yc * yc, minlength=nb)
    sse_a = np.bincount(inv, weights=(pa - y64) ** 2, minlength=nb)
    sse_b = np.bincount(inv, weights=(pb - y64) ** 2, minlength=nb)

    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=B).astype(np.float64)
    denom = W @ cnt
    sst_boot = (W @ sum_zc2) - ((W @ sum_zc) ** 2) / denom
    dr2 = (W @ (sse_b - sse_a)) / sst_boot

    return dict(
        n_blocks=nb,
        B_resamples=B,
        dR2_boot_mean=float(dr2.mean()),
        dR2_boot_lo=float(np.percentile(dr2, 2.5)),
        dR2_boot_hi=float(np.percentile(dr2, 97.5)),
    )


def compute_table5_r2(out_dir=None):
    """Load saved predictions for all 24 runs, compute TABLE5_R2.csv, and return it."""
    out_dir = out_dir or OUT_DIR
    runs_dir = os.path.join(out_dir, "runs")

    # Verify all saved predictions exist upfront; never retrain.
    missing = [
        rid for rid in RUNS_ORDER
        if not os.path.exists(os.path.join(runs_dir, rid, "preds.pkl"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Saved predictions missing for {len(missing)} run(s); refusing to retrain: {missing}"
        )

    rows = []
    for rid in RUNS_ORDER:
        rdir = os.path.join(runs_dir, rid)
        with open(os.path.join(rdir, "preds.pkl"), "rb") as f:
            preds_bundle = pickle.load(f)
        with open(os.path.join(rdir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        y = preds_bundle["meta"]["y"].astype(np.float64)
        blocks = preds_bundle["meta"]["block_date"]
        test_preds = preds_bundle["test"]

        r2_vals = {
            m: (calc_r2(y, test_preds[m]) if m in test_preds else float("nan"))
            for m in TABLE5_MODELS
        }
        boot = paired_block_bootstrap_r2(
            y,
            test_preds["equal_neural_ens"],
            test_preds["persistence"],
            blocks,
        )

        row = dict(
            run_id=rid,
            protocol=meta["protocol"],
            heldout=meta["heldout"],
            horizon=int(meta["horizon"]),
            weather_mode=meta["weather_mode"],
            N_test=int(len(y)),
            n_date_blocks=int(boot["n_blocks"]),
        )
        for m in TABLE5_MODELS:
            v = r2_vals[m]
            row[f"R2_{m}"] = round(v, 4) if np.isfinite(v) else np.nan
        row["dR2_ens_minus_persist"] = round(
            r2_vals["equal_neural_ens"] - r2_vals["persistence"], 4
        )
        row["dR2_boot_mean"] = round(boot["dR2_boot_mean"], 4)
        row["dR2_boot_lo"] = round(boot["dR2_boot_lo"], 4)
        row["dR2_boot_hi"] = round(boot["dR2_boot_hi"], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "TABLE5_R2.csv")
    df.to_csv(out_csv, index=False)
    return df, out_csv


def main():
    df, out_csv = compute_table5_r2()
    print(f"Saved {out_csv} ({len(df)} runs)")
    print(df.to_csv(index=False).rstrip())
    print(SUPPLEMENTARY_NOTE)


if __name__ == "__main__":
    main()

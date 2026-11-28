"""Extension runner: P1_LOCO at horizon h=24 for Mumbai, Kolkata, Chennai, Bengaluru.

Reuses frozen settings, models, QC v3 (clean_20_2017_2023_v3.pkl), splits, and seeds.
Reads clean_20_2017_2023_v3.pkl and Delhi h=24 preds.pkl strictly read-only from results_final/.
Writes all extension outputs exclusively to results_final_ext/.
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

import compute_r2 as CR2  # noqa: E402
import evaluate as E  # noqa: E402
import final_study as FS  # noqa: E402

EXT_CITIES = ("Mumbai", "Kolkata", "Chennai", "Bengaluru")
ALL_H24_CITIES = ("Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru")
WITHIN_PAIRS = [
    ("equal_neural_ens", "persistence"),
    ("ridge", "persistence"),
    ("ridge", "equal_neural_ens"),
    ("equal_neural_ens", "best_tree"),
]
SUMMARY_MODELS = ("ridge", "best_tree", "equal_neural_ens", "stack")


def build_extension_specs():
    """Return the 8 P1_LOCO h=24 specs for Mumbai, Kolkata, Chennai, Bengaluru."""
    specs = []
    for city in EXT_CITIES:
        for wm in ("none", "openmeteo"):
            specs.append(dict(
                phase=3,
                run_id=f"P1_LOCO_{city}_h24_direct_{wm}",
                protocol="P1_LOCO",
                heldout=city,
                horizon=24,
                target_mode="direct",
                weather_mode=wm,
            ))
    return specs


def _ci_sign(lo, hi):
    if hi < 0.0:
        return "entirely_below_zero"
    if lo > 0.0:
        return "entirely_above_zero"
    return "includes_zero"


def _load_run(runs_dir, rid):
    rdir = os.path.join(runs_dir, rid)
    with open(os.path.join(rdir, "preds.pkl"), "rb") as f:
        preds_bundle = pickle.load(f)
    with open(os.path.join(rdir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return preds_bundle, meta


def _summarize_run(rid, runs_dir, source_tag):
    preds_bundle, meta = _load_run(runs_dir, rid)
    y = preds_bundle["meta"]["y"].astype(np.float64)
    blocks = preds_bundle["meta"]["block_date"]
    tp = {k: v.astype(np.float64) for k, v in preds_bundle["test"].items() if v is not None}

    p = tp["persistence"]
    shy = tp["same_hour_yesterday"]
    shy_diff = float(np.max(np.abs(shy - p)))
    p_rmse = E.rmse(p - y)
    p_mae = E.mae(p - y)
    p_r2 = CR2.calc_r2(y, p)
    shy_rmse = E.rmse(shy - y)

    boot_rows = E.paired_block_bootstrap(y, tp, blocks, WITHIN_PAIRS)
    n_blk = int(boot_rows[0]["n_blocks"])
    assert n_blk == int(meta["n_date_blocks"]), f"block mismatch in {rid}"

    row = dict(
        source=source_tag,
        run_id=rid,
        city=meta["heldout"],
        horizon=int(meta["horizon"]),
        weather_mode=meta["weather_mode"],
        N_test=int(len(y)),
        n_date_blocks=n_blk,
        persist_RMSE=round(p_rmse, 3),
        persist_MAE=round(p_mae, 3),
        persist_R2=round(p_r2, 4),
        same_hr_yest_RMSE=round(shy_rmse, 3),
        same_hr_yest_eq_persist=(shy_diff == 0.0),
        same_hr_yest_max_diff=shy_diff,
        best_tree_choice=meta["best_tree_choice"],
    )
    for m in SUMMARY_MODELS:
        pred = tp[m]
        r_m = E.rmse(pred - y)
        a_m = E.mae(pred - y)
        row[f"{m}_RMSE"] = round(r_m, 3)
        row[f"{m}_MAE"] = round(a_m, 3)
        row[f"{m}_skill_RMSE"] = round(100.0 * (1.0 - r_m / p_rmse), 2)
        row[f"{m}_skill_MAE"] = round(100.0 * (1.0 - a_m / p_mae), 2)
        row[f"{m}_R2"] = round(CR2.calc_r2(y, pred), 4)

    ci_rows = []
    for b in boot_rows:
        pair_tag = f"{b['A']}_minus_{b['B']}"
        d_rmse_lo = float(b["dRMSE_lo"])
        d_rmse_hi = float(b["dRMSE_hi"])
        d_mae_lo = float(b["dMAE_lo"])
        d_mae_hi = float(b["dMAE_hi"])
        ci_rows.append(dict(
            source=source_tag,
            run_id=rid,
            city=meta["heldout"],
            weather_mode=meta["weather_mode"],
            A=b["A"],
            B=b["B"],
            n_blocks=n_blk,
            B_resamples=int(b["B_resamples"]),
            dRMSE_mean=round(float(b["dRMSE_mean"]), 3),
            dRMSE_lo=round(d_rmse_lo, 3),
            dRMSE_hi=round(d_rmse_hi, 3),
            dRMSE_ci_status=_ci_sign(d_rmse_lo, d_rmse_hi),
            dMAE_mean=round(float(b["dMAE_mean"]), 3),
            dMAE_lo=round(d_mae_lo, 3),
            dMAE_hi=round(d_mae_hi, 3),
            dMAE_ci_status=_ci_sign(d_mae_lo, d_mae_hi),
        ))
        if pair_tag in ("equal_neural_ens_minus_persistence", "ridge_minus_equal_neural_ens"):
            pfx = "ens_vs_pers" if b["A"] == "equal_neural_ens" else "ridge_vs_ens"
            row[f"{pfx}_dRMSE_mean"] = round(float(b["dRMSE_mean"]), 3)
            row[f"{pfx}_dRMSE_lo"] = round(d_rmse_lo, 3)
            row[f"{pfx}_dRMSE_hi"] = round(d_rmse_hi, 3)
            row[f"{pfx}_dRMSE_status"] = _ci_sign(d_rmse_lo, d_rmse_hi)
            row[f"{pfx}_dMAE_mean"] = round(float(b["dMAE_mean"]), 3)
            row[f"{pfx}_dMAE_lo"] = round(d_mae_lo, 3)
            row[f"{pfx}_dMAE_hi"] = round(d_mae_hi, 3)
            row[f"{pfx}_dMAE_status"] = _ci_sign(d_mae_lo, d_mae_hi)

    return row, ci_rows, preds_bundle


def compile_extension_outputs(ext_dir, main_dir):
    """Compile extension tables & combined 5-city h=24 table inside results_final_ext/."""
    ext_runs = os.path.join(ext_dir, "runs")
    main_runs = os.path.join(main_dir, "runs")

    ext_summary_rows, ext_ci_rows, ext_om_rows = [], [], []
    all_summary_rows, all_ci_rows = [], []
    by_city_preds = {}

    for city in ALL_H24_CITIES:
        source_tag = "main_results_final" if city == "Delhi" else "extension_post_main"
        runs_dir = main_runs if city == "Delhi" else ext_runs
        bundles = {}
        for wm in ("none", "openmeteo"):
            rid = f"P1_LOCO_{city}_h24_direct_{wm}"
            row, ci_rows, bundle = _summarize_run(rid, runs_dir, source_tag)
            all_summary_rows.append(row)
            all_ci_rows.extend(ci_rows)
            bundles[wm] = bundle
            if city != "Delhi":
                ext_summary_rows.append(row)
                ext_ci_rows.extend(ci_rows)

        # Open-Meteo vs None DATE-block bootstrap CI
        mn, mo = bundles["none"]["meta"], bundles["openmeteo"]["meta"]
        assert np.array_equal(mn["y"], mo["y"]), f"y mismatch for {city}"
        assert np.array_equal(mn["station"], mo["station"]), f"station mismatch for {city}"
        assert np.array_equal(mn["times"], mo["times"]), f"times mismatch for {city}"
        assert np.array_equal(mn["block_date"], mo["block_date"]), f"block_date mismatch for {city}"
        comp = {
            "openmeteo_ens": bundles["openmeteo"]["test"]["equal_neural_ens"].astype(np.float64),
            "none_ens": bundles["none"]["test"]["equal_neural_ens"].astype(np.float64),
        }
        brow = E.paired_block_bootstrap(mn["y"].astype(np.float64), comp, mn["block_date"], [("openmeteo_ens", "none_ens")])[0]
        om_row = dict(
            source=source_tag,
            city=city,
            horizon=24,
            N_test=int(len(mn["y"])),
            n_blocks=int(brow["n_blocks"]),
            B_resamples=int(brow["B_resamples"]),
            dRMSE_mean=round(float(brow["dRMSE_mean"]), 3),
            dRMSE_lo=round(float(brow["dRMSE_lo"]), 3),
            dRMSE_hi=round(float(brow["dRMSE_hi"]), 3),
            dMAE_mean=round(float(brow["dMAE_mean"]), 3),
            dMAE_lo=round(float(brow["dMAE_lo"]), 3),
            dMAE_hi=round(float(brow["dMAE_hi"]), 3),
        )
        by_city_preds[city] = om_row
        if city != "Delhi":
            ext_om_rows.append(om_row)

    df_ext_sum = pd.DataFrame(ext_summary_rows)
    df_ext_ci = pd.DataFrame(ext_ci_rows)
    df_ext_om = pd.DataFrame(ext_om_rows)
    df_comb = pd.DataFrame(all_summary_rows)
    df_comb_ci = pd.DataFrame(all_ci_rows)
    df_comb_om = pd.DataFrame([by_city_preds[c] for c in ALL_H24_CITIES])

    df_ext_sum.to_csv(os.path.join(ext_dir, "EXT_LOCO_H24_SUMMARY.csv"), index=False)
    df_ext_ci.to_csv(os.path.join(ext_dir, "EXT_LOCO_H24_WITHIN_CIS.csv"), index=False)
    df_ext_om.to_csv(os.path.join(ext_dir, "EXT_LOCO_H24_OM_VS_NONE_CIS.csv"), index=False)
    df_comb.to_csv(os.path.join(ext_dir, "COMBINED_5CITY_LOCO_H24.csv"), index=False)
    df_comb_ci.to_csv(os.path.join(ext_dir, "COMBINED_5CITY_LOCO_H24_WITHIN_CIS.csv"), index=False)
    df_comb_om.to_csv(os.path.join(ext_dir, "COMBINED_5CITY_LOCO_H24_OM_VS_NONE_CIS.csv"), index=False)

    print("=== EXTENSION (4 CITIES x 2 WEATHER = 8 RUNS): SUMMARY ===")
    print(df_ext_sum.to_csv(index=False).rstrip())
    print("\n=== EXTENSION: WITHIN-RUN DATE-BLOCK 95% CIs (1000 resamples, BOOT_SEED=0) ===")
    print(df_ext_ci.to_csv(index=False).rstrip())
    print("\n=== EXTENSION: OPEN-METEO VS NONE DATE-BLOCK 95% CIs ===")
    print(df_ext_om.to_csv(index=False).rstrip())
    print("\n=== COMBINED 5-CITY LOCO h=24 (extension, run after the main results were seen) ===")
    print(df_comb.to_csv(index=False).rstrip())
    print("\n=== COMBINED 5-CITY OPEN-METEO VS NONE AT h=24 ===")
    print(df_comb_om.to_csv(index=False).rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-out", default="results_final_ext")
    ap.add_argument("--main-dir", default="results_final")
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    ext_dir = os.path.join(ROOT, args.ext_out)
    main_dir = os.path.join(ROOT, args.main_dir)
    os.makedirs(ext_dir, exist_ok=True)

    if not args.compile_only:
        with open(os.path.join(main_dir, "clean_20_2017_2023_v3.pkl"), "rb") as f:
            clean_23 = pickle.load(f)
        om_23 = pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet"))

        specs = build_extension_specs()
        t0 = time.time()
        for sp in specs:
            FS.execute_run(sp, clean_23, None, om_23, None, ext_dir)
        print(f"\nAll {len(specs)} extension runs finished in {(time.time() - t0)/60:.1f} min -> {ext_dir}", flush=True)

    compile_extension_outputs(ext_dir, main_dir)


if __name__ == "__main__":
    main()

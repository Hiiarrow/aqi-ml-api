"""
Generate complete verified tables & markdown report for the FINAL 20-station QC v3 study
from results_final/ outputs ONLY. Never alters existing results folders.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results_final")

RUNS_ORDER = [
    "P2_2023_ALL20_h1_direct_none", "P2_2023_ALL20_h1_direct_openmeteo",
    "P2_2023_ALL20_h6_direct_none", "P2_2023_ALL20_h6_direct_openmeteo",
    "P2_2023_ALL20_h12_direct_none", "P2_2023_ALL20_h12_direct_openmeteo",
    "P2_2023_ALL20_h24_direct_none", "P2_2023_ALL20_h24_direct_openmeteo",
    "P2_2024_ALL19_h1_direct_none", "P2_2024_ALL19_h1_direct_openmeteo",
    "P2_2024_ALL19_h24_direct_none", "P2_2024_ALL19_h24_direct_openmeteo",
    "P1_LOCO_Delhi_h1_direct_none", "P1_LOCO_Delhi_h1_direct_openmeteo",
    "P1_LOCO_Delhi_h24_direct_none", "P1_LOCO_Delhi_h24_direct_openmeteo",
    "P1_LOCO_Mumbai_h1_direct_none", "P1_LOCO_Mumbai_h1_direct_openmeteo",
    "P1_LOCO_Kolkata_h1_direct_none", "P1_LOCO_Kolkata_h1_direct_openmeteo",
    "P1_LOCO_Chennai_h1_direct_none", "P1_LOCO_Chennai_h1_direct_openmeteo",
    "P1_LOCO_Bengaluru_h1_direct_none", "P1_LOCO_Bengaluru_h1_direct_openmeteo",
]


def main():
    fs = pd.read_csv(os.path.join(OUT_DIR, "FINAL_SUMMARY.csv")).set_index("run_id").loc[RUNS_ORDER].reset_index()
    lb = pd.read_csv(os.path.join(OUT_DIR, "LEADERBOARD.csv"))
    sd = pd.read_csv(os.path.join(OUT_DIR, "SEED_STATS.csv"))
    bw = pd.read_csv(os.path.join(OUT_DIR, "ALL_BOOTSTRAP_WITHIN.csv"))
    om = pd.read_csv(os.path.join(OUT_DIR, "OPENMETEO_VS_NONE_CIS.csv"))
    rg = pd.read_csv(os.path.join(OUT_DIR, "ALL_REGIMES.csv"))

    # Table 1: Baselines & Models (N, Date blocks, Persistence, Baselines, Trees, Equal Ens, Stack)
    t1_rows = []
    for rid in RUNS_ORDER:
        sub = lb[lb.run_id == rid].set_index("model")
        mr = fs[fs.run_id == rid].iloc[0]
        d = dict(
            run_id=rid,
            N_test=int(mr.N_test),
            date_blks=int(mr.n_date_blocks),
            persist_RMSE=round(float(mr.persist_RMSE), 3),
            persist_MAE=round(float(mr.persist_MAE), 3),
        )
        for mod in ["same_hour_yesterday", "climatology", "ridge", "gbm_direct", "xgb_direct", "gbm_residual", "xgb_residual"]:
            if mod in sub.index:
                r = sub.loc[mod]
                d[f"{mod}_RMSE"] = round(float(r.test_RMSE), 3)
                d[f"{mod}_skill"] = round(float(r.skill_RMSE_pct), 2)
            else:
                d[f"{mod}_RMSE"] = np.nan
                d[f"{mod}_skill"] = np.nan
        d["best_tree_choice"] = mr.best_tree_choice
        for mod in ["best_tree", "equal_neural_ens", "stack"]:
            r = sub.loc[mod]
            d[f"{mod}_RMSE"] = round(float(r.test_RMSE), 3)
            d[f"{mod}_MAE"] = round(float(r.test_MAE), 3)
            d[f"{mod}_skill_RMSE"] = round(float(r.skill_RMSE_pct), 2)
            d[f"{mod}_skill_MAE"] = round(float(r.skill_MAE_pct), 2)
        t1_rows.append(d)
    df_t1 = pd.DataFrame(t1_rows)
    df_t1.to_csv(os.path.join(OUT_DIR, "TABLE1_BASELINES_AND_MODELS.csv"), index=False)

    # Table 2: Neural family seed mean +- std (RMSE & MAE) + 5-seed ensemble RMSE & skill
    t2_rows = []
    for rid in RUNS_ORDER:
        sub_sd = sd[sd.run_id == rid].set_index("model")
        sub_lb = lb[lb.run_id == rid].set_index("model")
        for fam in ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]:
            rs = sub_sd.loc[fam]
            re = sub_lb.loc[fam + "_avg"]
            t2_rows.append(dict(
                run_id=rid,
                family=fam,
                n_seeds=int(rs.n_seeds),
                seed_RMSE_mean=round(float(rs.RMSE_mean), 3),
                seed_RMSE_std=round(float(rs.RMSE_std), 3),
                seed_MAE_mean=round(float(rs.MAE_mean), 3),
                seed_MAE_std=round(float(rs.MAE_std), 3),
                ens5_RMSE=round(float(re.test_RMSE), 3),
                ens5_skill_RMSE=round(float(re.skill_RMSE_pct), 2),
                ens5_MAE=round(float(re.test_MAE), 3),
                ens5_skill_MAE=round(float(re.skill_MAE_pct), 2),
            ))
    df_t2 = pd.DataFrame(t2_rows)
    df_t2.to_csv(os.path.join(OUT_DIR, "TABLE2_NEURAL_SEEDS.csv"), index=False)

    # Compact wide version of Table 2 for direct printing
    wide_rows = []
    for rid in RUNS_ORDER:
        sub = df_t2[df_t2.run_id == rid].set_index("family")
        sub_lb = lb[lb.run_id == rid].set_index("model")
        w = dict(run_id=rid)
        for fam in ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]:
            r = sub.loc[fam]
            w[fam] = f"{r.seed_RMSE_mean:.3f}+-{r.seed_RMSE_std:.3f} [{r.ens5_skill_RMSE:+.2f}%]"
        eq = sub_lb.loc["equal_neural_ens"]
        w["EQ_ENS"] = f"{eq.test_RMSE:.3f} [{eq.skill_RMSE_pct:+.2f}%]"
        wide_rows.append(w)
    df_t2_wide = pd.DataFrame(wide_rows)

    # Table 3A: Within-run DATE-block CIs (rounded 3 decimals)
    df_bw = bw.copy()
    for c in ["dRMSE_mean", "dRMSE_lo", "dRMSE_hi", "dMAE_mean", "dMAE_lo", "dMAE_hi"]:
        df_bw[c] = df_bw[c].round(3)
    df_bw.to_csv(os.path.join(OUT_DIR, "TABLE3A_WITHIN_CIS_DATEBLOCK.csv"), index=False)

    # Table 3B: OpenMeteo vs None DATE-block CIs (all 12 pairs)
    df_om = om.copy()
    for c in ["dRMSE_mean", "dRMSE_lo", "dRMSE_hi", "dMAE_mean", "dMAE_lo", "dMAE_hi"]:
        df_om[c] = df_om[c].round(3)
    df_om.to_csv(os.path.join(OUT_DIR, "TABLE3B_OPENMETEO_VS_NONE_CIS.csv"), index=False)

    # Print Sections 1-3 cleanly to stdout
    print("=== TABLE 1: ALL 24 RUNS — N_test, DATE BLOCKS, PERSISTENCE & ALL BASELINES/HEADLINES (RMSE & SKILL_RMSE%) ===")
    c_rmse = [
        "run_id", "N_test", "date_blks", "persist_RMSE",
        "same_hour_yesterday_skill", "climatology_skill", "ridge_skill",
        "gbm_direct_skill", "xgb_direct_skill", "gbm_residual_skill", "xgb_residual_skill",
        "best_tree_choice", "best_tree_RMSE", "best_tree_skill_RMSE",
        "equal_neural_ens_RMSE", "equal_neural_ens_skill_RMSE",
        "stack_RMSE", "stack_skill_RMSE",
    ]
    print(df_t1[c_rmse].to_string(index=False))

    print("\n=== TABLE 1B: MAE & SKILL_MAE% ACROSS KEY BASELINES & HEADLINES ===")
    rows_m = []
    for rid in RUNS_ORDER:
        sub = lb[lb.run_id == rid].set_index("model")
        mr = fs[fs.run_id == rid].iloc[0]
        row = dict(run_id=rid, persist_MAE=round(float(mr.persist_MAE), 3))
        for mod in ["same_hour_yesterday", "climatology", "ridge", "best_tree", "equal_neural_ens", "stack"]:
            if mod in sub.index:
                r = sub.loc[mod]
                row[f"{mod}_MAE"] = round(float(r.test_MAE), 3)
                row[f"{mod}_sMAE%"] = round(float(r.skill_MAE_pct), 2)
            else:
                row[f"{mod}_MAE"] = np.nan
                row[f"{mod}_sMAE%"] = np.nan
        rows_m.append(row)
    print(pd.DataFrame(rows_m).to_string(index=False))

    print("\n=== TABLE 2: PER-FAMILY 5-SEED MEAN +- STD RMSE [5-SEED FAMILY ENSEMBLE SKILL%] & EQUAL_NEURAL_ENS ===")
    print(df_t2_wide.to_string(index=False))

    print("\n=== TABLE 3A: WITHIN-RUN DATE-BLOCK 95% CIs (A minus B; 1000 resamples; NO p-values) ===")
    print(df_bw[["run_id", "A", "B", "n_blocks", "dRMSE_mean", "dRMSE_lo", "dRMSE_hi", "dMAE_mean", "dMAE_lo", "dMAE_hi"]].to_string(index=False))

    print("\n=== TABLE 3B: OPENMETEO vs NONE DATE-BLOCK 95% CIs (equal_neural_ens[openmeteo] - equal_neural_ens[none]; 1000 resamples) ===")
    print(df_om[["protocol", "heldout", "horizon", "N_test", "n_blocks", "dRMSE_mean", "dRMSE_lo", "dRMSE_hi", "dMAE_mean", "dMAE_lo", "dMAE_hi"]].to_string(index=False))

    # Build formatted Regime summary markdown & CSV
    reg_keep = ["persistence", "same_hour_yesterday", "climatology", "ridge", "best_tree", "equal_neural_ens", "stack"]
    reg_sub = rg[rg["model"].isin(reg_keep)].copy()
    for c in ["RMSE", "MAE", "skill_RMSE_pct", "skill_MAE_pct"]:
        reg_sub[c] = reg_sub[c].round(2)
    reg_order = [
        "y_t >= 60", "y_t >= 120", "y_t >= 250",
        "onset (y_t - y_t-3 > 30)", "winter (Nov-Feb)", "other months (Mar-Oct)", "all",
    ]
    reg_sub["regime"] = pd.Categorical(reg_sub["regime"], categories=reg_order, ordered=True)
    reg_sub["run_id"] = pd.Categorical(reg_sub["run_id"], categories=RUNS_ORDER, ordered=True)
    reg_sub = reg_sub.sort_values(["run_id", "regime", "model"])
    reg_sub.to_csv(os.path.join(OUT_DIR, "TABLE4_REGIMES_CLEAN.csv"), index=False)

    # Save complete FINAL_REPORT.md
    md = [
        "# Final Study Report (20 Study Stations, Frozen QC v3, DATE-Block CIs)",
        "",
        "Primary method (fixed in advance): `equal_neural_ens` (5-family equal average of `gru, lstm, bilstm, tcn, cnn_lstm`, 5 seeds `0..4`), `direct` target, `none` weather.",
        "",
        "## 1. Run Summary, Persistence & All Baselines",
        "",
        "```text",
        df_t1[c_rmse].to_string(index=False),
        "```",
        "",
        "## 1B. MAE & Skill MAE%",
        "",
        "```text",
        pd.DataFrame(rows_m).to_string(index=False),
        "```",
        "",
        "## 2. Per-Family Seed Mean ± Std & Ensemble Skill",
        "",
        "```text",
        df_t2_wide.to_string(index=False),
        "```",
        "",
        "## 3A. DATE-Block 95% CIs (Within-Run)",
        "",
        "```text",
        df_bw[["run_id", "A", "B", "n_blocks", "dRMSE_mean", "dRMSE_lo", "dRMSE_hi", "dMAE_mean", "dMAE_lo", "dMAE_hi"]].to_string(index=False),
        "```",
        "",
        "## 3B. DATE-Block 95% CIs (Open-Meteo vs None)",
        "",
        "```text",
        df_om[["protocol", "heldout", "horizon", "N_test", "n_blocks", "dRMSE_mean", "dRMSE_lo", "dRMSE_hi", "dMAE_mean", "dMAE_lo", "dMAE_hi"]].to_string(index=False),
        "```",
        "",
        "## 4. Regime Tables (All 24 Runs)",
        "",
    ]
    for rid, g in reg_sub.groupby("run_id", observed=True):
        md.append(f"### `{rid}`")
        md.append("```text")
        md.append(g[["regime", "model", "N", "RMSE", "MAE", "skill_RMSE_pct", "skill_MAE_pct"]].to_string(index=False))
        md.append("```")
        md.append("")
    rep_p = os.path.join(OUT_DIR, "FINAL_REPORT.md")
    with open(rep_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nSaved complete FINAL_REPORT.md ({os.path.getsize(rep_p):,} bytes) and TABLE1..TABLE4 CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()

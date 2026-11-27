"""
Aggregate all completed runs under results_sweep/runs/ into:
  - results_sweep/LEADERBOARD.csv
  - results_sweep/RUNS_STATUS.csv
  - results_sweep/MORNING_REPORT.md
"""
import glob
import json
import os
from datetime import datetime

import pandas as pd


def _df_to_md(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [f"{v:.3f}" if isinstance(v, float) and pd.notna(v) else ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report(out_dir):
    runs_dir = os.path.join(out_dir, "runs")
    meta_files = sorted(glob.glob(os.path.join(runs_dir, "*", "meta.json")))
    if not meta_files:
        return

    lb_rows = []
    status_rows = []
    boot_rows = []
    reg_rows = []
    seed_rows = []

    for mf in meta_files:
        rdir = os.path.dirname(mf)
        with open(mf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        rid = meta["run_id"]

        res_p = os.path.join(rdir, "results.csv")
        if os.path.exists(res_p):
            df = pd.read_csv(res_p)
            df.insert(0, "weather_mode", meta["weather_mode"])
            df.insert(0, "target_mode", meta["target_mode"])
            df.insert(0, "horizon", meta["horizon"])
            df.insert(0, "heldout", meta["heldout"])
            df.insert(0, "protocol", meta["protocol"])
            df.insert(0, "run_id", rid)
            lb_rows.append(df)

            # Best non-persistence model by val_RMSE (strict validation-only selection) & test_RMSE (audit only)
            cand = df[df["model"] != "persistence"]
            best_test = cand.loc[cand["test_RMSE"].idxmin()]
            best_val = cand.loc[cand["val_RMSE"].idxmin()]
            sing_cand = cand[cand["model"] != "stack"]
            best_val_sing = sing_cand.loc[sing_cand["val_RMSE"].idxmin()] if not sing_cand.empty else best_val
            stack_r = df[df["model"] == "stack"]
            pers_r = df[df["model"] == "persistence"].iloc[0]
            status_rows.append(dict(
                run_id=rid,
                protocol=meta["protocol"],
                heldout=meta["heldout"],
                horizon=meta["horizon"],
                target_mode=meta["target_mode"],
                weather_mode=meta["weather_mode"],
                train_n=meta["train_n"],
                val_n=meta["val_n"],
                test_n=meta["test_n"],
                elapsed_s=round(meta["elapsed_s"], 1),
                pers_RMSE=round(float(pers_r["test_RMSE"]), 3),
                best_val_model=best_val["model"],
                best_val_RMSE=round(float(best_val["val_RMSE"]), 3),
                best_val_test_RMSE=round(float(best_val["test_RMSE"]), 3),
                best_val_test_MAE=round(float(best_val["test_MAE"]), 3),
                best_val_skill_pct=round(float(best_val["skill_RMSE_pct"]), 2),
                val_single_model=best_val_sing["model"],
                val_single_test_RMSE=round(float(best_val_sing["test_RMSE"]), 3),
                val_single_skill_pct=round(float(best_val_sing["skill_RMSE_pct"]), 2),
                stack_RMSE=round(float(stack_r["test_RMSE"].iloc[0]), 3) if not stack_r.empty else None,
                stack_skill_pct=round(float(stack_r["skill_RMSE_pct"].iloc[0]), 2) if not stack_r.empty else None,
                best_test_model=best_test["model"],
                best_test_RMSE=round(float(best_test["test_RMSE"]), 3),
                best_test_MAE=round(float(best_test["test_MAE"]), 3),
                best_skill_RMSE_pct=round(float(best_test["skill_RMSE_pct"]), 2),
            ))

        bp = os.path.join(rdir, "bootstrap.csv")
        if os.path.exists(bp):
            b = pd.read_csv(bp)
            b.insert(0, "run_id", rid)
            boot_rows.append(b)

        rp = os.path.join(rdir, "regimes.csv")
        if os.path.exists(rp):
            rg = pd.read_csv(rp)
            rg.insert(0, "run_id", rid)
            reg_rows.append(rg)

        sp = os.path.join(rdir, "seed_rows.csv")
        if os.path.exists(sp):
            sr = pd.read_csv(sp)
            sr.insert(0, "run_id", rid)
            seed_rows.append(sr)

    lb = pd.concat(lb_rows, ignore_index=True) if lb_rows else pd.DataFrame()
    st = pd.DataFrame(status_rows)
    if not lb.empty:
        lb.to_csv(os.path.join(out_dir, "LEADERBOARD.csv"), index=False)
    if not st.empty:
        st.to_csv(os.path.join(out_dir, "RUNS_STATUS.csv"), index=False)
    if boot_rows:
        pd.concat(boot_rows, ignore_index=True).to_csv(os.path.join(out_dir, "ALL_BOOTSTRAP.csv"), index=False)
    if reg_rows:
        pd.concat(reg_rows, ignore_index=True).to_csv(os.path.join(out_dir, "ALL_REGIMES.csv"), index=False)
    if seed_rows:
        pd.concat(seed_rows, ignore_index=True).to_csv(os.path.join(out_dir, "ALL_SEEDS.csv"), index=False)

    # Generate concise markdown summary (headline selections STRICTLY on validation RMSE)
    lines = [
        f"# Overnight Model Selection & Benchmark Report",
        f"Updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` | Completed runs: **{len(st)}** | Selection rule: **Validation-only (`min val_RMSE`)**",
        "",
        "## 1. Run Summary Table (Validation-Selected Headline Models)",
        "",
    ]
    if not st.empty:
        st_s = st.sort_values(["protocol", "horizon", "heldout", "target_mode", "weather_mode"])
        cols_s1 = [
            "run_id", "protocol", "heldout", "horizon", "target_mode", "weather_mode",
            "train_n", "val_n", "test_n", "elapsed_s", "pers_RMSE",
            "best_val_model", "best_val_RMSE", "best_val_test_RMSE", "best_val_test_MAE", "best_val_skill_pct",
            "val_single_model", "val_single_test_RMSE", "stack_RMSE", "stack_skill_pct",
        ]
        lines.append(_df_to_md(st_s[cols_s1]))
        lines.append("")

        # Validation-only selection across (target_mode, weather_mode, model) per (protocol, heldout, horizon)
        cfg_rows = []
        for (prot, held, hor), grp in st_s.groupby(["protocol", "heldout", "horizon"]):
            bv = grp.loc[grp["best_val_RMSE"].idxmin()]
            gn = grp[grp["weather_mode"] == "none"]
            bvn = gn.loc[gn["best_val_RMSE"].idxmin()] if not gn.empty else bv
            cfg_rows.append(dict(
                protocol=prot, heldout=held, horizon=hor, pers_RMSE=bv["pers_RMSE"],
                val_selected_cfg=f"{bv['target_mode']}/{bv['weather_mode']}",
                val_model=bv["best_val_model"], val_RMSE=bv["best_val_RMSE"],
                test_RMSE=bv["best_val_test_RMSE"], test_MAE=bv["best_val_test_MAE"],
                skill_pct=bv["best_val_skill_pct"],
                val_none_target=bvn["target_mode"], val_none_RMSE=bvn["best_val_test_RMSE"],
                val_none_skill_pct=bvn["best_val_skill_pct"],
            ))
        lines.append("## 1b. Headline Selection Summary (Target, Weather, and Model Chosen on Validation Only)")
        lines.append("")
        lines.append(_df_to_md(pd.DataFrame(cfg_rows)))
        lines.append("")

        # Compare Direct vs Residual head-to-head using validation-selected models
        base = st[st["weather_mode"] == "none"]
        if not base.empty and {"direct", "residual"} <= set(base["target_mode"]):
            lines.append("## 2. Direct vs Residual Summary (`weather_mode = none`, Validation-Selected Models)")
            lines.append("")
            dr_rows = []
            for (prot, held, hor), grp in base.groupby(["protocol", "heldout", "horizon"]):
                if not {"direct", "residual"} <= set(grp["target_mode"]):
                    continue
                d = grp[grp["target_mode"] == "direct"].iloc[0]
                r = grp[grp["target_mode"] == "residual"].iloc[0]
                w_t = "direct" if d["best_val_RMSE"] <= r["best_val_RMSE"] else "residual"
                dr_rows.append(dict(
                    protocol=prot, heldout=held, horizon=hor, pers_RMSE=d["pers_RMSE"],
                    val_model_D=d["best_val_model"], val_RMSE_D=d["best_val_RMSE"],
                    test_RMSE_D=d["best_val_test_RMSE"], skill_pct_D=d["best_val_skill_pct"],
                    val_model_R=r["best_val_model"], val_RMSE_R=r["best_val_RMSE"],
                    test_RMSE_R=r["best_val_test_RMSE"], skill_pct_R=r["best_val_skill_pct"],
                    val_selected_target=w_t,
                ))
            if dr_rows:
                lines.append(_df_to_md(pd.DataFrame(dr_rows)))
                lines.append("")

        # Model family win counts & mean skill across baseline runs
        if not lb.empty:
            lines.append("## 3. Model Ranking Across Completed Runs (`weather_mode = none`)")
            lines.append("")
            lb_none = lb[(lb["weather_mode"] == "none") & (lb["model"] != "persistence")]
            if not lb_none.empty:
                m_sum = (
                    lb_none.groupby(["target_mode", "model"], as_index=False)
                    .agg(
                        n_runs=("run_id", "count"),
                        mean_skill_RMSE_pct=("skill_RMSE_pct", "mean"),
                        min_skill_RMSE_pct=("skill_RMSE_pct", "min"),
                        max_skill_RMSE_pct=("skill_RMSE_pct", "max"),
                        mean_test_RMSE=("test_RMSE", "mean"),
                        mean_test_MAE=("test_MAE", "mean"),
                    )
                    .sort_values(["target_mode", "mean_skill_RMSE_pct"], ascending=[True, False])
                    .round(3)
                )
                lines.append(_df_to_md(m_sum))
                lines.append("")

                # Common-subset P1_LOCO h=1 & h=24 across all 6 cities (avoids partial-backfill Simpson's paradox)
                for h_sub, label_sub in [(1, "3b. Common-Subset P1_LOCO h=1 (All 6 Cities, `weather_mode = none`)"),
                                         (24, "3c. Common-Subset P1_LOCO h=24 (All 6 Cities, `weather_mode = none`)")]:
                    sub_h = lb_none[(lb_none["protocol"] == "P1_LOCO") & (lb_none["horizon"] == h_sub)]
                    if not sub_h.empty:
                        cs = (
                            sub_h.groupby(["target_mode", "model"], as_index=False)
                            .agg(
                                n_cities=("run_id", "count"),
                                mean_skill_pct=("skill_RMSE_pct", "mean"),
                                min_skill_pct=("skill_RMSE_pct", "min"),
                                max_skill_pct=("skill_RMSE_pct", "max"),
                                mean_RMSE=("test_RMSE", "mean"),
                                mean_MAE=("test_MAE", "mean"),
                            )
                            .sort_values(["target_mode", "mean_skill_pct"], ascending=[True, False])
                            .round(3)
                        )
                        lines.append(f"### {label_sub}")
                        lines.append("")
                        lines.append(_df_to_md(cs))
                        lines.append("")

        # Weather ablation comparison (evaluated using validation-selected models & validation-winning weather)
        w_runs = st[st["weather_mode"] != "none"]
        if not w_runs.empty:
            lines.append("## 4. Weather Ablation (`none` vs `cpcb` vs `openmeteo`, lags <= t, Validation-Selected)")
            lines.append("")
            keys = w_runs[["protocol", "heldout", "horizon", "target_mode"]].drop_duplicates()
            comp = st.merge(keys, on=["protocol", "heldout", "horizon", "target_mode"])
            comp = comp.sort_values(["protocol", "heldout", "horizon", "target_mode", "weather_mode"])
            best_w_idx = comp.groupby(["protocol", "heldout", "horizon", "target_mode"])["best_val_RMSE"].idxmin()
            comp["val_selected_weather"] = comp.index.isin(best_w_idx)
            lines.append(_df_to_md(comp[[
                "run_id", "weather_mode", "pers_RMSE", "best_val_model", "best_val_RMSE",
                "best_val_test_RMSE", "best_val_test_MAE", "best_val_skill_pct", "stack_RMSE", "val_selected_weather",
            ]]))
            lines.append("")

    # Hybrid D+R summary if generated (compute validation-selected best single model from LEADERBOARD.csv)
    hdr_p = os.path.join(out_dir, "HYBRID_DR_SUMMARY.csv")
    if os.path.exists(hdr_p):
        hdf = pd.read_csv(hdr_p)
        if not hdf.empty:
            if not lb.empty:
                val_sing_names, val_sing_rmses = [], []
                for _, r in hdf.iterrows():
                    rd = f"{r['protocol']}_{r['heldout']}_h{int(r['horizon'])}_direct_{r['weather_mode']}"
                    rr = f"{r['protocol']}_{r['heldout']}_h{int(r['horizon'])}_residual_{r['weather_mode']}"
                    sd_ = lb[(lb["run_id"] == rd) & (~lb["model"].isin(["persistence", "stack"]))].copy()
                    sr_ = lb[(lb["run_id"] == rr) & (~lb["model"].isin(["persistence", "stack"]))].copy()
                    sd_["pref"] = "D_" + sd_["model"]
                    sr_["pref"] = "R_" + sr_["model"]
                    comb = pd.concat([sd_, sr_], ignore_index=True)
                    if not comb.empty:
                        bvs = comb.loc[comb["val_RMSE"].idxmin()]
                        val_sing_names.append(bvs["pref"])
                        val_sing_rmses.append(round(float(bvs["test_RMSE"]), 3))
                    else:
                        val_sing_names.append("")
                        val_sing_rmses.append(None)
                hdf = hdf.copy()
                hdf["val_best_single"] = val_sing_names
                hdf["val_best_single_RMSE"] = val_sing_rmses
                keep_cols = ["protocol", "heldout", "horizon", "weather_mode", "pers_RMSE",
                             "val_best_single", "val_best_single_RMSE",
                             "hybrid_DR_RMSE", "hybrid_DR_MAE", "hybrid_skill_pct", "top_weights"]
                hdf = hdf[[c for c in keep_cols if c in hdf.columns]]
            lines.append("## 5. Combined Direct + Residual NNLS Hybrid (`HYBRID_DR_SUMMARY.csv`, Validation-Selected Comparison)")
            lines.append("")
            lines.append(_df_to_md(hdf))
            lines.append("")

    # Spike & seasonal regime skill (ALL_REGIMES.csv)
    reg_p = os.path.join(out_dir, "ALL_REGIMES.csv")
    if os.path.exists(reg_p):
        rga = pd.read_csv(reg_p)
        if not rga.empty:
            lines.append("## 6. Spike & Seasonal Regime Skill (`ALL_REGIMES.csv`)")
            lines.append("")
            p1_d_ids = [r for r in st["run_id"] if r.startswith("P1_LOCO_") and "_h1_direct_none" in r] if not st.empty else []
            if p1_d_ids:
                piv_r = rga[rga["run_id"].isin(p1_d_ids) & (rga["model"] != "persistence")].pivot_table(
                    index="model", columns="regime", values="skill_RMSE_pct", aggfunc="mean"
                ).round(2)
                if not piv_r.empty:
                    lines.append("### 6a. P1_LOCO h=1 Direct (Mean Skill % Across 6 Cities)")
                    lines.append("")
                    lines.append(_df_to_md(piv_r.reset_index()))
                    lines.append("")
            key_reg_runs = [
                "P1_LOCO_Delhi_h1_direct_none",
                "P1_LOCO_Delhi_h1_direct_openmeteo",
                "P1_LOCO_Delhi_h1_residual_none",
                "P1_LOCO_Delhi_h1_residual_openmeteo",
                "P2_2023_ALL_h1_residual_openmeteo",
                "P2_2023_ALL_h6_residual_none",
                "P2_2023_ALL_h12_direct_none",
                "P2_2023_ALL_h24_residual_openmeteo",
            ]
            sub_rg = rga[rga["run_id"].isin(key_reg_runs) & (rga["model"] != "persistence")]
            if not sub_rg.empty:
                piv_k = sub_rg.pivot_table(
                    index=["run_id", "model"], columns="regime", values="skill_RMSE_pct", aggfunc="first"
                ).round(2)
                lines.append("### 6b. Key Benchmarks by Regime (Skill %)")
                lines.append("")
                lines.append(_df_to_md(piv_k.reset_index()))
                lines.append("")

    # Paired day-block 95% bootstrap CIs (no p-values)
    bt_p = os.path.join(out_dir, "ALL_BOOTSTRAP.csv")
    if os.path.exists(bt_p):
        bt_all = pd.read_csv(bt_p)
        if not bt_all.empty:
            lines.append("## 7. Paired Day-Block 95% Bootstrap CIs (`ALL_BOOTSTRAP.csv`, Negative = A Better)")
            lines.append("")
            lines.append(_df_to_md(bt_all[["run_id", "A", "B", "n_blocks", "dRMSE_mean", "dRMSE_lo", "dRMSE_hi",
                                           "dMAE_mean", "dMAE_lo", "dMAE_hi"]].round(3)))
            lines.append("")

    # Checkpoint info
    ckpt_dir = os.path.join(out_dir, "best_checkpoints")
    if os.path.isdir(ckpt_dir):
        man_p = os.path.join(ckpt_dir, "MANIFEST.csv")
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        if os.path.exists(man_p):
            mdf = pd.read_csv(man_p)
            lines.append("## 8. Saved Model Checkpoints (`best_checkpoints/`)")
            lines.append("")
            lines.append(_df_to_md(mdf))
            lines.append("")
        elif ckpts:
            lines.append("## 8. Saved Model Checkpoints (`best_checkpoints/`)")
            lines.append("")
            for c in ckpts:
                lines.append(f"- `{c}`")
            lines.append("")

    with open(os.path.join(out_dir, "MORNING_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

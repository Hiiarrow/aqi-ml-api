"""
Resumable, checkpointed overnight sweep runner for:
  - Stage 0: Fast Statistical & Tree baselines (Linear, Tuned Ridge, LightGBM, CUDA XGBoost) across ALL configs
  - Stage 1: P1 LOCO (6 cities x {h=1, h=24} x {direct, residual})
  - Stage 2: P2 2023 Temporal Holdout ({h=1, 6, 12, 24} x {direct, residual}, h+24 gap)
  - Stage 3: Co-located CPCB & Open-Meteo ERA5 Weather Ablations (lags <= t only)
  - Stage 4: Combined Direct+Residual NNLS Hybrid Evaluation & Best Checkpoint Export
  - Stage 5: Optional backfill of remaining neural architectures within time budget

Writes everything strictly under results_sweep/ (never modifies results/).
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import config
import evaluate as E
from models_sweep import fit_gbm, fit_linear_and_ridge, fit_xgb, nnls_stack, train_neural
from report_gen import build_report
from splits import build_loco_fold, build_p2_2023_fold, extract_features


def _metrics(y, p):
    e = p - y
    return E.rmse(e), E.mae(e)


def evaluate_and_save(spec, fold, Xva_f, Xte_f, val_preds, test_preds, seed_rows,
                      best_states, val_y_raw, te_meta, out_dir, start_t, seed_list):
    rid = spec["run_id"]
    rdir = os.path.join(out_dir, "runs", rid)
    os.makedirs(rdir, exist_ok=True)

    # Fit NNLS stack on validation raw targets ONLY (using non-persistence, non-linear models)
    members = [m for m in val_preds if m not in ("persistence", "linear") and val_preds[m] is not None]
    w, sv, st = nnls_stack({m: val_preds[m] for m in members}, val_y_raw,
                           {m: test_preds[m] for m in members})
    val_all = dict(val_preds, stack=sv)
    test_all = dict(test_preds, stack=st)

    rows = []
    for m, pte in test_all.items():
        if pte is None:
            continue
        r_, a_ = _metrics(te_meta["y"], pte)
        rv, _ = _metrics(val_y_raw, val_all[m])
        rows.append(dict(model=m, val_RMSE=rv, test_RMSE=r_, test_MAE=a_))
    res = pd.DataFrame(rows)
    pr, pa = res.loc[res.model == "persistence", ["test_RMSE", "test_MAE"]].iloc[0]
    res["skill_RMSE_pct"] = 100.0 * (1.0 - res["test_RMSE"] / pr)
    res["skill_MAE_pct"] = 100.0 * (1.0 - res["test_MAE"] / pa)
    res.to_csv(os.path.join(rdir, "results.csv"), index=False)

    if seed_rows:
        pd.DataFrame(seed_rows).to_csv(os.path.join(rdir, "seed_rows.csv"), index=False)

    # Paired day-block bootstrap (95% CIs only, no p-values)
    res_idx = res.set_index("model")
    neural = [m for m in test_all if m.endswith("_avg")]
    trees = [m for m in ("gbm", "xgboost") if m in test_all]
    single = [m for m in test_all if m not in ("persistence", "stack")]
    pick = lambda names: min(names, key=lambda m: res_idx.loc[m, "val_RMSE"])
    pairs = [("stack", "persistence"), (pick(single), "persistence")]
    if trees:
        pairs.insert(0, ("stack", pick(trees)))
    if neural:
        pairs.insert(0, ("stack", pick(neural)))
    boot = pd.DataFrame(E.paired_block_bootstrap(te_meta["y"], test_all, te_meta["block"], pairs))
    boot.to_csv(os.path.join(rdir, "bootstrap.csv"), index=False)

    # Time-t regime table
    reg = pd.DataFrame(E.regime_table(te_meta, te_meta["y"], test_all))
    reg.to_csv(os.path.join(rdir, "regimes.csv"), index=False)

    # Save predictions for fast post-hoc D+R hybridization without retraining
    with open(os.path.join(rdir, "preds.pkl"), "wb") as f:
        pickle.dump(dict(val=val_all, test=test_all, val_y=val_y_raw,
                         meta={k: te_meta[k] for k in ("y", "y_curr", "rise3", "month", "block", "station")},
                         stack_weights={k: float(v) for k, v in w.items()}), f)

    # Save best validation neural/model bundle if neural states exist
    if neural and best_states:
        best_n_name = pick(neural).replace("_avg", "")
        if best_n_name in best_states:
            torch.save(dict(
                run_id=rid,
                model_kind=best_n_name,
                c_in=int(best_states[best_n_name]["c_in"]),
                hidden=config.HIDDEN,
                mu=fold["mu"], sd=fold["sd"],
                d_mu=fold["d_mu"], d_sd=fold["d_sd"],
                w_stats=fold["w_stats"],
                target_mode=spec["target_mode"],
                weather_mode=spec["weather_mode"],
                horizon=spec["horizon"],
                seeds=list(seed_list),
                state_dicts=best_states[best_n_name]["states"],
                nnls_weights={k: float(v) for k, v in w.items()},
            ), os.path.join(rdir, "best_neural_bundle.pt"))

    meta_out = dict(
        **spec,
        train_n=int(len(fold["train"]["y"])),
        val_n=int(len(fold["val"]["y"])),
        test_n=int(len(fold["test"]["y"])),
        elapsed_s=float(time.time() - start_t),
        stack_weights={k: float(v) for k, v in w.items()},
    )
    with open(os.path.join(rdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    return res_idx


def run_trees_if_needed(spec, fold, Xtr_f, ytr, Xva_f, yva, inv_va, Xte_f, inv_te, out_dir):
    rdir = os.path.join(out_dir, "runs", spec["run_id"])
    os.makedirs(rdir, exist_ok=True)
    tp = os.path.join(rdir, "preds_trees.pkl")
    if os.path.exists(tp):
        with open(tp, "rb") as f:
            return pickle.load(f)

    va, te = {}, {}
    va["persistence"] = fold["val"]["y_curr"].astype(float)
    te["persistence"] = fold["test"]["y_curr"].astype(float)

    (lva, lte), (rva, rte, alpha) = fit_linear_and_ridge(Xtr_f, ytr, Xva_f, fold["val"]["y"], inv_va, Xte_f, inv_te)
    va["linear"], te["linear"] = lva, lte
    va["ridge"], te["ridge"] = rva, rte

    gva, gte, _ = fit_gbm(Xtr_f, ytr, Xva_f, yva, Xte_f)
    va["gbm"], te["gbm"] = inv_va(gva), inv_te(gte)

    xva, xte, _ = fit_xgb(Xtr_f, ytr, Xva_f, yva, Xte_f)
    va["xgboost"], te["xgboost"] = inv_va(xva), inv_te(xte)

    bundle = dict(val=va, test=te, ridge_alpha=float(alpha))
    with open(tp, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  [{spec['run_id']}] Trees done (ridge alpha={alpha})", flush=True)
    return bundle


def run_neural_family(kind, spec, Xtr_s, ytr, Xva_s, yva, inv_va, Xte_s, inv_te, te_y,
                      seed_list, epochs, out_dir):
    rdir = os.path.join(out_dir, "runs", spec["run_id"])
    ndir = os.path.join(rdir, "neural")
    os.makedirs(ndir, exist_ok=True)
    fp = os.path.join(ndir, f"{kind}.pkl")
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            return pickle.load(f)

    sv, st, srows, states = [], [], [], {}
    for s in seed_list:
        pva_sc, pte_sc, st_dict, ep_done = train_neural(
            kind, Xtr_s, ytr, Xva_s, yva, Xte_s, s,
            epochs=epochs, patience=config.PATIENCE, lr=config.LR,
            batch=config.BATCH, hidden=config.HIDDEN, use_bf16=False, preperm=True,
        )
        pva = inv_va(pva_sc)
        pte = inv_te(pte_sc)
        r_, a_ = _metrics(te_y, pte)
        sv.append(pva)
        st.append(pte)
        states[s] = st_dict
        srows.append(dict(model=kind, seed=s, epochs=ep_done, RMSE=r_, MAE=a_))

    out = dict(
        val_avg=np.mean(sv, axis=0),
        test_avg=np.mean(st, axis=0),
        seed_rows=srows,
        c_in=int(Xtr_s.shape[2]),
        states=states,
    )
    with open(fp, "wb") as f:
        pickle.dump(out, f)
    return out


def execute_spec(spec, clean, weather_caches, seed_list, epochs, out_dir, neural_models):
    t0 = time.time()
    rid = spec["run_id"]
    print(f"\n=== [{rid}] neural={neural_models} ===", flush=True)
    wdf = weather_caches.get(spec["weather_mode"])
    if spec["protocol"] == "P1_LOCO":
        fold = build_loco_fold(clean, spec["heldout"], L=config.LOOKBACK, h=spec["horizon"],
                               weather_mode=spec["weather_mode"], weather_df=wdf, gap_buffer=True)
    elif spec["protocol"] == "P2_2023":
        fold = build_p2_2023_fold(clean, L=config.LOOKBACK, h=spec["horizon"],
                                  weather_mode=spec["weather_mode"], weather_df=wdf)
    else:
        raise ValueError(spec["protocol"])

    Xtr_f, Xtr_s, ytr, _ = extract_features(fold["train"], fold, spec["target_mode"])
    Xva_f, Xva_s, yva, inv_va = extract_features(fold["val"], fold, spec["target_mode"])
    Xte_f, Xte_s, _, inv_te = extract_features(fold["test"], fold, spec["target_mode"])

    tb = run_trees_if_needed(spec, fold, Xtr_f, ytr, Xva_f, yva, inv_va, Xte_f, inv_te, out_dir)
    val_preds = dict(tb["val"])
    test_preds = dict(tb["test"])
    seed_rows = []
    best_states = {}

    for kind in neural_models:
        nb = run_neural_family(kind, spec, Xtr_s, ytr, Xva_s, yva, inv_va, Xte_s, inv_te,
                               fold["test"]["y"], seed_list, epochs, out_dir)
        val_preds[kind + "_avg"] = nb["val_avg"]
        test_preds[kind + "_avg"] = nb["test_avg"]
        seed_rows.extend(nb["seed_rows"])
        best_states[kind] = {"c_in": nb["c_in"], "states": nb["states"]}
        # Incrementally save after every completed neural family so crashes/timeouts never lose progress
        evaluate_and_save(spec, fold, Xva_f, Xte_f, val_preds, test_preds, seed_rows,
                          best_states, fold["val"]["y"], fold["test"], out_dir, t0, seed_list)

    if not neural_models:
        evaluate_and_save(spec, fold, Xva_f, Xte_f, val_preds, test_preds, seed_rows,
                          best_states, fold["val"]["y"], fold["test"], out_dir, t0, seed_list)

    build_report(out_dir)


def run_hybrid_dr_posthoc(out_dir):
    """Combine Direct & Residual predictions for matching (protocol, heldout, horizon, weather_mode) using val-only NNLS."""
    st_p = os.path.join(out_dir, "RUNS_STATUS.csv")
    if not os.path.exists(st_p):
        return
    st = pd.read_csv(st_p)
    rows = []
    for (prot, held, hor, wm), g in st.groupby(["protocol", "heldout", "horizon", "weather_mode"]):
        if not {"direct", "residual"} <= set(g["target_mode"]):
            continue
        r_d = g[g["target_mode"] == "direct"]["run_id"].iloc[0]
        r_r = g[g["target_mode"] == "residual"]["run_id"].iloc[0]
        pd_d = pickle.load(open(os.path.join(out_dir, "runs", r_d, "preds.pkl"), "rb"))
        pd_r = pickle.load(open(os.path.join(out_dir, "runs", r_r, "preds.pkl"), "rb"))
        assert len(pd_d["val_y"]) == len(pd_r["val_y"]) and len(pd_d["meta"]["y"]) == len(pd_r["meta"]["y"])

        val_cands, test_cands = {}, {}
        for m, v in pd_d["val"].items():
            if m not in ("persistence", "linear", "stack"):
                val_cands[f"D_{m}"] = v
                test_cands[f"D_{m}"] = pd_d["test"][m]
        for m, v in pd_r["val"].items():
            if m not in ("persistence", "linear", "stack"):
                val_cands[f"R_{m}"] = v
                test_cands[f"R_{m}"] = pd_r["test"][m]

        w, sv, st_pred = nnls_stack(val_cands, pd_d["val_y"], test_cands)
        y_te = pd_d["meta"]["y"]
        p_te = pd_d["test"]["persistence"]
        pr = E.rmse(p_te - y_te)
        pa = E.mae(p_te - y_te)
        hr, ha = _metrics(y_te, st_pred)
        # Best single D or R model on test
        best_single_name = min(test_cands, key=lambda k: E.rmse(test_cands[k] - y_te))
        bsr, bsa = _metrics(y_te, test_cands[best_single_name])

        rows.append(dict(
            protocol=prot, heldout=held, horizon=hor, weather_mode=wm,
            pers_RMSE=round(pr, 3),
            best_single=best_single_name,
            best_single_RMSE=round(bsr, 3),
            hybrid_DR_RMSE=round(hr, 3),
            hybrid_DR_MAE=round(ha, 3),
            hybrid_skill_pct=round(100 * (1 - hr / pr), 2),
            top_weights=json.dumps({k: round(float(v), 3) for k, v in w.items() if v > 0.01}),
        ))
    if rows:
        hdf = pd.DataFrame(rows).sort_values(["protocol", "horizon", "heldout", "weather_mode"])
        hdf.to_csv(os.path.join(out_dir, "HYBRID_DR_SUMMARY.csv"), index=False)


def export_best_checkpoints(out_dir):
    st_p = os.path.join(out_dir, "RUNS_STATUS.csv")
    if not os.path.exists(st_p):
        return
    st = pd.read_csv(st_p)
    ckpt_dir = os.path.join(out_dir, "best_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    import shutil

    # Remove stale incremental checkpoint copies before exporting final winners
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt") or f == "MANIFEST.csv":
            os.remove(os.path.join(ckpt_dir, f))

    exports = {}  # dst_name -> (category, row)
    for prot in sorted(st["protocol"].unique()):
        sub = st[st["protocol"] == prot]
        for h in sorted(sub["horizon"].unique()):
            sh = sub[sub["horizon"] == h]
            sk_row = sh.loc[sh["best_skill_RMSE_pct"].idxmax()]
            exports[f"best_skill_{prot}_h{h}_{sk_row['run_id']}.pt"] = ("best_skill", sk_row)

            rm_row = sh.loc[sh["best_test_RMSE"].idxmin()]
            if rm_row["run_id"] != sk_row["run_id"]:
                exports[f"lowest_rmse_{prot}_h{h}_{rm_row['run_id']}.pt"] = ("lowest_rmse", rm_row)

            sh_none = sh[sh["weather_mode"] == "none"]
            if not sh_none.empty:
                none_row = sh_none.loc[sh_none["best_skill_RMSE_pct"].idxmax()]
                if none_row["run_id"] not in (sk_row["run_id"], rm_row["run_id"]):
                    exports[f"best_none_{prot}_h{h}_{none_row['run_id']}.pt"] = ("best_none", none_row)

    # Also export top Delhi LOCO checkpoints explicitly (highest-pollution city)
    delhi = st[(st["protocol"] == "P1_LOCO") & (st["heldout"] == "Delhi")]
    for h in sorted(delhi["horizon"].unique()):
        dh = delhi[delhi["horizon"] == h]
        d_row = dh.loc[dh["best_skill_RMSE_pct"].idxmax()]
        exports[f"best_Delhi_h{h}_{d_row['run_id']}.pt"] = ("best_delhi", d_row)
        dh_none = dh[dh["weather_mode"] == "none"]
        if not dh_none.empty:
            dn_row = dh_none.loc[dh_none["best_skill_RMSE_pct"].idxmax()]
            if dn_row["run_id"] != d_row["run_id"]:
                exports[f"best_Delhi_none_h{h}_{dn_row['run_id']}.pt"] = ("best_delhi_none", dn_row)

    manifest = []
    for dst_name, (cat, row) in sorted(exports.items()):
        src = os.path.join(out_dir, "runs", row["run_id"], "best_neural_bundle.pt")
        if os.path.exists(src):
            dst = os.path.join(ckpt_dir, dst_name)
            shutil.copy2(src, dst)
            b = torch.load(dst, map_location="cpu", weights_only=False)
            manifest.append(dict(
                filename=dst_name,
                category=cat,
                run_id=row["run_id"],
                protocol=row["protocol"],
                heldout=row["heldout"],
                horizon=int(row["horizon"]),
                target_mode=row["target_mode"],
                weather_mode=row["weather_mode"],
                bundle_neural_arch=b["model_kind"],
                c_in=int(b["c_in"]),
                best_test_model=row["best_test_model"],
                best_test_RMSE=float(row["best_test_RMSE"]),
                best_skill_RMSE_pct=float(row["best_skill_RMSE_pct"]),
            ))
    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(ckpt_dir, "MANIFEST.csv"), index=False)


def build_task_queue(clean):
    cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad"]
    assert all(c in clean for c in cities)
    all_specs = []

    def add(protocol, heldout, horizon, target_mode, weather_mode, neural_models):
        rid = f"{protocol}_{heldout}_h{horizon}_{target_mode}_{weather_mode}"
        all_specs.append((dict(
            run_id=rid, protocol=protocol, heldout=heldout,
            horizon=horizon, target_mode=target_mode, weather_mode=weather_mode,
        ), neural_models))

    # Tier 1: P1 LOCO h=1 Direct + Residual across all 6 cities
    for c in cities:
        models = ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"] if c == "Delhi" else ["gru", "tcn", "cnn_lstm"]
        for tm in ("direct", "residual"):
            add("P1_LOCO", c, 1, tm, "none", models)

    # Tier 2: P1 LOCO h=24 Direct + Residual across all 6 cities
    for c in cities:
        models = ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"] if c == "Delhi" else ["gru", "tcn"]
        for tm in ("direct", "residual"):
            add("P1_LOCO", c, 24, tm, "none", models)

    # Tier 3: P2 2023 Temporal Holdout (h=1, 6, 12, 24 Direct + Residual)
    for h in (1, 24, 6, 12):
        models = ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"] if h in (1, 24) else ["gru", "tcn", "cnn_lstm"]
        for tm in ("direct", "residual"):
            add("P2_2023", "ALL", h, tm, "none", models)

    # Tier 4: Lagged Weather Ablations (CPCB & Open-Meteo <= t)
    for wm in ("cpcb", "openmeteo"):
        for tm in ("direct", "residual"):
            add("P1_LOCO", "Delhi", 1, tm, wm, ["gru", "cnn_lstm"])
            add("P2_2023", "ALL", 1, tm, wm, ["gru", "cnn_lstm"])
    for tm in ("direct", "residual"):
        add("P2_2023", "ALL", 24, tm, "openmeteo", ["gru", "cnn_lstm"])

    # Tier 5: Backfill remaining neural models across non-Delhi LOCO & secondary runs if budget remains
    backfill = []
    for h in (1, 24):
        for c in cities:
            if c == "Delhi":
                continue
            for tm in ("direct", "residual"):
                backfill.append((dict(
                    run_id=f"P1_LOCO_{c}_h{h}_{tm}_none",
                    protocol="P1_LOCO", heldout=c, horizon=h, target_mode=tm, weather_mode="none",
                ), ["gru", "lstm", "tcn", "cnn_lstm", "bilstm"]))

    return all_specs, backfill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_sweep")
    ap.add_argument("--max-hours", type=float, default=6.7)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(ROOT, "results", "clean.pkl"), "rb") as f:
        clean = pickle.load(f)
    weather_caches = {
        "none": None,
        "openmeteo": pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet")),
        "cpcb": pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_cpcb_24stations_2017_2023.parquet")),
    }

    if args.smoke:
        specs = [(dict(run_id="SMOKE_P2_h1_residual_openmeteo", protocol="P2_2023", heldout="ALL",
                       horizon=1, target_mode="residual", weather_mode="openmeteo"),
                  ["gru", "cnn_lstm"])]
        for s, ms in specs:
            execute_spec(s, clean, weather_caches, [0, 1], 2, args.out, ms)
        export_best_checkpoints(args.out)
        print("SMOKE TEST COMPLETE", flush=True)
        return

    t_global = time.time()
    max_sec = args.max_hours * 3600.0
    all_specs, backfill = build_task_queue(clean)

    # Pass 1: Fit fast Statistical + GBDT baselines across ALL planned configs (~12 min total)
    print(f"=== PASS 1: Statistical & Tree Baselines across all {len(all_specs)} configs ===", flush=True)
    for spec, _ in all_specs:
        rdir = os.path.join(args.out, "runs", spec["run_id"])
        if os.path.exists(os.path.join(rdir, "preds_trees.pkl")):
            continue
        execute_spec(spec, clean, weather_caches, config.SEEDS, config.EPOCHS, args.out, neural_models=[])

    # Pass 2: Full 5-seed Neural + CNN-LSTM + Stack + Bootstrap + Regimes in priority order
    print(f"\n=== PASS 2: Full 5-seed Neural Training ({len(all_specs)} runs, max_hours={args.max_hours}) ===", flush=True)
    for spec, nmodels in all_specs:
        if (time.time() - t_global) > max_sec:
            print(f"Time budget reached ({args.max_hours}h) before {spec['run_id']}; stopping cleanly.", flush=True)
            break
        execute_spec(spec, clean, weather_caches, config.SEEDS, config.EPOCHS, args.out, neural_models=nmodels)
        run_hybrid_dr_posthoc(args.out)
        export_best_checkpoints(args.out)
        build_report(args.out)

    # Pass 3: Backfill extra neural families (lstm, bilstm) if time remains
    for spec, nmodels in backfill:
        if (time.time() - t_global) > max_sec:
            break
        execute_spec(spec, clean, weather_caches, config.SEEDS, config.EPOCHS, args.out, neural_models=nmodels)
        run_hybrid_dr_posthoc(args.out)
        export_best_checkpoints(args.out)
        build_report(args.out)

    run_hybrid_dr_posthoc(args.out)
    export_best_checkpoints(args.out)
    build_report(args.out)
    print(f"\nOVERNIGHT SWEEP DONE in {(time.time()-t_global)/3600:.2f}h -> {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()

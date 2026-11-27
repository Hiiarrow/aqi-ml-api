"""
Final 20-station QC v3 study runner -> results_final/.
Does NOT touch results/, results_sweep/, results_qc2/, or results_qc3_stepA/.

Frozen config: LOOKBACK=24, HIDDEN=64, EPOCHS=60, PATIENCE=7, BATCH=2048, LR=1e-3, SEEDS=[0..4].
Neural families for EVERY config: gru, lstm, bilstm, tcn, cnn_lstm.
Primary fixed method: equal-weight neural ensemble (equal_neural_ens), direct target, no weather.

Baselines everywhere:
  - persistence (y_t)
  - same_hour_yesterday (h >= 6: y_{t+h-24})
  - climatology (train-only hour-of-day mean)
  - linear & validation-tuned ridge
  - gbm_direct, xgb_direct, gbm_residual, xgb_residual
  - best_tree (chosen on validation RMSE among the 4 trees)
  - equal_neural_ens (equal-weight mean of the 5 neural family ensembles)
  - stack (NNLS ablation on validation over direct models)

Statistics:
  - Paired DATE-block bootstrap (all stations on a date in one block), 1000 resamples, 95% CIs, NO p-values.
  - Regime tables (y_t >= 60, 120, 250, onset, winter, other, all) with RMSE & MAE skill, negatives kept.
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))

import config  # noqa: E402
import evaluate as E  # noqa: E402
from sweep.models_sweep import (  # noqa: E402
    fit_gbm,
    fit_linear_and_ridge,
    fit_xgb,
    nnls_stack,
    train_neural,
)
from sweep.splits import _align_weather, _cat, _finalize_fold, _slice_mask, _time_feats, extract_features  # noqa: E402


NEURAL_FAMS = ["gru", "lstm", "bilstm", "tcn", "cnn_lstm"]
LOCO_CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]
TREE_NAMES = ("gbm_direct", "xgb_direct", "gbm_residual", "xgb_residual")


def _metrics(y, p):
    e = p - y
    return E.rmse(e), E.mae(e)


def station_windows_final(key, city, name, st, L, h, weather_mode="none", weather_df=None):
    """Build sliding windows with DATE-level blocks and same_hour_yesterday (h>=6)."""
    v, t = st["value"], st["times"]
    n = len(v)
    valid = np.isfinite(v)
    cs = np.concatenate([[0], np.cumsum(valid)])
    idx = np.arange(L - 1, n - h)
    ok = (cs[idx + 1] - cs[idx + 1 - L] == L) & valid[idx + h] & ~st["interp"][idx + h]
    idx = idx[ok]
    if len(idx) == 0:
        return None

    tf = _time_feats(t)
    lag = sliding_window_view(v, L)[idx - L + 1].astype(np.float32)
    tfs = sliding_window_view(tf, L, axis=0)[idx - L + 1].transpose(0, 2, 1)
    tt = pd.DatetimeIndex(t[idx + h])

    out = dict(
        lag=lag,
        tf_seq=np.ascontiguousarray(tfs),
        tf_tgt=tf[idx + h],
        y=v[idx + h].astype(np.float32),
        y_curr=v[idx].astype(np.float32),
        rise3=(v[idx] - v[idx - 3]).astype(np.float32),
        month=tt.month.values,
        times=t[idx + h],
        # DATE block: all stations on calendar date d share the exact same block ID
        block_date=np.array(tt.strftime("%Y-%m-%d")),
        station=np.array([key] * len(idx)),
    )
    if h >= 6:
        # Target t+h minus 24h = t - (24 - h); for L=24 lag[:, j] = v[idx - 23 + j], so j = h - 1.
        out["same_hour_yesterday"] = lag[:, h - 1].astype(np.float32)

    W, _, _ = _align_weather(city, name, t, weather_mode, weather_df)
    if W is not None:
        w_seq = sliding_window_view(W, L, axis=0)[idx - L + 1].transpose(0, 2, 1)
        out["w_seq"] = np.ascontiguousarray(w_seq, dtype=np.float32)
    return out


def build_loco_fold_final(clean, heldout, L=24, h=1, weather_mode="none", weather_df=None):
    tr, va, te = [], [], []
    gap = np.timedelta64(h + L, "h")
    for city, stations in sorted(clean.items()):
        for name, st in sorted(stations.items()):
            key = f"{city}/{name}"
            w = station_windows_final(key, city, name, st, L, h, weather_mode, weather_df)
            if w is None:
                continue
            if city == heldout:
                te.append(w)
            else:
                n = len(w["y"])
                nv = int(n * config.VAL_FRACTION)
                val_start = w["times"][n - nv]
                m_va = w["times"] >= val_start
                m_tr = w["times"] <= (val_start - gap)
                if m_tr.sum() > 0:
                    tr.append(_slice_mask(w, m_tr))
                if m_va.sum() > 0:
                    va.append(_slice_mask(w, m_va))
    tr, va, te = _cat(tr), _cat(va), _cat(te)

    tr_st, va_st, te_st = set(tr["station"]), set(va["station"]), set(te["station"])
    assert tr_st.isdisjoint(te_st) and va_st.isdisjoint(te_st), "LEAKAGE: test stations in train/val"
    assert all(s.startswith(heldout + "/") for s in te_st), "test partition has non-target cities"
    assert not any(s.startswith(heldout + "/") for s in tr_st | va_st), "held-out city in train/val"
    return _finalize_fold(tr, va, te, L, h, weather_mode)


def build_p2_temporal_fold(clean, test_year, L=24, h=1, weather_mode="none", weather_df=None):
    """Train/val strictly < test_year (with h+L purge before val and before test_year), test in test_year."""
    cutoff_test = np.datetime64(f"{test_year}-01-01T00", "h")
    cutoff_end = np.datetime64(f"{test_year + 1}-01-01T00", "h")
    gap = np.timedelta64(h + L, "h")
    tr, va, te = [], [], []
    for city, stations in sorted(clean.items()):
        for name, st in sorted(stations.items()):
            key = f"{city}/{name}"
            w = station_windows_final(key, city, name, st, L, h, weather_mode, weather_df)
            if w is None:
                continue
            m_te = (w["times"] >= cutoff_test) & (w["times"] < cutoff_end)
            m_pre = w["times"] <= (cutoff_test - gap)
            if m_pre.sum() >= 200:
                pre = _slice_mask(w, m_pre)
                m = len(pre["y"])
                nv = max(1, int(m * config.VAL_FRACTION))
                val_start = pre["times"][m - nv]
                m_va = pre["times"] >= val_start
                m_tr = pre["times"] <= (val_start - gap)
                assert pre["times"][m_tr].max() + gap <= pre["times"][m_va].min(), f"train/val gap violation on {key}"
                assert pre["times"][m_va].max() + gap <= cutoff_test, f"val/test gap violation on {key}"
                tr.append(_slice_mask(pre, m_tr))
                va.append(_slice_mask(pre, m_va))
            if m_te.sum() > 0:
                te.append(_slice_mask(w, m_te))
    tr, va, te = _cat(tr), _cat(va), _cat(te)
    assert tr["times"].max() + gap <= te["times"].min()
    assert va["times"].max() + gap <= te["times"].min()
    assert te["times"].min() >= cutoff_test and te["times"].max() < cutoff_end
    return _finalize_fold(tr, va, te, L, h, weather_mode)


def compute_hour_climatology(fold):
    """Train-only hour-of-day climatology predicting E[y_{t+h} | target_hour] strictly from fold['train']."""
    tr_hr = pd.DatetimeIndex(fold["train"]["times"]).hour.values
    tr_y = fold["train"]["y"].astype(np.float64)
    gmu = float(np.mean(tr_y))
    by_hr = np.array([float(np.mean(tr_y[tr_hr == hr])) if np.any(tr_hr == hr) else gmu for hr in range(24)], dtype=np.float64)
    va_c = by_hr[pd.DatetimeIndex(fold["val"]["times"]).hour.values]
    te_c = by_hr[pd.DatetimeIndex(fold["test"]["times"]).hour.values]
    return va_c, te_c, by_hr


def run_baselines_and_trees(spec, fold, out_dir):
    rdir = os.path.join(out_dir, "runs", spec["run_id"])
    os.makedirs(rdir, exist_ok=True)
    tp = os.path.join(rdir, "preds_baselines.pkl")
    if os.path.exists(tp):
        with open(tp, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    Xtr_d, _, ytr_d, _ = extract_features(fold["train"], fold, "direct")
    Xva_d, _, yva_d, inv_va_d = extract_features(fold["val"], fold, "direct")
    Xte_d, _, _, inv_te_d = extract_features(fold["test"], fold, "direct")

    _, _, ytr_r, _ = extract_features(fold["train"], fold, "residual")
    _, _, yva_r, inv_va_r = extract_features(fold["val"], fold, "residual")
    _, _, _, inv_te_r = extract_features(fold["test"], fold, "residual")

    va, te = {}, {}
    va["persistence"] = fold["val"]["y_curr"].astype(np.float64)
    te["persistence"] = fold["test"]["y_curr"].astype(np.float64)

    if spec["horizon"] >= 6:
        va["same_hour_yesterday"] = fold["val"]["same_hour_yesterday"].astype(np.float64)
        te["same_hour_yesterday"] = fold["test"]["same_hour_yesterday"].astype(np.float64)

    va_cl, te_cl, _ = compute_hour_climatology(fold)
    va["climatology"], te["climatology"] = va_cl, te_cl

    (lva, lte), (rva, rte, alpha) = fit_linear_and_ridge(
        Xtr_d, ytr_d, Xva_d, fold["val"]["y"], inv_va_d, Xte_d, inv_te_d
    )
    va["linear"], te["linear"] = lva, lte
    va["ridge"], te["ridge"] = rva, rte

    # Direct trees
    gd_va, gd_te, _ = fit_gbm(Xtr_d, ytr_d, Xva_d, yva_d, Xte_d)
    va["gbm_direct"], te["gbm_direct"] = inv_va_d(gd_va), inv_te_d(gd_te)
    xd_va, xd_te, _ = fit_xgb(Xtr_d, ytr_d, Xva_d, yva_d, Xte_d)
    va["xgb_direct"], te["xgb_direct"] = inv_va_d(xd_va), inv_te_d(xd_te)

    # Residual trees (y_{t+h} - y_t)
    gr_va, gr_te, _ = fit_gbm(Xtr_d, ytr_r, Xva_d, yva_r, Xte_d)
    va["gbm_residual"], te["gbm_residual"] = inv_va_r(gr_va), inv_te_r(gr_te)
    xr_va, xr_te, _ = fit_xgb(Xtr_d, ytr_r, Xva_d, yva_r, Xte_d)
    va["xgb_residual"], te["xgb_residual"] = inv_va_r(xr_va), inv_te_r(xr_te)

    # Validation-selected best_tree across all 4 tree models
    val_y = fold["val"]["y"].astype(np.float64)
    best_tree_choice = min(TREE_NAMES, key=lambda m: E.rmse(va[m] - val_y))
    va["best_tree"] = va[best_tree_choice]
    te["best_tree"] = te[best_tree_choice]

    bundle = dict(val=va, test=te, ridge_alpha=float(alpha), best_tree_choice=best_tree_choice)
    with open(tp, "wb") as f:
        pickle.dump(bundle, f)
    print(
        f"  [{spec['run_id']}] Baselines+Trees done in {time.time()-t0:.1f}s "
        f"(ridge_alpha={alpha}, best_tree={best_tree_choice})",
        flush=True,
    )
    return bundle


def run_neural_family_final(kind, spec, Xtr_s, ytr_d, Xva_s, yva_d, inv_va_d, Xte_s, inv_te_d, te_y, out_dir):
    rdir = os.path.join(out_dir, "runs", spec["run_id"])
    ndir = os.path.join(rdir, "neural")
    os.makedirs(ndir, exist_ok=True)
    fp = os.path.join(ndir, f"{kind}.pkl")
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            return pickle.load(f)

    sv, st, srows = [], [], []
    for s in config.SEEDS:
        pva_sc, pte_sc, _, ep_done = train_neural(
            kind, Xtr_s, ytr_d, Xva_s, yva_d, Xte_s, s,
            epochs=config.EPOCHS, patience=config.PATIENCE, lr=config.LR,
            batch=config.BATCH, hidden=config.HIDDEN, use_bf16=False, preperm=True,
        )
        pva = inv_va_d(pva_sc)
        pte = inv_te_d(pte_sc)
        r_, a_ = _metrics(te_y, pte)
        sv.append(pva)
        st.append(pte)
        srows.append(dict(model=kind, seed=s, epochs=ep_done, RMSE=r_, MAE=a_))

    out = dict(
        val_avg=np.mean(sv, axis=0),
        test_avg=np.mean(st, axis=0),
        seed_rows=srows,
        c_in=int(Xtr_s.shape[2]),
    )
    with open(fp, "wb") as f:
        pickle.dump(out, f)
    return out


def save_run_outputs(spec, fold, val_preds, test_preds, seed_rows, ridge_alpha, best_tree_choice, out_dir, start_t):
    rid = spec["run_id"]
    rdir = os.path.join(out_dir, "runs", rid)
    os.makedirs(rdir, exist_ok=True)

    val_y = fold["val"]["y"].astype(np.float64)
    te = fold["test"]
    te_y = te["y"].astype(np.float64)

    nfams_done = [m + "_avg" for m in NEURAL_FAMS if (m + "_avg") in test_preds]
    if nfams_done:
        val_preds["equal_neural_ens"] = np.mean([val_preds[m] for m in nfams_done], axis=0)
        test_preds["equal_neural_ens"] = np.mean([test_preds[m] for m in nfams_done], axis=0)

    # NNLS stack ablation on validation over direct-trained config models (ridge, gbm_direct, xgb_direct, neural_avgs)
    stack_members = [m for m in (["ridge", "gbm_direct", "xgb_direct"] + nfams_done) if m in val_preds]
    w, sv, st = nnls_stack({m: val_preds[m] for m in stack_members}, val_y,
                           {m: test_preds[m] for m in stack_members})
    val_all = dict(val_preds, stack=sv)
    test_all = dict(test_preds, stack=st)

    # Results table
    p_rmse, p_mae = _metrics(te_y, test_all["persistence"])
    rows = []
    for m, pte in test_all.items():
        if pte is None:
            continue
        r_, a_ = _metrics(te_y, pte)
        rv, _ = _metrics(val_y, val_all[m])
        rows.append(dict(
            model=m, val_RMSE=rv, test_RMSE=r_, test_MAE=a_,
            skill_RMSE_pct=100.0 * (1.0 - r_ / p_rmse),
            skill_MAE_pct=100.0 * (1.0 - a_ / p_mae),
        ))
    pd.DataFrame(rows).to_csv(os.path.join(rdir, "results.csv"), index=False)

    if seed_rows:
        pd.DataFrame(seed_rows).to_csv(os.path.join(rdir, "seed_rows.csv"), index=False)

    # DATE-block paired bootstrap CIs (no p-values): negative dRMSE/dMAE means A < B (A has lower error)
    pairs = []
    if "equal_neural_ens" in test_all:
        pairs.append(("equal_neural_ens", "persistence"))
        pairs.append(("equal_neural_ens", "best_tree"))
        if spec["horizon"] >= 6:
            pairs.append(("equal_neural_ens", "same_hour_yesterday"))
        pairs.append(("stack", "equal_neural_ens"))
    if pairs:
        boot = pd.DataFrame(E.paired_block_bootstrap(te_y, test_all, te["block_date"], pairs))
        boot.to_csv(os.path.join(rdir, "bootstrap.csv"), index=False)

    # Time-t regime table
    reg = pd.DataFrame(E.regime_table(te, te_y, test_all))
    reg.to_csv(os.path.join(rdir, "regimes.csv"), index=False)

    # Save exact predictions & DATE-block metadata for cross-run openmeteo-vs-none CIs and audit
    with open(os.path.join(rdir, "preds.pkl"), "wb") as f:
        pickle.dump(dict(
            val=val_all, test=test_all, val_y=val_y,
            meta={k: te[k] for k in ("y", "y_curr", "rise3", "month", "times", "block_date", "station")},
            stack_weights={k: float(v) for k, v in w.items()},
            best_tree_choice=best_tree_choice,
        ), f)

    tr_sts = sorted(set(fold["train"]["station"]))
    te_sts = sorted(set(fold["test"]["station"]))
    meta_out = dict(
        **spec,
        n_train_stations=len(tr_sts),
        n_test_stations=len(te_sts),
        train_stations=tr_sts,
        test_stations=te_sts,
        train_n=int(len(fold["train"]["y"])),
        val_n=int(len(fold["val"]["y"])),
        test_n=int(len(fold["test"]["y"])),
        n_date_blocks=int(len(np.unique(te["block_date"]))),
        ridge_alpha=float(ridge_alpha),
        best_tree_choice=best_tree_choice,
        elapsed_s=float(time.time() - start_t),
        stack_weights={k: float(v) for k, v in w.items()},
    )
    with open(os.path.join(rdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)


def execute_run(spec, clean_23, clean_24, om_23, om_24, out_dir):
    rdir = os.path.join(out_dir, "runs", spec["run_id"])
    meta_p = os.path.join(rdir, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p, "r", encoding="utf-8") as f:
            m = json.load(f)
        # Skip only if all 5 neural families already completed
        ndir = os.path.join(rdir, "neural")
        if all(os.path.exists(os.path.join(ndir, f"{k}.pkl")) for k in NEURAL_FAMS):
            print(f"\n=== [{spec['run_id']}] SKIP (already complete) ===", flush=True)
            return

    t0 = time.time()
    rid = spec["run_id"]
    print(f"\n=== [{rid}] START ===", flush=True)
    prot = spec["protocol"]
    if prot == "P2_2023":
        fold = build_p2_temporal_fold(
            clean_23, test_year=2023, L=config.LOOKBACK, h=spec["horizon"],
            weather_mode=spec["weather_mode"], weather_df=om_23 if spec["weather_mode"] == "openmeteo" else None,
        )
    elif prot == "P2_2024":
        fold = build_p2_temporal_fold(
            clean_24, test_year=2024, L=config.LOOKBACK, h=spec["horizon"],
            weather_mode=spec["weather_mode"], weather_df=om_24 if spec["weather_mode"] == "openmeteo" else None,
        )
    elif prot == "P1_LOCO":
        fold = build_loco_fold_final(
            clean_23, heldout=spec["heldout"], L=config.LOOKBACK, h=spec["horizon"],
            weather_mode=spec["weather_mode"], weather_df=om_23 if spec["weather_mode"] == "openmeteo" else None,
        )
    else:
        raise ValueError(prot)

    bb = run_baselines_and_trees(spec, fold, out_dir)
    val_preds = dict(bb["val"])
    test_preds = dict(bb["test"])

    _, Xtr_s, ytr_d, _ = extract_features(fold["train"], fold, "direct")
    _, Xva_s, yva_d, inv_va_d = extract_features(fold["val"], fold, "direct")
    _, Xte_s, _, inv_te_d = extract_features(fold["test"], fold, "direct")
    te_y = fold["test"]["y"].astype(np.float64)

    seed_rows = []
    for kind in NEURAL_FAMS:
        nb = run_neural_family_final(kind, spec, Xtr_s, ytr_d, Xva_s, yva_d, inv_va_d, Xte_s, inv_te_d, te_y, out_dir)
        val_preds[kind + "_avg"] = nb["val_avg"]
        test_preds[kind + "_avg"] = nb["test_avg"]
        seed_rows.extend(nb["seed_rows"])
        save_run_outputs(
            spec, fold, val_preds, test_preds, seed_rows,
            bb["ridge_alpha"], bb["best_tree_choice"], out_dir, t0,
        )

    del fold, Xtr_s, Xva_s, Xte_s
    gc.collect()
    print(f"=== [{rid}] COMPLETE in {(time.time() - t0)/60:.1f} min ===", flush=True)


def build_final_specs():
    specs = []

    def add(phase, protocol, heldout, horizon, weather_mode):
        rid = f"{protocol}_{heldout}_h{horizon}_direct_{weather_mode}"
        specs.append(dict(
            phase=phase, run_id=rid, protocol=protocol, heldout=heldout,
            horizon=horizon, target_mode="direct", weather_mode=weather_mode,
        ))

    # Phase 1: P2_2023 (h=1,6,12,24) + P1_LOCO Delhi (h=1,24)
    for h in (1, 6, 12, 24):
        for wm in ("none", "openmeteo"):
            add(1, "P2_2023", "ALL20", h, wm)
    for h in (1, 24):
        for wm in ("none", "openmeteo"):
            add(1, "P1_LOCO", "Delhi", h, wm)

    # Phase 2: P2_2024 (h=1,24) + P1_LOCO remaining 4 cities (h=1)
    for h in (1, 24):
        for wm in ("none", "openmeteo"):
            add(2, "P2_2024", "ALL19", h, wm)
    for city in ("Mumbai", "Kolkata", "Chennai", "Bengaluru"):
        for wm in ("none", "openmeteo"):
            add(2, "P1_LOCO", city, 1, wm)

    return specs


def compile_final_summary(out_dir):
    specs = build_final_specs()
    runs_dir = os.path.join(out_dir, "runs")

    lb_rows, boot_rows, reg_rows, seed_rows, run_meta_rows = [], [], [], [], []
    for sp in specs:
        rid = sp["run_id"]
        rdir = os.path.join(runs_dir, rid)
        if not os.path.exists(os.path.join(rdir, "meta.json")):
            continue
        with open(os.path.join(rdir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        res = pd.read_csv(os.path.join(rdir, "results.csv"))
        for col, val in [("run_id", rid), ("protocol", meta["protocol"]), ("heldout", meta["heldout"]),
                         ("horizon", meta["horizon"]), ("weather_mode", meta["weather_mode"]),
                         ("N_test", meta["test_n"]), ("n_date_blocks", meta["n_date_blocks"]),
                         ("best_tree_choice", meta["best_tree_choice"])]:
            res[col] = val
        lb_rows.append(res)

        pers = res.loc[res["model"] == "persistence"].iloc[0]
        eq = res.loc[res["model"] == "equal_neural_ens"].iloc[0]
        bt = res.loc[res["model"] == "best_tree"].iloc[0]
        st = res.loc[res["model"] == "stack"].iloc[0]
        rg = res.loc[res["model"] == "ridge"].iloc[0]
        cl = res.loc[res["model"] == "climatology"].iloc[0]
        sy = res.loc[res["model"] == "same_hour_yesterday"]
        run_meta_rows.append(dict(
            run_id=rid, protocol=meta["protocol"], heldout=meta["heldout"], horizon=meta["horizon"],
            weather_mode=meta["weather_mode"], N_train=meta["train_n"], N_val=meta["val_n"],
            N_test=meta["test_n"], n_test_stations=meta["n_test_stations"], n_date_blocks=meta["n_date_blocks"],
            persist_RMSE=round(float(pers["test_RMSE"]), 3),
            persist_MAE=round(float(pers["test_MAE"]), 3),
            eq_ens_RMSE=round(float(eq["test_RMSE"]), 3),
            eq_ens_MAE=round(float(eq["test_MAE"]), 3),
            eq_ens_skill_RMSE=round(float(eq["skill_RMSE_pct"]), 2),
            eq_ens_skill_MAE=round(float(eq["skill_MAE_pct"]), 2),
            best_tree_choice=meta["best_tree_choice"],
            best_tree_RMSE=round(float(bt["test_RMSE"]), 3),
            best_tree_skill_RMSE=round(float(bt["skill_RMSE_pct"]), 2),
            ridge_skill_RMSE=round(float(rg["skill_RMSE_pct"]), 2),
            clim_skill_RMSE=round(float(cl["skill_RMSE_pct"]), 2),
            same_hr_yest_skill_RMSE=round(float(sy["skill_RMSE_pct"].iloc[0]), 2) if not sy.empty else np.nan,
            stack_RMSE=round(float(st["test_RMSE"]), 3),
            stack_skill_RMSE=round(float(st["skill_RMSE_pct"]), 2),
            stack_weights=json.dumps({k: round(v, 4) for k, v in meta["stack_weights"].items() if v > 0.001}),
        ))

        bp = os.path.join(rdir, "bootstrap.csv")
        if os.path.exists(bp):
            b = pd.read_csv(bp)
            b.insert(0, "weather_mode", meta["weather_mode"])
            b.insert(0, "horizon", meta["horizon"])
            b.insert(0, "heldout", meta["heldout"])
            b.insert(0, "protocol", meta["protocol"])
            b.insert(0, "run_id", rid)
            boot_rows.append(b)

        rp = os.path.join(rdir, "regimes.csv")
        if os.path.exists(rp):
            rg_df = pd.read_csv(rp)
            rg_df.insert(0, "weather_mode", meta["weather_mode"])
            rg_df.insert(0, "horizon", meta["horizon"])
            rg_df.insert(0, "heldout", meta["heldout"])
            rg_df.insert(0, "protocol", meta["protocol"])
            rg_df.insert(0, "run_id", rid)
            reg_rows.append(rg_df)

        sdp = os.path.join(rdir, "seed_rows.csv")
        if os.path.exists(sdp):
            sd_df = pd.read_csv(sdp)
            sd_df.insert(0, "weather_mode", meta["weather_mode"])
            sd_df.insert(0, "horizon", meta["horizon"])
            sd_df.insert(0, "heldout", meta["heldout"])
            sd_df.insert(0, "protocol", meta["protocol"])
            sd_df.insert(0, "run_id", rid)
            seed_rows.append(sd_df)

    if run_meta_rows:
        pd.DataFrame(run_meta_rows).to_csv(os.path.join(out_dir, "FINAL_SUMMARY.csv"), index=False)
    if lb_rows:
        pd.concat(lb_rows, ignore_index=True).to_csv(os.path.join(out_dir, "LEADERBOARD.csv"), index=False)
    if boot_rows:
        pd.concat(boot_rows, ignore_index=True).to_csv(os.path.join(out_dir, "ALL_BOOTSTRAP_WITHIN.csv"), index=False)
    if reg_rows:
        pd.concat(reg_rows, ignore_index=True).to_csv(os.path.join(out_dir, "ALL_REGIMES.csv"), index=False)
    if seed_rows:
        s_all = pd.concat(seed_rows, ignore_index=True)
        s_all.to_csv(os.path.join(out_dir, "ALL_SEEDS.csv"), index=False)
        s_stat = (
            s_all.groupby(["run_id", "protocol", "heldout", "horizon", "weather_mode", "model"], as_index=False)
            .agg(
                n_seeds=("seed", "count"),
                RMSE_mean=("RMSE", "mean"),
                RMSE_std=("RMSE", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            )
        )
        s_stat.to_csv(os.path.join(out_dir, "SEED_STATS.csv"), index=False)

    # Cross-run Open-Meteo vs None DATE-block bootstrap CIs on matched pairs
    om_rows = []
    by_key = {(sp["protocol"], sp["heldout"], sp["horizon"], sp["weather_mode"]): sp["run_id"] for sp in specs}
    pairs_seen = sorted({(p, hld, h) for (p, hld, h, _) in by_key})
    for prot, hld, hor in pairs_seen:
        rn_none = by_key.get((prot, hld, hor, "none"))
        rn_om = by_key.get((prot, hld, hor, "openmeteo"))
        if not (rn_none and rn_om):
            continue
        pn = os.path.join(runs_dir, rn_none, "preds.pkl")
        po = os.path.join(runs_dir, rn_om, "preds.pkl")
        if not (os.path.exists(pn) and os.path.exists(po)):
            continue
        with open(pn, "rb") as f:
            dn = pickle.load(f)
        with open(po, "rb") as f:
            do = pickle.load(f)
        mn, mo = dn["meta"], do["meta"]
        assert np.array_equal(mn["y"], mo["y"]), f"y mismatch for {prot} {hld} h={hor}"
        assert np.array_equal(mn["station"], mo["station"]), f"station mismatch for {prot} {hld} h={hor}"
        assert np.array_equal(mn["times"], mo["times"]), f"times mismatch for {prot} {hld} h={hor}"
        assert np.array_equal(mn["block_date"], mo["block_date"]), f"block_date mismatch for {prot} {hld} h={hor}"
        y = mn["y"].astype(np.float64)
        comp = {
            "openmeteo_ens": do["test"]["equal_neural_ens"],
            "none_ens": dn["test"]["equal_neural_ens"],
        }
        brow = E.paired_block_bootstrap(y, comp, mn["block_date"], [("openmeteo_ens", "none_ens")])[0]
        om_rows.append(dict(protocol=prot, heldout=hld, horizon=hor, N_test=len(y), **brow))
    if om_rows:
        pd.DataFrame(om_rows).to_csv(os.path.join(out_dir, "OPENMETEO_VS_NONE_CIS.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_final")
    ap.add_argument("--phase", type=int, choices=[0, 1, 2], default=0, help="0=all, 1=P2_2023+Delhi, 2=P2_2024+other LOCO")
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    if args.compile_only:
        compile_final_summary(out_dir)
        print("Compiled final summary tables ->", out_dir, flush=True)
        return

    with open(os.path.join(out_dir, "clean_20_2017_2023_v3.pkl"), "rb") as f:
        clean_23 = pickle.load(f)
    with open(os.path.join(out_dir, "clean_20_2017_2024_v3.pkl"), "rb") as f:
        clean_24 = pickle.load(f)
    om_23 = pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet"))
    om_24 = pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2024.parquet"))

    specs = build_final_specs()
    if args.phase in (1, 2):
        specs = [s for s in specs if s["phase"] == args.phase]

    t0 = time.time()
    for sp in specs:
        execute_run(sp, clean_23, clean_24, om_23, om_24, out_dir)
        compile_final_summary(out_dir)

    print(f"\nPhase {args.phase} finished cleanly in {(time.time() - t0)/60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

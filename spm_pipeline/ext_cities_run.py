"""External validation runner for new CPCB cities (Steps A-E).

Reuses frozen settings, models, QC v3 rules, seeds, and splits from spm_pipeline.
Reads original 5-city clean caches & results_final/ strictly read-only.
Writes all new weights, predictions, and tables exclusively to results_ext_cities/.
Runs direct_none completely (Steps A-D + saved CSVs) before starting direct_openmeteo.
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))

import compute_r2 as CR2  # noqa: E402
import config  # noqa: E402
import evaluate as E  # noqa: E402
import final_study as FS  # noqa: E402
import qc  # noqa: E402
from sweep.models_sweep import build_torch_model, fit_gbm, fit_xgb, train_neural  # noqa: E402
from sweep.splits import _cat, extract_features  # noqa: E402

EXT_DIR = os.path.join(ROOT, "results_ext_cities")
MAIN_DIR = os.path.join(ROOT, "results_final")
MODELS_DIR = os.path.join(EXT_DIR, "models")
PREDS_DIR = os.path.join(EXT_DIR, "preds")
OC_DIR = os.path.join(EXT_DIR, "opencity_2022_2024")
CACHE_DIR = os.path.join(EXT_DIR, "cache")

PRIMARY_CITIES = [
    "Agra", "Chandigarh", "Gaya", "Ghaziabad", "Meerut",
    "Muzaffarpur", "Noida", "Patna", "Prayagraj", "Varanasi",
]
SECONDARY_CITIES = [
    "Lucknow", "Moradabad", "Jaipur", "Howrah", "Gurugram", "Kanpur",
]
NEURAL_FAMS = FS.NEURAL_FAMS
TREE_NAMES = FS.TREE_NAMES
SUR_PRES_IDX = 5  # index of 'surface_pressure' in splits._align_weather(..., 'openmeteo')


def build_config_specs(weather_mode):
    """Return frozen P2_2023 (h=1,6,12,24) and P2_2024 (h=1,24) specs for a given weather_mode."""
    specs = []
    for h in (1, 6, 12, 24):
        specs.append(dict(
            run_id=f"P2_2023_ALL20_h{h}_direct_{weather_mode}",
            protocol="P2_2023",
            test_year=2023,
            heldout="ALL20",
            horizon=h,
            weather_mode=weather_mode,
        ))
    for h in (1, 24):
        specs.append(dict(
            run_id=f"P2_2024_ALL19_h{h}_direct_{weather_mode}",
            protocol="P2_2024",
            test_year=2024,
            heldout="ALL19",
            horizon=h,
            weather_mode=weather_mode,
        ))
    return specs


def fit_linear_and_ridge_with_models(Xtr, ytr, Xva, va_y_raw, inv_va, Xte, inv_te):
    """Exact replica of models_sweep.fit_linear_and_ridge that also returns fitted sklearn models."""
    from sklearn.linear_model import LinearRegression, Ridge

    lin = LinearRegression()
    lin.fit(Xtr, ytr)
    lin_va = inv_va(lin.predict(Xva))
    lin_te = inv_te(lin.predict(Xte))

    best_alpha, best_rmse, best_rva, best_r = None, float("inf"), None, None
    for a in (0.1, 1.0, 3.2, 10.0, 32.0, 100.0, 1000.0):
        m = Ridge(alpha=a)
        m.fit(Xtr, ytr)
        pva = inv_va(m.predict(Xva))
        r = float(np.sqrt(np.mean((pva - va_y_raw) ** 2)))
        if r < best_rmse:
            best_rmse, best_alpha, best_rva, best_r = r, a, pva, m
    best_rte = inv_te(best_r.predict(Xte))
    return (lin, lin_va, lin_te), (best_r, best_rva, best_rte, best_alpha)


def predict_torch_from_checkpoint(ckpt_path, X_seq, batch_size=8192):
    """Load saved PyTorch state_dict checkpoint and run batched inference."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_torch_model(ckpt["kind"], ckpt["c_in"], config.HIDDEN).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in ckpt["state_dict"].items()})
    model.eval()
    X_t = torch.from_numpy(np.ascontiguousarray(X_seq, dtype=np.float32))
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(dev)
            out.append(model(xb).float().cpu().numpy())
    del model, X_t
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0) if out else np.empty(0, dtype=np.float32)


def prepare_ext_city_clean_cache():
    """Run qc.qc_station_v3 on fixed PRIMARY (39) and SECONDARY (22) stations and cache."""
    cache_p = os.path.join(CACHE_DIR, "clean_ext16_2022_2024_v3.pkl")
    if os.path.exists(cache_p):
        with open(cache_p, "rb") as f:
            return pickle.load(f)

    st_audit = pd.read_csv(os.path.join(EXT_DIR, "step1_station_audit_2022_2024.csv"))
    keep = st_audit[
        st_audit["city"].isin(PRIMARY_CITIES + SECONDARY_CITIES)
        & (st_audit["valid_frac"] >= 0.85)
        & (~st_audit["ceiling_flag"])
        & (st_audit["qc_status"] == "retained")
    ].copy()

    clean_by_group = {"PRIMARY": {}, "SECONDARY": {}}
    for _, r in keep.iterrows():
        city, st_name, rel = r["city"], r["station"], r["csv_rel"]
        grp = "PRIMARY" if city in PRIMARY_CITIES else "SECONDARY"
        st_obj, rep = qc.qc_station_v3(os.path.join(OC_DIR, rel))
        assert st_obj is not None and rep["status"] == "retained", f"Unexpected QC drop for {city}/{st_name}"
        clean_by_group[grp].setdefault(city, {})[st_name] = st_obj

    n_prim_c = len(clean_by_group["PRIMARY"])
    n_prim_s = sum(len(v) for v in clean_by_group["PRIMARY"].values())
    n_sec_c = len(clean_by_group["SECONDARY"])
    n_sec_s = sum(len(v) for v in clean_by_group["SECONDARY"].values())
    assert n_prim_c == 10 and n_prim_s == 39, f"Expected 10/39 primary, got {n_prim_c}/{n_prim_s}"
    assert n_sec_c == 6 and n_sec_s == 22, f"Expected 6/22 secondary, got {n_sec_c}/{n_sec_s}"

    with open(cache_p, "wb") as f:
        pickle.dump(clean_by_group, f)
    print(f"Cached clean QC v3 dicts -> {cache_p} (PRIMARY={n_prim_c}/{n_prim_s}, SECONDARY={n_sec_c}/{n_sec_s})", flush=True)
    return clean_by_group


def _ci_status(lo, hi):
    if hi < 0.0:
        return "entirely_below_zero"
    if lo > 0.0:
        return "entirely_above_zero"
    return "includes_zero"


def train_and_verify_config(spec, clean_23, clean_24, om_23, om_24):
    """Step A: retrain one configuration on original 5-city fold, save all weights+scalers, verify vs preds.pkl."""
    rid = spec["run_id"]
    wm = spec["weather_mode"]
    h = spec["horizon"]
    mdir = os.path.join(MODELS_DIR, rid)
    ndir = os.path.join(mdir, "neural")
    os.makedirs(ndir, exist_ok=True)
    ver_p = os.path.join(mdir, "verification.json")
    scalers_p = os.path.join(mdir, "scalers.pkl")
    bb_p = os.path.join(mdir, "baselines_models.pkl")

    clean = clean_23 if spec["protocol"] == "P2_2023" else clean_24
    wdf = (om_23 if spec["protocol"] == "P2_2023" else om_24) if wm == "openmeteo" else None
    fold = FS.build_p2_temporal_fold(clean, test_year=spec["test_year"], L=config.LOOKBACK, h=h, weather_mode=wm, weather_df=wdf)

    # Save exact scalers + training surface_pressure bounds
    train_sp_range = None
    if wm == "openmeteo" and "w_seq" in fold["train"]:
        tr_sp = fold["train"]["w_seq"][:, :, SUR_PRES_IDX]
        train_sp_range = (float(np.nanmin(tr_sp)), float(np.nanmax(tr_sp)))
    scalers = dict(
        mu=fold["mu"], sd=fold["sd"], d_mu=fold["d_mu"], d_sd=fold["d_sd"],
        L=fold["L"], h=fold["h"], weather_mode=wm, w_stats=fold["w_stats"],
        train_sp_range=train_sp_range,
    )
    if not os.path.exists(scalers_p):
        with open(scalers_p, "wb") as f:
            pickle.dump(scalers, f)

    Xtr_d, Xtr_s, ytr_d, _ = extract_features(fold["train"], fold, "direct")
    Xva_d, Xva_s, yva_d, inv_va_d = extract_features(fold["val"], fold, "direct")
    Xte_d, Xte_s, _, inv_te_d = extract_features(fold["test"], fold, "direct")
    _, _, ytr_r, _ = extract_features(fold["train"], fold, "residual")
    _, _, yva_r, inv_va_r = extract_features(fold["val"], fold, "residual")
    _, _, _, inv_te_r = extract_features(fold["test"], fold, "residual")

    if not os.path.exists(bb_p):
        t_bb = time.time()
        val_y = fold["val"]["y"].astype(np.float64)
        (lin_m, lin_va, _), (ridge_m, ridge_va, _, ridge_alpha) = fit_linear_and_ridge_with_models(
            Xtr_d, ytr_d, Xva_d, val_y, inv_va_d, Xte_d, inv_te_d
        )
        gd_va, _, gd_m = fit_gbm(Xtr_d, ytr_d, Xva_d, yva_d, Xte_d)
        xd_va, _, xd_m = fit_xgb(Xtr_d, ytr_d, Xva_d, yva_d, Xte_d)
        gr_va, _, gr_m = fit_gbm(Xtr_d, ytr_r, Xva_d, yva_r, Xte_d)
        xr_va, _, xr_m = fit_xgb(Xtr_d, ytr_r, Xva_d, yva_r, Xte_d)

        va_trees = {
            "gbm_direct": inv_va_d(gd_va),
            "xgb_direct": inv_va_d(xd_va),
            "gbm_residual": inv_va_r(gr_va),
            "xgb_residual": inv_va_r(xr_va),
        }
        best_tree_choice = min(TREE_NAMES, key=lambda m: E.rmse(va_trees[m] - val_y))
        bundle = dict(
            linear=lin_m,
            ridge=ridge_m,
            ridge_alpha=float(ridge_alpha),
            trees=dict(gbm_direct=gd_m, xgb_direct=xd_m, gbm_residual=gr_m, xgb_residual=xr_m),
            best_tree_choice=best_tree_choice,
            val_preds=dict(ridge=ridge_va, **va_trees),
        )
        with open(bb_p, "wb") as f:
            pickle.dump(bundle, f)
        print(f"  [{rid}] Saved ridge + 4 tree models (best_tree={best_tree_choice}, ridge_alpha={ridge_alpha}) in {time.time()-t_bb:.1f}s", flush=True)
    else:
        with open(bb_p, "rb") as f:
            bundle = pickle.load(f)

    # Train and save all 25 PyTorch state_dicts cleanly with per-seed resume
    for kind in NEURAL_FAMS:
        for s in config.SEEDS:
            pt_p = os.path.join(ndir, f"{kind}_seed{s}.pt")
            if os.path.exists(pt_p):
                continue
            _, _, best_state, ep_done = train_neural(
                kind, Xtr_s, ytr_d, Xva_s, yva_d, Xte_s, s,
                epochs=config.EPOCHS, patience=config.PATIENCE, lr=config.LR,
                batch=config.BATCH, hidden=config.HIDDEN, use_bf16=False, preperm=True,
            )
            torch.save(
                dict(state_dict=best_state, c_in=int(Xtr_s.shape[2]), epochs=int(ep_done), kind=kind, seed=int(s)),
                pt_p,
            )

    if os.path.exists(ver_p):
        with open(ver_p, "r", encoding="utf-8") as f:
            ver_rows = json.load(f)
        return fold, scalers, bundle, ver_rows

    # Verify by RELOADING saved models from disk on original test set & comparing to results_final preds.pkl
    with open(os.path.join(MAIN_DIR, "runs", rid, "preds.pkl"), "rb") as f:
        saved = pickle.load(f)
    y_orig = fold["test"]["y"].astype(np.float64)
    assert np.array_equal(y_orig, saved["meta"]["y"].astype(np.float64)), f"y mismatch in {rid}"

    re_preds = {}
    re_preds["ridge"] = inv_te_d(bundle["ridge"].predict(Xte_d)).astype(np.float64)
    for tm in TREE_NAMES:
        raw_p = bundle["trees"][tm].predict(Xte_d)
        re_preds[tm] = (inv_te_r(raw_p) if tm.endswith("_residual") else inv_te_d(raw_p)).astype(np.float64)
    assert bundle["best_tree_choice"] == saved["best_tree_choice"], (
        f"best_tree_choice mismatch in {rid}: {bundle['best_tree_choice']} vs {saved['best_tree_choice']}"
    )
    re_preds["best_tree"] = re_preds[bundle["best_tree_choice"]]

    fam_avgs = {}
    for kind in NEURAL_FAMS:
        seed_preds = [
            inv_te_d(predict_torch_from_checkpoint(os.path.join(ndir, f"{kind}_seed{s}.pt"), Xte_s)).astype(np.float64)
            for s in config.SEEDS
        ]
        fam_avgs[kind + "_avg"] = np.mean(seed_preds, axis=0)
        re_preds[kind + "_avg"] = fam_avgs[kind + "_avg"]
    re_preds["equal_neural_ens"] = np.mean([fam_avgs[k + "_avg"] for k in NEURAL_FAMS], axis=0)

    ver_rows = []
    for mname in ("equal_neural_ens", "ridge", "best_tree", "gbm_direct", "xgb_direct", "gbm_residual", "xgb_residual"):
        pr = re_preds[mname]
        ps = saved["test"][mname].astype(np.float64)
        diff = np.abs(pr - ps)
        r_new = E.rmse(pr - y_orig)
        r_old = E.rmse(ps - y_orig)
        dr = float(r_new - r_old)
        ver_rows.append(dict(
            run_id=rid,
            weather_mode=wm,
            horizon=h,
            test_year=spec["test_year"],
            model=mname,
            N_orig_test=int(len(y_orig)),
            max_abs_diff=round(float(np.max(diff)), 6),
            mean_abs_diff=round(float(np.mean(diff)), 6),
            rmse_retrained=round(r_new, 4),
            rmse_saved=round(r_old, 4),
            rmse_diff=round(dr, 5),
            abs_rmse_diff_gt_0_01=bool(abs(dr) > 0.01),
        ))

    with open(ver_p, "w", encoding="utf-8") as f:
        json.dump(ver_rows, f, indent=2)
    ens_row = [r for r in ver_rows if r["model"] == "equal_neural_ens"][0]
    print(
        f"  [{rid}] Verified vs saved preds.pkl: equal_neural_ens max_abs={ens_row['max_abs_diff']:.5f}, "
        f"mean_abs={ens_row['mean_abs_diff']:.5f}, dRMSE={ens_row['rmse_diff']:+.5f} (gt_0.01={ens_row['abs_rmse_diff_gt_0_01']})",
        flush=True,
    )
    del Xtr_d, Xtr_s, Xva_d, Xva_s, Xte_d, Xte_s
    gc.collect()
    return fold, scalers, bundle, ver_rows


def build_city_test_windows(city, st_dict, L, h, test_year, weather_mode, weather_df):
    """Build test windows for one external city strictly in test_year (prior year lookback only)."""
    t_lo = np.datetime64(f"{test_year}-01-01T00", "h")
    t_hi = np.datetime64(f"{test_year + 1}-01-01T00", "h")
    parts = []
    for st_name, st_obj in sorted(st_dict.items()):
        key = f"{city}/{st_name}"
        w = FS.station_windows_final(key, city, st_name, st_obj, L, h, weather_mode=weather_mode, weather_df=weather_df)
        if w is None:
            continue
        sel = (w["times"] >= t_lo) & (w["times"] < t_hi)
        if np.any(sel):
            parts.append({k: a[sel] for k, a in w.items()})
    if not parts:
        return None
    te = _cat(parts)
    assert te["times"].min() >= t_lo and te["times"].max() < t_hi
    return te


def predict_on_city(spec, scalers, bundle, city_te):
    """Load saved weights for spec and predict on city_te without retraining."""
    rid = spec["run_id"]
    ndir = os.path.join(MODELS_DIR, rid, "neural")
    X_flat, X_seq, _, inv_d = extract_features(city_te, scalers, "direct")
    _, _, _, inv_r = extract_features(city_te, scalers, "residual")

    preds = {
        "persistence": city_te["y_curr"].astype(np.float64),
        "ridge": inv_d(bundle["ridge"].predict(X_flat)).astype(np.float64),
    }
    if spec["horizon"] >= 6:
        preds["same_hour_yesterday"] = city_te["same_hour_yesterday"].astype(np.float64)

    bt = bundle["best_tree_choice"]
    raw_bt = bundle["trees"][bt].predict(X_flat)
    preds["best_tree"] = (inv_r(raw_bt) if bt.endswith("_residual") else inv_d(raw_bt)).astype(np.float64)

    fam_preds = []
    for kind in NEURAL_FAMS:
        sp = [
            inv_d(predict_torch_from_checkpoint(os.path.join(ndir, f"{kind}_seed{s}.pt"), X_seq)).astype(np.float64)
            for s in config.SEEDS
        ]
        fam_preds.append(np.mean(sp, axis=0))
    preds["equal_neural_ens"] = np.mean(fam_preds, axis=0)
    return preds


def evaluate_weather_mode(wm, clean_ext, clean_23, clean_24, om_23, om_24, om_ext):
    """Run Steps A-D (and E for openmeteo) completely for one weather_mode and save all CSVs."""
    print(f"\n==================== PHASE: weather_mode = {wm} ====================", flush=True)
    specs = build_config_specs(wm)
    ver_all, city_rows, boot_rows, fail_rows, sp_rows = [], [], [], [], []
    city_pred_cache = {}  # (group, city, test_year, horizon) -> dict(y, block_date, station, times, preds)

    for sp in specs:
        rid = sp["run_id"]
        print(f"\n--- Step A: {rid} ---", flush=True)
        fold, scalers, bundle, ver_rows = train_and_verify_config(sp, clean_23, clean_24, om_23, om_24)
        ver_all.extend(ver_rows)
        train_sp_rng = scalers["train_sp_range"]
        del fold
        gc.collect()

        # Check saved predictions checkpoint per config so resume skips finished forward passes too
        for grp_name, city_map in [("PRIMARY", clean_ext["PRIMARY"]), ("SECONDARY", clean_ext["SECONDARY"])]:
            for city, st_dict in sorted(city_map.items()):
                try:
                    wdf = om_ext if wm == "openmeteo" else None
                    te = build_city_test_windows(city, st_dict, config.LOOKBACK, sp["horizon"], sp["test_year"], wm, wdf)
                    if te is None or len(te["y"]) == 0:
                        fail_rows.append(dict(
                            weather_mode=wm, city_group=grp_name, city=city,
                            test_year=sp["test_year"], horizon=sp["horizon"],
                            reason="zero_valid_test_windows",
                        ))
                        continue

                    pred_pkl = os.path.join(PREDS_DIR, rid, f"{grp_name}_{city}_preds.pkl")
                    os.makedirs(os.path.dirname(pred_pkl), exist_ok=True)
                    if os.path.exists(pred_pkl):
                        with open(pred_pkl, "rb") as f:
                            saved_cp = pickle.load(f)
                        preds = saved_cp["preds"]
                    else:
                        preds = predict_on_city(sp, scalers, bundle, te)
                        with open(pred_pkl, "wb") as f:
                            pickle.dump(dict(
                                meta={k: te[k] for k in ("y", "y_curr", "station", "times", "block_date")},
                                preds=preds,
                            ), f)

                    y = te["y"].astype(np.float64)
                    blocks = te["block_date"]
                    p_rmse = E.rmse(preds["persistence"] - y)
                    p_mae = E.mae(preds["persistence"] - y)
                    p_r2 = CR2.calc_r2(y, preds["persistence"])

                    # Surface pressure check for Step E
                    if wm == "openmeteo":
                        c_sp = te["w_seq"][:, :, SUR_PRES_IDX]
                        c_min, c_max = float(np.nanmin(c_sp)), float(np.nanmax(c_sp))
                        tr_min, tr_max = train_sp_rng
                        out_flag = (c_min < tr_min) or (c_max > tr_max)
                        sp_rows.append(dict(
                            city_group=grp_name, city=city, test_year=sp["test_year"], horizon=sp["horizon"],
                            N_test=int(len(y)),
                            city_sp_min=round(c_min, 2), city_sp_max=round(c_max, 2),
                            train_sp_min=round(tr_min, 2), train_sp_max=round(tr_max, 2),
                            outside_train_sp_range=out_flag,
                        ))

                    pairs = [
                        ("equal_neural_ens", "persistence"),
                        ("equal_neural_ens", "best_tree"),
                        ("ridge", "equal_neural_ens"),
                        ("ridge", "persistence"),
                        ("best_tree", "persistence"),
                    ]
                    if sp["horizon"] >= 6:
                        pairs.insert(1, ("equal_neural_ens", "same_hour_yesterday"))
                    boot = E.paired_block_bootstrap(y, preds, blocks, pairs)
                    n_blk = int(boot[0]["n_blocks"])
                    b_map = {(b["A"], b["B"]): b for b in boot}

                    row = dict(
                        weather_mode=wm,
                        city_group=grp_name,
                        city=city,
                        n_stations_used=int(len(set(te["station"]))),
                        test_year=int(sp["test_year"]),
                        horizon=int(sp["horizon"]),
                        N_test=int(len(y)),
                        n_date_blocks=n_blk,
                        best_tree_choice=bundle["best_tree_choice"],
                        persist_RMSE=round(p_rmse, 3),
                        persist_MAE=round(p_mae, 3),
                        persist_R2=round(p_r2, 4),
                    )
                    if sp["horizon"] >= 6:
                        shy = preds["same_hour_yesterday"]
                        r_sy = E.rmse(shy - y)
                        a_sy = E.mae(shy - y)
                        row["same_hr_yest_RMSE"] = round(r_sy, 3)
                        row["same_hr_yest_MAE"] = round(a_sy, 3)
                        row["same_hr_yest_skill_RMSE"] = round(100.0 * (1.0 - r_sy / p_rmse), 2)
                        row["same_hr_yest_skill_MAE"] = round(100.0 * (1.0 - a_sy / p_mae), 2)
                        row["same_hr_yest_R2"] = round(CR2.calc_r2(y, shy), 4)
                    else:
                        for suf in ("RMSE", "MAE", "skill_RMSE", "skill_MAE", "R2"):
                            row[f"same_hr_yest_{suf}"] = np.nan

                    for mname, pfx in [("equal_neural_ens", "ens"), ("ridge", "ridge"), ("best_tree", "best_tree")]:
                        rm = E.rmse(preds[mname] - y)
                        am = E.mae(preds[mname] - y)
                        row[f"{pfx}_RMSE"] = round(rm, 3)
                        row[f"{pfx}_MAE"] = round(am, 3)
                        row[f"{pfx}_skill_RMSE"] = round(100.0 * (1.0 - rm / p_rmse), 2)
                        row[f"{pfx}_skill_MAE"] = round(100.0 * (1.0 - am / p_mae), 2)
                        row[f"{pfx}_R2"] = round(CR2.calc_r2(y, preds[mname]), 4)

                    # Inline key CI fields + status for Step C & D
                    for (pa, pb), tag in [
                        (("equal_neural_ens", "persistence"), "ens_vs_pers"),
                        (("equal_neural_ens", "best_tree"), "ens_vs_best_tree"),
                        (("ridge", "equal_neural_ens"), "ridge_vs_ens"),
                        (("ridge", "persistence"), "ridge_vs_pers"),
                        (("best_tree", "persistence"), "best_tree_vs_pers"),
                    ]:
                        br = b_map[(pa, pb)]
                        row[f"{tag}_dRMSE_mean"] = round(float(br["dRMSE_mean"]), 3)
                        row[f"{tag}_dRMSE_lo"] = round(float(br["dRMSE_lo"]), 3)
                        row[f"{tag}_dRMSE_hi"] = round(float(br["dRMSE_hi"]), 3)
                        row[f"{tag}_dRMSE_status"] = _ci_status(float(br["dRMSE_lo"]), float(br["dRMSE_hi"]))
                        row[f"{tag}_dMAE_mean"] = round(float(br["dMAE_mean"]), 3)
                        row[f"{tag}_dMAE_lo"] = round(float(br["dMAE_lo"]), 3)
                        row[f"{tag}_dMAE_hi"] = round(float(br["dMAE_hi"]), 3)
                        row[f"{tag}_dMAE_status"] = _ci_status(float(br["dMAE_lo"]), float(br["dMAE_hi"]))

                    if ("equal_neural_ens", "same_hour_yesterday") in b_map:
                        br = b_map[("equal_neural_ens", "same_hour_yesterday")]
                        row["ens_vs_shy_dRMSE_mean"] = round(float(br["dRMSE_mean"]), 3)
                        row["ens_vs_shy_dRMSE_lo"] = round(float(br["dRMSE_lo"]), 3)
                        row["ens_vs_shy_dRMSE_hi"] = round(float(br["dRMSE_hi"]), 3)
                        row["ens_vs_shy_dMAE_mean"] = round(float(br["dMAE_mean"]), 3)
                        row["ens_vs_shy_dMAE_lo"] = round(float(br["dMAE_lo"]), 3)
                        row["ens_vs_shy_dMAE_hi"] = round(float(br["dMAE_hi"]), 3)
                    else:
                        for suf in ("dRMSE_mean", "dRMSE_lo", "dRMSE_hi", "dMAE_mean", "dMAE_lo", "dMAE_hi"):
                            row[f"ens_vs_shy_{suf}"] = np.nan

                    city_rows.append(row)
                    for b in boot:
                        boot_rows.append(dict(
                            weather_mode=wm,
                            city_group=grp_name,
                            city=city,
                            test_year=int(sp["test_year"]),
                            horizon=int(sp["horizon"]),
                            N_test=int(len(y)),
                            n_blocks=n_blk,
                            B_resamples=int(b["B_resamples"]),
                            A=b["A"],
                            B=b["B"],
                            dRMSE_mean=round(float(b["dRMSE_mean"]), 3),
                            dRMSE_lo=round(float(b["dRMSE_lo"]), 3),
                            dRMSE_hi=round(float(b["dRMSE_hi"]), 3),
                            dRMSE_status=_ci_status(float(b["dRMSE_lo"]), float(b["dRMSE_hi"])),
                            dMAE_mean=round(float(b["dMAE_mean"]), 3),
                            dMAE_lo=round(float(b["dMAE_lo"]), 3),
                            dMAE_hi=round(float(b["dMAE_hi"]), 3),
                            dMAE_status=_ci_status(float(b["dMAE_lo"]), float(b["dMAE_hi"])),
                        ))
                    city_pred_cache[(grp_name, city, sp["test_year"], sp["horizon"])] = dict(
                        y=y, block_date=blocks, station=te["station"], times=te["times"],
                        ens=preds["equal_neural_ens"],
                    )
                except Exception as exc:
                    fail_rows.append(dict(
                        weather_mode=wm, city_group=grp_name, city=city,
                        test_year=sp["test_year"], horizon=sp["horizon"],
                        reason=f"exception: {exc}",
                    ))
                    print(f"  [WARN] Failed {rid} on {grp_name}/{city}: {exc}", flush=True)

    df_ver = pd.DataFrame(ver_all)
    df_city = pd.DataFrame(city_rows)
    df_boot = pd.DataFrame(boot_rows)
    df_fail = pd.DataFrame(fail_rows)
    df_sum = summarize_across_cities(df_city, wm)

    tag = wm  # 'none' or 'openmeteo'
    df_ver.to_csv(os.path.join(EXT_DIR, f"STEP_A_VERIFY_{tag}.csv"), index=False)
    df_city.to_csv(os.path.join(EXT_DIR, f"STEP_BC_CITIES_{tag}.csv"), index=False)
    df_boot.to_csv(os.path.join(EXT_DIR, f"STEP_C_BOOT_CIS_{tag}.csv"), index=False)
    df_sum.to_csv(os.path.join(EXT_DIR, f"STEP_D_SUMMARY_{tag}.csv"), index=False)
    if fail_rows:
        df_fail.to_csv(os.path.join(EXT_DIR, f"FAILED_{tag}.csv"), index=False)
    if sp_rows:
        pd.DataFrame(sp_rows).to_csv(os.path.join(EXT_DIR, f"STEP_E_SURFACE_PRESSURE_{tag}.csv"), index=False)

    print(f"\nSaved all {tag} CSVs cleanly -> {EXT_DIR}", flush=True)
    return dict(ver=df_ver, city=df_city, boot=df_boot, sum=df_sum, fail=df_fail, pred_cache=city_pred_cache)


def summarize_across_cities(df_city, weather_mode):
    """Step D: across-city median/min/max skill and CI-direction counts per (group, test_year, horizon)."""
    rows = []
    for (grp, yr, hor), g in df_city.groupby(["city_group", "test_year", "horizon"], sort=True):
        r = dict(
            weather_mode=weather_mode,
            city_group=grp,
            test_year=int(yr),
            horizon=int(hor),
            n_cities=int(len(g)),
            best_tree_choice=str(g["best_tree_choice"].iloc[0]),
        )
        for pfx in ("ens", "ridge", "best_tree"):
            s = g[f"{pfx}_skill_RMSE"]
            r[f"{pfx}_skill_RMSE_med"] = round(float(np.median(s)), 2)
            r[f"{pfx}_skill_RMSE_min"] = round(float(np.min(s)), 2)
            r[f"{pfx}_skill_RMSE_max"] = round(float(np.max(s)), 2)
        for ctag in ("ens_vs_pers", "ridge_vs_pers", "best_tree_vs_pers", "ridge_vs_ens"):
            col_r = f"{ctag}_dRMSE_status"
            col_m = f"{ctag}_dMAE_status"
            r[f"{ctag}_dRMSE_below"] = int((g[col_r] == "entirely_below_zero").sum())
            r[f"{ctag}_dRMSE_incl"] = int((g[col_r] == "includes_zero").sum())
            r[f"{ctag}_dRMSE_above"] = int((g[col_r] == "entirely_above_zero").sum())
            r[f"{ctag}_dMAE_below"] = int((g[col_m] == "entirely_below_zero").sum())
            r[f"{ctag}_dMAE_incl"] = int((g[col_m] == "includes_zero").sum())
            r[f"{ctag}_dMAE_above"] = int((g[col_m] == "entirely_above_zero").sum())
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["city_group", "test_year", "horizon"]).reset_index(drop=True)


def compute_om_vs_none_table(none_cache, om_cache):
    """Compute matched Open-Meteo vs None DATE-block 95% CIs per external city, year, horizon."""
    rows = []
    for key in sorted(none_cache):
        if key not in om_cache:
            continue
        grp, city, yr, hor = key
        dn, do = none_cache[key], om_cache[key]
        assert np.array_equal(dn["y"], do["y"]), f"y mismatch {key}"
        assert np.array_equal(dn["station"], do["station"]), f"station mismatch {key}"
        assert np.array_equal(dn["times"], do["times"]), f"times mismatch {key}"
        assert np.array_equal(dn["block_date"], do["block_date"]), f"block_date mismatch {key}"
        brow = E.paired_block_bootstrap(
            dn["y"], {"openmeteo_ens": do["ens"], "none_ens": dn["ens"]},
            dn["block_date"], [("openmeteo_ens", "none_ens")],
        )[0]
        rows.append(dict(
            city_group=grp, city=city, test_year=int(yr), horizon=int(hor),
            N_test=int(len(dn["y"])), n_blocks=int(brow["n_blocks"]), B_resamples=int(brow["B_resamples"]),
            dRMSE_mean=round(float(brow["dRMSE_mean"]), 3),
            dRMSE_lo=round(float(brow["dRMSE_lo"]), 3),
            dRMSE_hi=round(float(brow["dRMSE_hi"]), 3),
            dRMSE_status=_ci_status(float(brow["dRMSE_lo"]), float(brow["dRMSE_hi"])),
            dMAE_mean=round(float(brow["dMAE_mean"]), 3),
            dMAE_lo=round(float(brow["dMAE_lo"]), 3),
            dMAE_hi=round(float(brow["dMAE_hi"]), 3),
            dMAE_status=_ci_status(float(brow["dMAE_lo"]), float(brow["dMAE_hi"])),
        ))
    return pd.DataFrame(rows)


def load_pred_cache_from_disk(weather_mode):
    """Reconstruct pred_cache from saved results_ext_cities/preds/<run_id>/ bundles."""
    cache = {}
    for sp in build_config_specs(weather_mode):
        rid = sp["run_id"]
        pdir = os.path.join(PREDS_DIR, rid)
        if not os.path.isdir(pdir):
            continue
        for fname in sorted(os.listdir(pdir)):
            if not fname.endswith("_preds.pkl"):
                continue
            grp, rest = fname[:-len("_preds.pkl")].split("_", 1)
            with open(os.path.join(pdir, fname), "rb") as f:
                b = pickle.load(f)
            m = b["meta"]
            cache[(grp, rest, sp["test_year"], sp["horizon"])] = dict(
                y=m["y"].astype(np.float64),
                block_date=m["block_date"],
                station=m["station"],
                times=m["times"],
                ens=b["preds"]["equal_neural_ens"].astype(np.float64),
            )
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "openmeteo", "all"], default="all")
    args = ap.parse_args()

    os.makedirs(EXT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    print("Note: P2_2024 in results_final/ exists only at h=1 and h=24; 2024 h=6 and h=12 are skipped.", flush=True)
    clean_ext = prepare_ext_city_clean_cache()
    with open(os.path.join(MAIN_DIR, "clean_20_2017_2023_v3.pkl"), "rb") as f:
        clean_23 = pickle.load(f)
    with open(os.path.join(MAIN_DIR, "clean_20_2017_2024_v3.pkl"), "rb") as f:
        clean_24 = pickle.load(f)

    om_23 = pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2023.parquet"))
    om_24 = pd.read_parquet(os.path.join(ROOT, "raw_cpcb", "weather_openmeteo_2017_2024.parquet"))
    om_ext = pd.read_parquet(os.path.join(CACHE_DIR, "weather_openmeteo_ext_cities_2022_2024.parquet"))

    t0 = time.time()
    res_none = None
    if args.mode in ("none", "all"):
        res_none = evaluate_weather_mode("none", clean_ext, clean_23, clean_24, om_23, om_24, om_ext)

    res_om = None
    if args.mode in ("openmeteo", "all"):
        res_om = evaluate_weather_mode("openmeteo", clean_ext, clean_23, clean_24, om_23, om_24, om_ext)

    if res_om is not None:
        none_cache = res_none["pred_cache"] if res_none is not None else load_pred_cache_from_disk("none")
        if none_cache:
            df_om_vs_none = compute_om_vs_none_table(none_cache, res_om["pred_cache"])
            df_om_vs_none.to_csv(os.path.join(EXT_DIR, "STEP_E_OM_VS_NONE_CIS.csv"), index=False)
        ver_none_p = os.path.join(EXT_DIR, "STEP_A_VERIFY_none.csv")
        sum_none_p = os.path.join(EXT_DIR, "STEP_D_SUMMARY_none.csv")
        ver_none = res_none["ver"] if res_none is not None else (pd.read_csv(ver_none_p) if os.path.exists(ver_none_p) else None)
        sum_none = res_none["sum"] if res_none is not None else (pd.read_csv(sum_none_p) if os.path.exists(sum_none_p) else None)
        if ver_none is not None:
            pd.concat([ver_none, res_om["ver"]], ignore_index=True).to_csv(os.path.join(EXT_DIR, "STEP_A_VERIFY_ALL.csv"), index=False)
        if sum_none is not None:
            pd.concat([sum_none, res_om["sum"]], ignore_index=True).to_csv(os.path.join(EXT_DIR, "STEP_D_SUMMARY_ALL.csv"), index=False)

    print(f"\nEXTERNAL VALIDATION FINISHED CLEANLY in {(time.time()-t0)/60:.1f} min -> {EXT_DIR}", flush=True)


if __name__ == "__main__":
    main()

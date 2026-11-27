"""Step runner.  Usage:  python run.py step0|step1|step2|step3|step4|step5 [options]
Each step stops and prints what it did. Nothing starts the next step for you."""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

import config
import data as D
import evaluate as E
import models as M
import qc

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)


def out(args, name):
    os.makedirs(args.out, exist_ok=True)
    return os.path.join(args.out, name)


def step0(args):
    files = qc.list_station_files(args.data)
    rows = []
    for city, name, p in files:
        idx, vals, slots, dups = qc.parse_opencity(p)
        fin = np.isfinite(vals)
        t_ok = idx[fin]
        rows.append(dict(
            city=city, station=name, path=os.path.abspath(p), sha256=qc.sha256(p),
            hourly_slots=slots, nonnull_raw=int(fin.sum()), duplicate_ts=dups,
            start_date=t_ok[0].strftime("%Y-%m-%d %H:%M") if len(t_ok) else "",
            end_date=t_ok[-1].strftime("%Y-%m-%d %H:%M") if len(t_ok) else "",
        ))
    df = pd.DataFrame(rows)
    print(df.drop(columns=["path"]).to_string(index=False))
    print(f"\n{len(df)} CSV files, {df.city.nunique()} cities. Layout is data/<City>/<Station>.csv only.")
    d = df[df.sha256.duplicated(keep=False)]
    print("Files with identical content (same SHA-256):", "none" if d.empty else d[["city", "station"]].values.tolist())
    df.to_csv(out(args, "step0_files.csv"), index=False)


def step1(args):
    print(f"QC spec: upper_bound={config.CAP} (kept if <= cap; >cap -> NaN) | identical runs > {config.MAX_STUCK}h -> NaN |"
          f" interpolate gaps <= {config.MAX_GAP}h | drop span < {config.MIN_SPAN_YEARS}y")
    clean, reps, overs, stats = {}, [], [], []
    for city, name, p in qc.list_station_files(args.data):
        st, rep, over = qc.qc_station(p)
        reps.append(dict(city=city, station=name, **rep))
        overs += [dict(city=city, station=name, timestamp=t, value=v) for t, v in over]
        if st is not None:
            clean.setdefault(city, {})[name] = st
            x = st["value"][np.isfinite(st["value"])]
            stats.append(dict(city=city, station=name, N=len(x), mean=x.mean(), median=np.median(x),
                              p90=np.percentile(x, 90), p99=np.percentile(x, 99), max=x.max(),
                              ge90=(x >= 90).mean() * 100, ge150=(x >= 150).mean() * 100,
                              ge250=(x >= 250).mean() * 100))
    rep = pd.DataFrame(reps)
    print("\nPer-station QC report:\n", rep.to_string(index=False))
    print("\nValues above the cap (removed):",
          "none" if not overs else "\n" + pd.DataFrame(overs).to_string(index=False))
    print(f"\nDistribution of cleaned {config.POLLUTANT} (percent columns = share of hours):\n",
          pd.DataFrame(stats).round(2).to_string(index=False))
    rep.to_csv(out(args, "step1_qc_report.csv"), index=False)
    pd.DataFrame(overs).to_csv(out(args, "step1_above_cap.csv"), index=False)
    pd.DataFrame(stats).to_csv(out(args, "step1_distribution.csv"), index=False)
    with open(out(args, "clean.pkl"), "wb") as f:
        pickle.dump(clean, f)
    print("\nSaved cleaned data. Every fold uses it.")


def step2(args):
    cfgd = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    print(json.dumps(cfgd, indent=2))
    print("torch available:", M.torch_available())
    if M.torch_available():
        import torch
        print("cuda available:", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    print("AMP:", "ON" if config.AMP else "OFF (frozen)")
    with open(out(args, "config.json"), "w") as f:
        json.dump(cfgd, f, indent=2)


def _metrics(y, p):
    e = p - y
    return E.rmse(e), E.mae(e)


def step3(args):
    if args.smoke:
        config.SEEDS, config.EPOCHS = [0, 1], 2
    config.HORIZON = args.horizon
    with open(out(args, "clean.pkl"), "rb") as f:
        clean = pickle.load(f)
    fold = D.build_fold(clean, args.heldout, config.LOOKBACK, args.horizon)
    tr, va, te, mu, sd = fold["train"], fold["val"], fold["test"], fold["mu"], fold["sd"]
    if args.smoke:
        k = max(1, len(tr["y"]) // 20000)
        tr = {a: v[::k] for a, v in tr.items()}
    print(f"Held out: {args.heldout} | train {len(tr['y']):,} val {len(va['y']):,} test {len(te['y']):,} windows"
          f" | train stations {len(set(tr['station']))}, test stations {sorted(str(s) for s in set(te["station"]))}")
    print("Leakage assertions passed (no test station in train/val; no held-out city in train/val).")

    Xtr_f, Xtr_s, ytr = D.features(tr, mu, sd)
    Xva_f, Xva_s, yva = D.features(va, mu, sd)
    Xte_f, Xte_s, yte = D.features(te, mu, sd)
    back = lambda p: p * sd + mu  # noqa: E731

    val, test, info = {}, {}, {}
    val["persistence"], test["persistence"] = va["y_curr"].astype(float), te["y_curr"].astype(float)
    pv, pt = M.fit_linear(Xtr_f, ytr, Xva_f, Xte_f)
    val["linear"], test["linear"] = back(pv), back(pt)
    pv, pt = M.fit_linear(Xtr_f, ytr, Xva_f, Xte_f, ridge=True)
    val["ridge"], test["ridge"] = back(pv), back(pt)
    pv, pt, name = M.fit_gbm(Xtr_f, ytr, Xva_f, yva, Xte_f)
    val["gbm"], test["gbm"], info["gbm_impl"] = back(pv), back(pt), name
    r = M.fit_xgb(Xtr_f, ytr, Xva_f, yva, Xte_f)
    if r is not None:
        val["xgboost"], test["xgboost"] = back(r[0]), back(r[1])
    else:
        print("xgboost not installed: skipped.")
    print("GBM implementation:", info["gbm_impl"])

    seed_rows, member_val, member_test = [], {}, {}
    if M.torch_available() and not args.no_neural:
        for kind in config.NEURAL:
            sv, st = [], []
            for s in config.SEEDS:
                pv, pt = M.train_neural(kind, Xtr_s, ytr, Xva_s, yva, Xte_s, s)
                pv, pt = back(pv), back(pt)
                sv.append(pv), st.append(pt)
                r_, a_ = _metrics(te["y"], pt)
                seed_rows.append(dict(model=kind, seed=s, RMSE=r_, MAE=a_))
            val[kind + "_avg"], test[kind + "_avg"] = np.mean(sv, 0), np.mean(st, 0)
            for s_i, s in enumerate(config.SEEDS):
                test[f"{kind}_seed{s}"] = st[s_i]
    else:
        print("Neural models skipped (torch not installed or --no-neural).")

    member = [m for m in val if m not in ("persistence", "linear") and "_seed" not in m]
    w, sv_, st_ = M.nnls_stack({m: val[m] for m in member}, va["y"], {m: test[m] for m in member})
    val["stack"], test["stack"] = sv_, st_
    print("NNLS stack weights (fit on validation only):", {k: round(float(v), 3) for k, v in w.items()})

    rows = []
    for m in test:
        if "_seed" in m:
            continue
        r_, a_ = _metrics(te["y"], test[m])
        rv, _ = _metrics(va["y"], val[m]) if m in val else (np.nan, 0)
        rows.append(dict(model=m, val_RMSE=rv, test_RMSE=r_, test_MAE=a_))
    res = pd.DataFrame(rows)
    pr, pa = res.loc[res.model == "persistence", ["test_RMSE", "test_MAE"]].iloc[0]
    res["skill_RMSE_pct"] = 100 * (1 - res.test_RMSE / pr)
    res["skill_MAE_pct"] = 100 * (1 - res.test_MAE / pa)
    print("\nTest results (raw units; skill vs persistence, negatives kept):\n", res.round(3).to_string(index=False))

    if seed_rows:
        sr = pd.DataFrame(seed_rows)
        agg = sr.groupby("model").agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                                      MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")).round(4)
        print("\nNeural seed mean/std (test, per single seed):\n", agg)
        best = agg.RMSE_mean.idxmin()
        srmse = res.loc[res.model == "stack", "test_RMSE"].iloc[0]
        gap = agg.loc[best, "RMSE_mean"] - srmse
        print(f"\nBest neural by seed-mean RMSE: {best} ({agg.loc[best,'RMSE_mean']:.3f} +/- {agg.loc[best,'RMSE_std']:.3f})."
              f" Stack RMSE {srmse:.3f}; gap {gap:.3f} vs seed std {agg.loc[best,'RMSE_std']:.3f} ->"
              f" {'gap exceeds seed std' if gap > agg.loc[best,'RMSE_std'] else 'gap does NOT exceed seed std'}.")
        sr.to_csv(out(args, "step3_seed_rows.csv"), index=False)
    res.to_csv(out(args, f"step3_{args.heldout}_h{args.horizon}_results.csv"), index=False)
    meta = {k: te[k] for k in ("y", "y_curr", "rise3", "cens_any", "month", "block", "station")}
    with open(out(args, f"fold_{args.heldout}_h{args.horizon}.pkl"), "wb") as f:
        pickle.dump(dict(test=test, val=val, meta=meta, results=res, val_y=va["y"]), f)


def _load_fold(args):
    with open(out(args, f"fold_{args.heldout}_h{args.horizon}.pkl"), "rb") as f:
        return pickle.load(f)


def step4(args):
    F = _load_fold(args)
    res, test, meta = F["results"].set_index("model"), F["test"], F["meta"]
    test = {k: v for k, v in test.items() if "_seed" not in k}
    neural = [m for m in test if m.endswith("_avg")]
    trees = [m for m in ("gbm", "xgboost") if m in test]
    single = [m for m in test if m not in ("persistence", "stack")]
    pick = lambda names: min(names, key=lambda m: res.loc[m, "val_RMSE"])  # noqa: E731  (chosen on validation)
    bn = pick(neural) if neural else None
    bt, bs = pick(trees), pick(single)
    pairs = [("stack", bt), ("stack", "persistence"), (bs, "persistence")]
    if bn:
        pairs.insert(0, ("stack", bn))
    print(f"Best chosen on validation RMSE: neural={bn}, tree={bt}, single={bs}")
    rows = E.paired_block_bootstrap(meta["y"], test, meta["block"], pairs)
    df = pd.DataFrame(rows)
    print(f"\nPaired day-block bootstrap ({config.BOOT_RESAMPLES} resamples, {df.n_blocks.iloc[0]} blocks). "
          "Differences A-B; negative = A better. 95% CI only:\n", df.round(4).to_string(index=False))
    df.to_csv(out(args, f"step4_bootstrap_{args.heldout}_h{args.horizon}.csv"), index=False)


def step5(args):
    F = _load_fold(args)
    test = {k: v for k, v in F["test"].items() if "_seed" not in k}
    tab = pd.DataFrame(E.regime_table(F["meta"], F["meta"]["y"], test))
    print("Regimes defined from time-t information only. Skill vs persistence, negatives kept.\n")
    print(tab.round(3).to_string(index=False))
    tab.to_csv(out(args, f"step5_regimes_{args.heldout}_h{args.horizon}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["step0", "step1", "step2", "step3", "step4", "step5"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--heldout", default="Delhi")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the code, not for results")
    ap.add_argument("--no-neural", action="store_true")
    a = ap.parse_args()
    {"step0": step0, "step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5}[a.step](a)

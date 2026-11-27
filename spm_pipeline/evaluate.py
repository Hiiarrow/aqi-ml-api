"""Metrics, paired day-block bootstrap, and time-t regime analysis."""
import numpy as np

import config


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def mae(e):
    return float(np.mean(np.abs(e)))


def paired_block_bootstrap(y, preds, blocks, pairs, B=None, seed=None):
    """Resample whole (station, day) blocks with replacement; same resample for all models.
    pairs: list of (name_a, name_b). Returns rows with mean and 95% CI of RMSE_a-RMSE_b and MAE_a-MAE_b."""
    B = B or config.BOOT_RESAMPLES
    rng = np.random.default_rng(config.BOOT_SEED if seed is None else seed)
    uniq, inv = np.unique(blocks, return_inverse=True)
    nb = len(uniq)
    cnt = np.bincount(inv, minlength=nb).astype(float)
    sse, sae = {}, {}
    for m, p in preds.items():
        e = p - y
        sse[m] = np.bincount(inv, weights=e ** 2, minlength=nb)
        sae[m] = np.bincount(inv, weights=np.abs(e), minlength=nb)
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=B).astype(float)
    denom = W @ cnt
    boot_rmse = {m: np.sqrt((W @ sse[m]) / denom) for m in preds}
    boot_mae = {m: (W @ sae[m]) / denom for m in preds}
    rows = []
    for a, b in pairs:
        dr, dm = boot_rmse[a] - boot_rmse[b], boot_mae[a] - boot_mae[b]
        rows.append(dict(A=a, B=b, n_blocks=nb, B_resamples=B,
                         dRMSE_mean=dr.mean(), dRMSE_lo=np.percentile(dr, 2.5), dRMSE_hi=np.percentile(dr, 97.5),
                         dMAE_mean=dm.mean(), dMAE_lo=np.percentile(dm, 2.5), dMAE_hi=np.percentile(dm, 97.5)))
    return rows


def regime_masks(t):
    m = {}
    for th in config.THRESHOLDS:
        m[f"y_t >= {th:g}"] = t["y_curr"] >= th
    m[f"onset (y_t - y_t-3 > {config.SPIKE_ONSET_RISE:g})"] = t["rise3"] > config.SPIKE_ONSET_RISE
    w = np.isin(t["month"], config.WINTER_MONTHS)
    m["winter (Nov-Feb)"] = w
    m["other months (Mar-Oct)"] = ~w
    m["all"] = np.ones(len(t["y"]), bool)
    return m


def regime_table(t, y, preds, persist_name="persistence"):
    """Per regime, per model: N, RMSE, MAE, skill vs persistence on both metrics (negatives kept)."""
    rows = []
    masks = regime_masks(t)
    for rname, sel in masks.items():
        if sel.sum() == 0:
            continue
        pe = preds[persist_name][sel] - y[sel]
        p_r, p_m = rmse(pe), mae(pe)
        for m, p in preds.items():
            e = p[sel] - y[sel]
            r, a = rmse(e), mae(e)
            rows.append(dict(regime=rname, model=m, N=int(sel.sum()),
                             RMSE=r, MAE=a,
                             skill_RMSE_pct=100 * (1 - r / p_r), skill_MAE_pct=100 * (1 - a / p_m)))
    return rows

"""Unit tests for supplementary R^2 computation and paired block bootstrap."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import compute_r2 as CR2  # noqa: E402


def test_calc_r2_identities():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(CR2.calc_r2(y, y) - 1.0) < 1e-12
    assert abs(CR2.calc_r2(y, np.full_like(y, y.mean())) - 0.0) < 1e-12
    worse = np.array([40.0, 10.0, 30.0, 10.0])
    expected = 1.0 - np.sum((worse - y) ** 2) / np.sum((y - y.mean()) ** 2)
    assert abs(CR2.calc_r2(y, worse) - expected) < 1e-12


def test_paired_block_bootstrap_r2_matches_naive():
    rng_data = np.random.default_rng(7)
    blocks = np.repeat(np.array(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]), 12)
    y = 50.0 + rng_data.normal(0.0, 15.0, len(blocks))
    pred_b = y + rng_data.normal(0.0, 8.0, len(blocks))
    pred_a = y + rng_data.normal(0.0, 5.0, len(blocks))

    B, seed = 250, 11
    fast = CR2.paired_block_bootstrap_r2(y, pred_a, pred_b, blocks, B=B, seed=seed)

    # Brute-force row-level bootstrap using exact same multinomial weights
    uniq = np.unique(blocks)
    nb = len(uniq)
    idx_by_blk = [np.where(blocks == b)[0] for b in uniq]
    W = np.random.default_rng(seed).multinomial(nb, np.full(nb, 1.0 / nb), size=B)
    naive_dr2 = np.empty(B, dtype=np.float64)
    for b_idx in range(B):
        sel = np.concatenate([idx_by_blk[k] for k in range(nb) for _ in range(W[b_idx, k])])
        naive_dr2[b_idx] = CR2.calc_r2(y[sel], pred_a[sel]) - CR2.calc_r2(y[sel], pred_b[sel])

    assert abs(fast["dR2_boot_mean"] - float(naive_dr2.mean())) < 1e-12
    assert abs(fast["dR2_boot_lo"] - float(np.percentile(naive_dr2, 2.5))) < 1e-12
    assert abs(fast["dR2_boot_hi"] - float(np.percentile(naive_dr2, 97.5))) < 1e-12


if __name__ == "__main__":
    test_calc_r2_identities()
    test_paired_block_bootstrap_r2_matches_naive()
    print("ALL R2 UNIT TESTS PASSED")

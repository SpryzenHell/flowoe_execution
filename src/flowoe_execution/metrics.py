from __future__ import annotations
import numpy as np


def bootstrap_mean_ci(values, seed=7, n_boot=2000, alpha=0.05):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("values must be a non-empty finite 1-D array")
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("n_boot must be positive and alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_block_bootstrap_ci(baseline, policy, seed=7, n_boot=2000, alpha=0.05, block_size=5):
    """CI for baseline-policy improvement, resampling paired consecutive windows."""
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(policy, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(a) != len(b):
        raise ValueError("baseline and policy must be non-empty 1-D arrays of equal length")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("baseline and policy values must be finite")
    if n_boot < 1 or not 0 < alpha < 1 or block_size < 1:
        raise ValueError("n_boot/block_size must be positive and alpha must be in (0, 1)")
    diff = a - b
    n = len(diff)
    size = min(int(block_size), n)
    blocks = (n + size - 1) // size
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, blocks))
    offsets = np.arange(size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    sample = diff[indices.reshape(n_boot, -1)[:, :n]]
    means = sample.mean(axis=1)
    return float(diff.mean()), (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def improvement_bps(baseline_slippage_bps, policy_slippage_bps):
    return float(baseline_slippage_bps - policy_slippage_bps)


def percentile_ms(values, q=99):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all() or not 0 <= q <= 100:
        raise ValueError("values must be non-empty and finite, and q must be in [0, 100]")
    return float(np.percentile(x, q))


def summary_stats(values, seed=7, n_boot=2000):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("values must be non-empty and finite")
    mean, ci = bootstrap_mean_ci(x, seed=seed, n_boot=n_boot)
    return {
        "mean": mean,
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "ci95": ci,
    }

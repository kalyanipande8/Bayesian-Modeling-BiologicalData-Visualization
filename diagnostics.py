"""
Honest, hand-written MCMC convergence diagnostics.

No PyMC/ArviZ is used (see README for why), so this implements from scratch:

- Effective sample size (ESS) via the initial-monotone-sequence estimator
  applied to the autocorrelation function (Geyer 1992), the same method
  ArviZ's `ess` is based on.
- Gelman-Rubin R-hat, computed from >=2 independent chains, using the
  standard between/within-chain variance formula.
- A simple trace-stability check (comparing the mean of the first half vs.
  second half of a chain, post burn-in).
"""
from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - np.mean(x)
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / n / var
    return acf


def effective_sample_size(x: np.ndarray) -> float:
    """Geyer's initial monotone sequence ESS estimator."""
    n = len(x)
    max_lag = min(n - 1, 1000)
    acf = autocorrelation(x, max_lag)

    # Pair up lags (2k, 2k+1); sum of pairs must be positive and monotone
    # decreasing (Geyer 1992 initial positive / monotone sequence).
    tau = 1.0
    k = 1
    prev_pair_sum = np.inf
    while 2 * k + 1 <= max_lag:
        pair_sum = acf[2 * k] + acf[2 * k + 1]
        if pair_sum < 0:
            break
        pair_sum = min(pair_sum, prev_pair_sum)  # enforce monotonicity
        tau += 2 * pair_sum
        prev_pair_sum = pair_sum
        k += 1
    ess = n / tau
    return float(max(1.0, min(ess, n)))


def gelman_rubin(chains: np.ndarray) -> float:
    """
    R-hat for M chains of length N: chains has shape (M, N).
    Standard Gelman & Rubin (1992) potential scale reduction factor.
    """
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    W = chain_vars.mean()
    grand_mean = chain_means.mean()
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)

    var_hat = (n - 1) / n * W + B / n
    if W <= 0:
        return float("nan")
    r_hat = np.sqrt(var_hat / W)
    return float(r_hat)


def trace_half_stability(x: np.ndarray) -> float:
    """Relative difference between first-half and second-half chain means.

    Returns |mean_2 - mean_1| / std(x); small values indicate a stable,
    well-mixed trace with no residual drift.
    """
    n = len(x)
    h = n // 2
    mean1, mean2 = np.mean(x[:h]), np.mean(x[h:])
    sd = np.std(x)
    if sd == 0:
        return 0.0
    return float(abs(mean2 - mean1) / sd)

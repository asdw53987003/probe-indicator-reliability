# -*- coding: utf-8 -*-
"""Site lists: ranking reliability, the expected overlap of Eq. (5), observed overlap and capture."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import multivariate_normal, norm


def ranking_reliability(x: np.ndarray, err_var: np.ndarray) -> float:
    """One minus the mean error variance over the between-unit variance of the degraded ranking quantity."""
    x = np.asarray(x, float)
    err_var = np.asarray(err_var, float)
    ok = np.isfinite(x) & np.isfinite(err_var)
    if ok.sum() < 3:
        return np.nan
    v = np.var(x[ok], ddof=1)
    return float(1.0 - np.mean(err_var[ok]) / v) if v > 0 else np.nan


def predicted_overlap(lam_rank: float, q: float) -> float:
    """Eq. (5): expected share of a top-q list kept when error-free and degraded quantities are bivariate normal with
    correlation sqrt(lam_rank), O = Phi_2(z_q, z_q; rho) / q with z_q = Phi^-1(1 - q)."""
    if not np.isfinite(lam_rank):
        return np.nan
    rho = float(np.sqrt(np.clip(lam_rank, 0.0, 0.999999)))
    z = norm.ppf(1.0 - q)
    both_below = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([z, z])
    return float((1.0 - 2.0 * norm.cdf(z) + both_below) / q)


def list_size(n_units: int, top_share: float) -> int:
    return max(1, int(round(top_share * n_units)))


def top_set(values: np.ndarray, k: int, index: Iterable = None) -> set:
    """Positions (or labels from `index`) of the k largest finite values."""
    v = np.asarray(values, float)
    v = np.where(np.isfinite(v), v, -np.inf)
    pos = np.argsort(-v, kind='stable')[:k]
    if index is None:
        return set(int(i) for i in pos)
    idx = list(index)
    return set(idx[i] for i in pos)


def list_overlap(reference: np.ndarray, degraded: np.ndarray, k: int) -> float:
    """Share of the units on the reference top-k list that remain on the degraded top-k list."""
    return len(top_set(reference, k) & top_set(degraded, k)) / float(k)


def capture(listed: Iterable[int], expected: np.ndarray, k: int) -> float:
    """Expected crashes of the listed units divided by those of the ideal list (the k highest expected counts)."""
    e = np.asarray(expected, float)
    ideal = np.sort(e)[::-1][:k].sum()
    return float(e[list(listed)].sum() / ideal) if ideal > 0 else np.nan

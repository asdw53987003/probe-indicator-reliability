# -*- coding: utf-8 -*-
"""Data requirements, Eq. (4): reliability rises with vehicle-equivalents approximately as lambda = n / (n + k)."""
from __future__ import annotations

import numpy as np

MIN_VEQ_NBAR = 10.0             # units entering the volume summary need at least this many vehicle-equivalents


def volume_summary(veq: np.ndarray, weights: np.ndarray, min_veq: float = MIN_VEQ_NBAR) -> float:
    """n-bar: harmonic mean of the vehicle-equivalents of units with at least `min_veq`, weighted by working weights."""
    veq = np.asarray(veq, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(veq) & (veq >= min_veq) & np.isfinite(w)
    if not ok.any():
        return np.nan
    return float(np.sum(w[ok]) / np.sum(w[ok] / veq[ok]))


def k_hat(nbar: float, lam: float) -> float:
    """Volume at reliability 0.5, k = n-bar (1 - lambda) / lambda."""
    return float(nbar * (1.0 - lam) / lam) if lam > 0 else np.inf


def n_required(k: float, lam_req: float = 0.9) -> float:
    """Vehicle-equivalents needed for reliability lam_req, n_req = k lam_req / (1 - lam_req)."""
    return float(k * lam_req / (1.0 - lam_req))


def reliability_at(n: np.ndarray, k: float) -> np.ndarray:
    """Reliability expected at n vehicle-equivalents, lambda = n / (n + k)."""
    n = np.asarray(n, float)
    return n / (n + k)


def weeks_needed(n_req: float, nbar: float, period_weeks: float) -> float:
    """Observation period that supplies n_req at the same penetration, given n-bar over `period_weeks`."""
    return float(n_req / (nbar / float(period_weeks)))

# -*- coding: utf-8 -*-
"""Checks of distortions shared by all probes of a unit (Algorithm A.1, step 9).

A coarser sampling interval rescales acceleration-based indicators in both halves alike, so the split halves cannot
reveal it. Regressing the degraded indicator on the 1 Hz indicator in a validation subset of units with full 1 Hz data
gives the calibration slope gamma_1; the coefficient of the indicator grows by 1 / gamma_1 even when it is precise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

NOT_ESTIMABLE_SLOPE = 0.1       # an indicator is treated as not estimable when gamma_1 < 0.1


@dataclass
class Calibration:
    gamma0: float
    gamma1: float
    se_gamma1: float
    p_identity: float               # joint test of intercept 0 and slope 1
    residual_var: float
    sigma2_eps: float               # residual variance not explained by sampling error (0 unless significant)
    p_excess: float
    n_units: int

    @property
    def estimable(self) -> bool:
        return self.gamma1 >= NOT_ESTIMABLE_SLOPE

    def rejects_identity(self, level: float = 0.05) -> bool:
        return self.p_identity < level

    def rescale(self, x):
        """Degraded indicator on the 1 Hz scale, (x - gamma_0) / gamma_1."""
        return (np.asarray(x, float) - self.gamma0) / self.gamma1


def calibration_slope(degraded: pd.Series, reference_1hz: pd.Series, split_half_var: Optional[pd.Series] = None,
                      weights: Optional[pd.Series] = None, level: float = 0.05) -> Calibration:
    """Regress the degraded on the 1 Hz indicator over the validation units (the common index of both series).

    If `split_half_var` (the Eq. (2) variances of the degraded indicator) is given, the residual variance is compared
    with their mean: the excess is kept as sigma2_eps (Eq. (6)) only when the residual variance exceeds it at `level`."""
    d = pd.concat({'x': reference_1hz, 'y': degraded}, axis=1).dropna()
    w = np.ones(len(d)) if weights is None else weights.reindex(d.index).fillna(0.0).to_numpy(float)
    x, y = d['x'].to_numpy(float), d['y'].to_numpy(float)
    n = len(d)
    if n < 3:
        raise ValueError('the validation subset needs at least three units with both indicators')
    A = np.column_stack([np.ones(n), x])
    sw = np.sqrt(w)
    g, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    r = y - A @ g
    dof = n - 2
    s2 = float(np.sum(w * r ** 2) / np.sum(w) * n / dof)
    cov = s2 * np.linalg.inv((A * w[:, None]).T @ A / np.mean(w))
    dlt = g - np.array([0.0, 1.0])
    F = float(dlt @ np.linalg.solve(cov, dlt) / 2.0)
    p_id = float(stats.f.sf(F, 2, dof))
    s2eps, p_ex = 0.0, np.nan
    if split_half_var is not None:
        sh = float(np.nanmean(split_half_var.reindex(d.index).to_numpy(float)))
        stat = dof * s2 / sh if sh > 0 else np.inf
        p_ex = float(stats.chi2.sf(stat, dof))
        if p_ex < level:
            s2eps = max(s2 - sh, 0.0)
    return Calibration(gamma0=float(g[0]), gamma1=float(g[1]), se_gamma1=float(np.sqrt(cov[1, 1])), p_identity=p_id,
                       residual_var=s2, sigma2_eps=s2eps, p_excess=p_ex, n_units=n)


def slope_heterogeneity(degraded: pd.Series, reference_1hz: pd.Series, groups: pd.Series) -> dict:
    """F-test of whether the calibration slope differs between groups of units (e.g. routes or unit types).

    Compares y = a_g + b x with y = a_g + b_g x over the validation units; returns the slopes by group and the p-value."""
    d = pd.concat({'x': reference_1hz, 'y': degraded, 'g': groups}, axis=1).dropna()
    gi, gl = pd.factorize(d['g'])
    G = len(gl)
    x, y = d['x'].to_numpy(float), d['y'].to_numpy(float)
    Dg = np.eye(G)[gi]
    A0 = np.column_stack([Dg, x])
    A1 = np.column_stack([Dg, Dg * x[:, None]])
    rss0 = float(np.sum((y - A0 @ np.linalg.lstsq(A0, y, rcond=None)[0]) ** 2))
    b1 = np.linalg.lstsq(A1, y, rcond=None)[0]
    rss1 = float(np.sum((y - A1 @ b1) ** 2))
    df1, df2 = G - 1, len(d) - 2 * G
    p = float(stats.f.sf(((rss0 - rss1) / df1) / (rss1 / df2), df1, df2)) if df1 > 0 and df2 > 0 else np.nan
    return dict(slopes=pd.Series(b1[G:], index=gl), p_value=p, n_units=len(d))

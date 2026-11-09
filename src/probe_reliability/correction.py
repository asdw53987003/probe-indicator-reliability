# -*- coding: utf-8 -*-
"""Correction of attenuated coefficients where a longer observation period cannot be obtained (Algorithm A.1, step 7).

Two parts are provided:
- `unit_error_variances`, Eq. (6): unit-specific error variances on the 1 Hz scale, the input of a latent
  measurement-error model with unit-specific variances;
- `simex_nb2`: empirical SIMEX (Cook and Stefanski, 1994; Devanarayan and Stefanski, 2002) for the NB2 model of
  Eq. (1), with pseudo-errors drawn from the split-half contrasts. Indicators whose calibration on a 1 Hz validation
  subset rejects the identity line are first rescaled by the calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .calibration import Calibration
from .nb2 import fit_nb2

ZETAS = (0.5, 1.0, 1.5, 2.0)


def unit_error_variances(sigma2: pd.Series, gamma1: float = 1.0, sigma2_eps: float = 0.0) -> pd.Series:
    """Eq. (6): (sigma2_if + sigma2_eps_f) / gamma1_f^2."""
    return (sigma2 + float(sigma2_eps)) / float(gamma1) ** 2


@dataclass
class SimexResult:
    coef: pd.Series
    se: pd.Series
    naive_coef: pd.Series
    naive_se: pd.Series
    alpha: float
    path: pd.DataFrame              # mean coefficient at each zeta (zeta = 0 is the naive fit)
    corrected: list


def simex_nb2(y: pd.Series, design: pd.DataFrame, indicators: pd.DataFrame, sigma2: pd.DataFrame,
              correct: Sequence[str], offset: Optional[np.ndarray] = None, half_diff: Optional[np.ndarray] = None,
              half_diff_cols: Optional[Sequence[str]] = None, calibrations: Optional[Dict[str, Calibration]] = None,
              zetas: Sequence[float] = ZETAS, B: int = 20, seed: Optional[int] = None, alpha: Optional[float] = None,
              level: float = 0.05, flags: Optional[pd.DataFrame] = None) -> SimexResult:
    """Empirical SIMEX for the coefficients of `correct`.

    y, design, indicators and sigma2 share the unit index. `half_diff` (J x indicators x units, e.g.
    `SplitHalfResult.half_diff` with `half_diff_cols` = its used indicators) supplies the pseudo-errors: for each
    unit, d_ij / 2 from a random partition j with a random sign, scaled by sqrt(zeta). Without it, normal
    pseudo-errors with variance zeta * sigma2 are used; they are also used for units flagged in step 3 (`flags`) or
    without a half difference, whose variance step 5 sets by rule. The dispersion is held at the naive estimate (or
    `alpha`).
    The SIMEX estimate is the quadratic extrapolation of the mean coefficient path to zeta = -1; its variance follows
    Stefanski and Cook (mean naive variance minus the variance between pseudo-data sets, extrapolated the same way)."""
    rng = np.random.default_rng(seed)
    uid = design.index
    X = indicators.loc[uid].copy()
    s2 = sigma2.loc[uid].copy()
    for f in X.columns:
        X[f] = X[f].fillna(np.nanmedian(X[f]))
    scale = {}
    if calibrations:
        for f, cal in calibrations.items():
            if f in X.columns and cal.rejects_identity(level):
                X[f] = cal.rescale(X[f].to_numpy())
                s2[f] = s2[f] / cal.gamma1 ** 2
                scale[f] = cal.gamma1
    Z = pd.concat([design, X], axis=1)
    cols = list(Z.columns)
    Zm = Z.to_numpy(float)
    yy = y.reindex(uid).to_numpy(float)
    off = np.zeros(len(uid)) if offset is None else np.asarray(offset, float)
    naive = fit_nb2(yy, Zm, off, alpha)
    a = naive.alpha
    idx = [cols.index(f) for f in correct]
    if half_diff is not None:
        hcols = list(half_diff_cols) if half_diff_cols is not None else list(correct)
        hd = {f: half_diff[:, hcols.index(f), :] / 2.0 / scale.get(f, 1.0) for f in correct}
        by_rule = {}
        for f in correct:
            by_rule[f] = ~np.isfinite(hd[f]).all(axis=0)
            if flags is not None and f in flags.columns:
                by_rule[f] |= flags[f].reindex(uid).fillna(False).to_numpy(bool)
            hd[f] = np.where(np.isfinite(hd[f]), hd[f], 0.0)
    path_b, path_v = [naive.coef], [naive.se ** 2]
    for z in zetas:
        bb, vv = [], []
        for _ in range(B):
            Zp = Zm.copy()
            for f, j in zip(correct, idx):
                if half_diff is not None:
                    J = hd[f].shape[0]
                    pick = rng.integers(0, J, len(uid))
                    sign = rng.choice([-1.0, 1.0], len(uid))
                    e = sign * hd[f][pick, np.arange(len(uid))]
                    e = np.where(by_rule[f], rng.normal(0.0, 1.0, len(uid)) * np.sqrt(s2[f].to_numpy(float)), e)
                else:
                    e = rng.normal(0.0, 1.0, len(uid)) * np.sqrt(s2[f].to_numpy(float))
                Zp[:, j] = Zp[:, j] + np.sqrt(z) * e
            fz = fit_nb2(yy, Zp, off, a)
            bb.append(fz.coef)
            vv.append(fz.se ** 2)
        bb = np.array(bb)
        path_b.append(bb.mean(0))
        path_v.append(np.mean(vv, 0) - bb.var(0, ddof=1))
    zz = np.r_[0.0, list(zetas)]
    pb, pv = np.array(path_b), np.array(path_v)
    est = naive.coef.copy()
    var = naive.se ** 2
    for j in idx:
        est[j] = np.polyval(np.polyfit(zz, pb[:, j], 2), -1.0)
        var[j] = max(np.polyval(np.polyfit(zz, pv[:, j], 2), -1.0), pv[0, j])
    path = pd.DataFrame(pb[:, idx], index=pd.Index(zz, name='zeta'), columns=list(correct))
    return SimexResult(coef=pd.Series(est, index=cols), se=pd.Series(np.sqrt(var), index=cols),
                       naive_coef=pd.Series(naive.coef, index=cols), naive_se=pd.Series(naive.se, index=cols),
                       alpha=a, path=path, corrected=list(correct))

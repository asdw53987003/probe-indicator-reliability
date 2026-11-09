# -*- coding: utf-8 -*-
"""Executing the split-half check on individual vehicles against held-out vehicles.

Where full 1 Hz records of a fleet with persistent vehicle identifiers are available, part of the vehicles can be
retained as the "degraded" data and the rest held out as a reference. The held-out reliability is the weighted
covariance of the degraded indicator with the 1 s indicator of the held-out vehicles, divided by the weighted variance
of the degraded indicator, both given the other columns of Eq. (1) (Carroll et al., 2006). Under a coarser sampling
interval it is an attenuation that includes rescaling and can exceed 1.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .indicators import INDICATORS, hdc, indicators_table, sums_from_points, thin_records
from .lists import list_overlap, list_size
from .split_half import split_half_check


def _filled(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: c.fillna(np.nanmedian(c)))


def held_out_reliability(degraded: pd.DataFrame, held_out: pd.DataFrame, design: pd.DataFrame,
                         weights: pd.Series) -> pd.Series:
    """Diagonal of M^-1 Z*' W Z_ho for each indicator column shared by `degraded` and `held_out`.

    Z* is the design with the degraded indicators and Z_ho the same design with the held-out indicators; missing
    indicator values take the median of the column."""
    feats = [f for f in degraded.columns if f in held_out.columns]
    uid = design.index
    Zd = pd.concat([design, _filled(degraded.loc[uid, feats])], axis=1)
    Zh = pd.concat([design, _filled(held_out.loc[uid, feats])], axis=1)
    w = weights.reindex(uid).to_numpy(float)
    A, Bm = Zd.to_numpy(float), Zh.to_numpy(float)
    M = (A * w[:, None]).T @ A
    L = np.linalg.solve(M, (A * w[:, None]).T @ Bm)
    cols = list(Zd.columns)
    return pd.Series({f: L[cols.index(f), cols.index(f)] for f in feats})


def retention_mask(vehicles: np.ndarray, unit_of_row: np.ndarray, rates: pd.Series, uniforms: pd.Series) -> np.ndarray:
    """Rows of vehicle x unit sums kept when vehicle v is retained in unit i if U_v < r_i."""
    u = uniforms.reindex(vehicles).to_numpy(float)
    r = rates.reindex(unit_of_row).to_numpy(float)
    return u < r


def executed_check(points: pd.DataFrame, units: pd.DataFrame, design: pd.DataFrame, period_weeks: float,
                   retain_share: float = 0.5, tau: int = 1, draws: int = 20, seed: Optional[int] = None,
                   indicators: Sequence[str] = INDICATORS, J: int = 20, coef: Optional[pd.Series] = None,
                   top_share: float = 0.10, sigma_u: str = 'diagonal') -> pd.DataFrame:
    """Retain vehicles at random (a share, or a per-unit rate in units['retain']), thin them to `tau` seconds, run
    the split-half check on them and compare it with the held-out vehicles of each unit.

    `units` must carry working weights ('weight'). Returns one row per draw and quantity with the split-half value
    (prediction from the retained vehicles alone) and the held-out value; for lists, the predicted overlap and the
    overlap of the lists from retained and held-out vehicles (which itself contains the error of the held-out
    vehicles). `sigma_u` is passed to split_half_check ('diagonal' as in Eq. (3), or 'full')."""
    if 'weight' not in units.columns:
        raise ValueError("executed_check needs working weights in units['weight']")
    rng = np.random.default_rng(seed)
    rates = units['retain'] if 'retain' in units.columns else pd.Series(retain_share, index=units.index)
    s1 = sums_from_points(points, 1.0)
    veh = np.unique(points['vehicle'].to_numpy())
    days = 7.0 * float(period_weeks)
    k = list_size(len(units), top_share)
    rows = []
    for d in range(draws):
        U = pd.Series(rng.random(len(veh)), index=veh)
        keep_rows = retention_mask(s1['vehicle'].to_numpy(), s1['unit'].to_numpy(), rates, U)
        held = s1.loc[~keep_rows]
        if tau > 1:
            pk = points.merge(s1.loc[keep_rows, ['vehicle', 'unit']], on=['vehicle', 'unit'], how='inner')
            deg = sums_from_points(thin_records(pk, tau, seed=int(rng.integers(2 ** 31))), tau)
        else:
            deg = s1.loc[keep_rows]
        res = split_half_check(deg, units, design, period_weeks, coef=coef, J=J, seed=int(rng.integers(2 ** 31)),
                               indicators=indicators, top_share=top_share, sampling_interval_s=tau,
                               sigma_u=sigma_u)
        used = res.settings['used']
        hx = indicators_table(held, units, used)
        dx = res.indicators[used]
        rho = held_out_reliability(dx, hx[used], design, res.weights)
        for f in used:
            rows.append(dict(draw=d, quantity='reliability', indicator=f, split_half=float(res.lam[f]),
                             held_out=float(rho[f])))
        if 'HDR' in used:
            pairs = {'HDR': (dx['HDR'].to_numpy(float), hx['HDR'].to_numpy(float)),
                     'HDC': (hdc(dx['HDR'], res.indicators['veq'], days), hdc(hx['HDR'], hx['veq'], days))}
            for nm, (a, b) in pairs.items():
                rows.append(dict(draw=d, quantity='list overlap', indicator=nm, split_half=float(res.lists[nm]),
                                 held_out=list_overlap(b, a, k)))
    out = pd.DataFrame(rows)
    out['difference'] = out['split_half'] - out['held_out']
    return out

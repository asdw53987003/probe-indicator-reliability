# -*- coding: utf-8 -*-
"""Split-half check of probe indicators for a crash-frequency model and site lists (Algorithm A.1, steps 1-8).

Inputs
------
records : one row per vehicle x unit over ONE observation period, with the sums of `indicators.sums_from_points`
          (vehicle, unit, km, kmv, kmv2, kma2, km_acc, hd_events). Vehicle identifiers must stay the same over the
          whole period: splitting passes or daily identifiers instead of vehicles understates the error.
units   : DataFrame indexed by unit with length_km, group (route and direction, the fixed-effect group of Eq. (1)),
          and either weight (working weight mu / (1 + alpha mu) of a fitted NB2 model) or crash counts passed in
          `crash_counts`, from which Eq. (1) is fitted (step 1). Optional: offset (ln length_km when absent),
          Q (daily volume, for the EHC list) and route (for lists within each route).
design  : design matrix of Eq. (1) WITHOUT the probe indicators (constant, fixed-effect dummies, covariates).
coef    : optional fitted coefficients of every design column and indicator; needed for the predicted coefficient
          ratios and the risk-score list. Taken from the step-1 fit when crash counts are given.
period_weeks : length of the observation period (sets the days of the HDC list and the weeks needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .indicators import INDICATORS, standard_columns, unit_indicators
from .lists import list_size, predicted_overlap, ranking_reliability
from .nb2 import fit_nb2
from .requirements import MIN_VEQ_NBAR, k_hat, n_required, volume_summary, weeks_needed

MIN_VEH = 2                     # a unit needs at least this many vehicles in each half (step 3)
MAX_FLAGGED = 0.05              # an indicator is not used if more than this share of units is flagged (step 5)
MIN_VALUED = 0.95               # an indicator stays in the model while at least this share of units has a value


@dataclass
class SplitHalfResult:
    sigma2: pd.DataFrame            # unit x indicator, Eq. (2) from centred half differences (model inputs)
    sigma2_uncentred: pd.DataFrame  # unit x indicator, Eq. (2) from uncentred half differences (network lists)
    Lambda: pd.DataFrame            # attenuation matrix, Eq. (3)
    lam: pd.Series                  # split-half reliability of each used indicator
    unit_rel: pd.DataFrame          # unit reliability
    nbar: float                     # volume summary (weighted harmonic mean of vehicle-equivalents)
    k: pd.Series
    n_req: pd.Series
    weeks_req: pd.Series
    pred_ratio: Optional[pd.Series]
    coef_se: Optional[pd.Series]    # standard errors when the coefficients were fitted in step 1
    lists: pd.Series                # Eq. (5), network lists
    lists_by_route: Optional[pd.DataFrame]
    flags: pd.DataFrame
    usable: pd.Series
    balance: pd.Series
    indicators: pd.DataFrame        # indicators from all vehicles, with veq
    weights: pd.Series
    half_diff: np.ndarray           # J x used indicators x units, centred half differences (pseudo-errors for SIMEX)
    settings: dict
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        s = self.settings
        out = ['Split-half check of probe indicators (Algorithm A.1)',
               '  observation period %g weeks; J = %d partitions; divisor %s; few-vehicle rule %s; Sigma_U %s; '
               'lambda_req %.2f; list size %d of %d units' % (s['period_weeks'], s['J'], s['divisor'], s['few_rule'],
                                                              s['sigma_u'], s['lam_req'], s['top_k'], s['n_units']),
               '  volume summary n-bar (working-weighted harmonic mean, units with >= %g veh.-equiv.): %.1f'
               % (MIN_VEQ_NBAR, self.nbar),
               '  split-half reliability, Eq. (4) and observation period needed:']
        for f in self.usable.index:
            if not self.usable[f]:
                out.append('    %-4s not used: %.1f%% of units flagged, %.1f%% without a value'
                           % (f, 100 * self.flags[f].mean(), 100 * s['missing_share'][f]))
                continue
            out.append('    %-4s lambda %.3f   k = %s   n_req = %s   weeks needed = %s   flagged units %.1f%%'
                       % (f, self.lam[f], _fmt(self.k[f]), _fmt(self.n_req[f]), _fmt(self.weeks_req[f], 1),
                          100 * self.flags[f].mean()))
        if self.pred_ratio is not None:
            parts = []
            for f, v in self.pred_ratio.items():
                weak = self.coef_se is not None and 2.0 * self.coef_se[f] > abs(self.settings['coef'][f])
                parts.append('%s %.3f%s' % (f, v, ' (coefficient within 2 SE of zero)' if weak else ''))
            out.append('  predicted coefficient ratio: ' + ', '.join(parts))
        out.append('  expected list overlap, Eq. (5): ' + ', '.join('%s %.3f' % kv for kv in self.lists.items()))
        if self.lists_by_route is not None and len(self.lists_by_route):
            out.append('  expected list overlap within routes:')
            out.append('    ' + self.lists_by_route.to_string(float_format=lambda v: '%.3f' % v).replace('\n', '\n    '))
        out.append("  smaller half's share of a unit's indicator weight, median over partitions: %.3f to %.3f"
                   % (np.nanmin(self.balance), np.nanmax(self.balance)))
        out += ['  note: ' + n for n in self.notes]
        return '\n'.join(out)


def _fmt(x, nd=0):
    return 'n/a' if not np.isfinite(x) else ('{:,.%df}' % nd).format(x)


def _centre(D: np.ndarray, grp: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Subtract the w-weighted mean of the half differences within each group (route and direction), step 4."""
    D = D.copy()
    for g in np.unique(grp):
        sel = grp == g
        dk = D[:, :, sel]
        wk = np.where(np.isfinite(dk), w[sel][None, None, :], 0.0)
        mean = np.nansum(np.where(np.isfinite(dk), dk, 0.0) * wk, axis=2, keepdims=True) / \
            np.maximum(wk.sum(axis=2, keepdims=True), 1e-300)
        D[:, :, sel] = dk - mean
    return D


def _apply_rule(v: np.ndarray, bad: np.ndarray, rule: str, x: np.ndarray, veq: np.ndarray) -> np.ndarray:
    v = v.copy()
    good = ~bad & np.isfinite(v)
    if rule == 'F':                                         # largest variance of the unflagged units (paper)
        v[~good] = np.nanmax(v[good]) if good.any() else np.nan
    elif rule == 'O':                                       # flagged units add no error variance
        v[~good] = 0.0
    else:                                                   # single-vehicle variance function s1^2(X) / n
        nn = np.maximum(veq, 1.0)
        ok = good & (x > 0) & (v > 0)
        if ok.sum() >= 3:
            b, a = np.polyfit(np.log(x[ok]), np.log(v[ok] * nn[ok]), 1)
            fn = np.exp(a) * np.maximum(x, 1e-12) ** b / nn
            fn *= np.nanmedian(v[ok] / fn[ok])
            v[~good] = fn[~good]
        else:
            v[~good] = np.nanmax(v[good]) if good.any() else np.nan
    return v


def attenuation_matrix(Z: np.ndarray, w: np.ndarray, sigma_u: np.ndarray) -> np.ndarray:
    """Eq. (3): Lambda = (Z' W Z)^-1 (Z' W Z - Sigma_U). `sigma_u` is the diagonal (sum_i w_i sigma2_if per column)
    or a full matrix."""
    M = (Z * w[:, None]).T @ Z
    su = np.asarray(sigma_u, float)
    su = np.diag(su) if su.ndim == 1 else su
    try:
        return np.linalg.solve(M, M - su)
    except np.linalg.LinAlgError:
        raise ValueError('Z\' W Z is singular: an indicator or design column is collinear with the others '
                         '(e.g. constant across units)') from None


def split_half_check(records: pd.DataFrame, units: pd.DataFrame, design: pd.DataFrame, period_weeks: float,
                     coef: Optional[pd.Series] = None, crash_counts: Optional[pd.Series] = None, J: int = 20,
                     seed: Optional[int] = None, lam_req: float = 0.9, top_share: float = 0.10,
                     indicators: Sequence[str] = INDICATORS, few_rule: str = 'F', divisor: str = '4',
                     assignment: Optional[np.ndarray] = None,
                     ratio_indicators: Optional[Dict[str, Tuple[str, str]]] = None,
                     sampling_interval_s: float = 1.0, sigma_u: str = 'diagonal') -> SplitHalfResult:
    """Run steps 1-8 of Algorithm A.1.

    J            partitions of the whole fleet into two halves of equal size, the same half in every unit (step 2).
    seed         seed of the partitions; None draws fresh entropy (results then vary by partition noise only).
    few_rule     flagged units: 'F' the largest variance of the unflagged units (paper); 'V' a single-vehicle variance
                 function fitted on unflagged units; 'O' no error variance.
    divisor      '4' (Eq. 2) divides squared half differences by 4, exact for halves of equal weight in each unit;
                 'size' divides by W^2 / (W_1 W_2) per unit and partition, for units whose halves differ in weight.
    assignment   optional J x vehicles boolean array (sorted vehicle identifiers) replacing the random halves.
    top_share    share of units on a site list.
    sigma_u      'diagonal' (Eq. 3 of the paper): error variances only. 'full' adds the error covariances between
                 indicators, estimated from cross-products of the centred half differences; indicators computed from
                 the same records (e.g. ASD and HDR) can have correlated errors, which the diagonal form ignores."""
    if sigma_u not in ('diagonal', 'full'):
        raise ValueError("sigma_u must be 'diagonal' or 'full'")
    if divisor not in ('4', 'size'):
        raise ValueError("divisor must be '4' or 'size'")
    if few_rule not in ('F', 'V', 'O'):
        raise ValueError("few_rule must be 'F', 'V' or 'O'")
    indicators = list(indicators)
    uid = units.index
    n = len(uid)
    rec = standard_columns(records[records['unit'].isin(uid)].reset_index(drop=True))
    uidx = uid.get_indexer(rec['unit'])
    veh_codes, vidx = np.unique(rec['vehicle'].to_numpy(), return_inverse=True)
    nv = len(veh_codes)
    grp = pd.factorize(units['group'])[0]
    L = units['length_km'].to_numpy(float)
    off = units['offset'].to_numpy(float) if 'offset' in units.columns else np.log(L)
    days = 7.0 * float(period_weeks)
    top_k = list_size(n, top_share)
    q = top_k / n

    # step 1: indicators from all retained vehicles; working weights of Eq. (1)
    full = unit_indicators(rec, n, uidx, indicators, ratio_indicators)
    X = pd.DataFrame({f: full[f] for f in indicators}, index=uid)
    veq = full['km'] / L
    nveh = np.bincount(uidx, minlength=n)
    coef_se = None
    if crash_counts is not None:
        Xf = X.apply(lambda c: c.fillna(np.nanmedian(c)))
        Z1 = pd.concat([design.loc[uid], Xf], axis=1)
        fit = fit_nb2(crash_counts.reindex(uid).to_numpy(float), Z1.to_numpy(float), off)
        if coef is None:
            coef = pd.Series(fit.coef, index=Z1.columns)
            coef_se = pd.Series(fit.se, index=Z1.columns)
    if 'weight' in units.columns:
        w = units['weight'].to_numpy(float)
    elif crash_counts is not None:
        w = fit.working_weights
    else:
        raise ValueError("units need a 'weight' column, or pass crash_counts to fit Eq. (1)")

    # steps 2-3: J partitions of the whole fleet; indicators of both halves in each unit
    rng = np.random.default_rng(seed)
    D = np.full((J, len(indicators), n), np.nan)
    S = np.full((J, len(indicators), n), np.nan)                       # W1 W2 / W^2 of each split
    minveh = np.zeros((J, n))
    for j in range(J):
        if assignment is not None:
            bitv = np.asarray(assignment[j], bool)
        else:
            perm = rng.permutation(nv)
            bitv = np.zeros(nv, bool)
            bitv[perm[: nv // 2]] = True
        bit = bitv[vidx]
        a = unit_indicators(rec[bit], n, uidx[bit], indicators, ratio_indicators)
        b = unit_indicators(rec[~bit], n, uidx[~bit], indicators, ratio_indicators)
        ca = np.bincount(uidx[bit], minlength=n)
        minveh[j] = np.minimum(ca, nveh - ca)
        for fi, f in enumerate(indicators):
            D[j, fi] = a[f] - b[f]
            wa, wb = a['_w'][f], b['_w'][f]
            with np.errstate(all='ignore'):
                S[j, fi] = wa * wb / (wa + wb) ** 2
    # step 4: centre within route and direction
    Dc = _centre(D, grp, w)

    # step 5: Eq. (2), flags and the few-vehicle rule
    def eq2(DD):
        with np.errstate(all='ignore'):
            return np.nanmean(DD ** 2, axis=0) / 4.0 if divisor == '4' else np.nanmean(DD ** 2 * S, axis=0)
    s2c, s2u = eq2(Dc), eq2(D)
    few = (minveh < MIN_VEH).any(axis=0)
    flags = pd.DataFrame(False, index=uid, columns=indicators)
    sigma2 = pd.DataFrame(index=uid, columns=indicators, dtype=float)
    sigma2_unc = pd.DataFrame(index=uid, columns=indicators, dtype=float)
    missing = {}
    for fi, f in enumerate(indicators):
        undefined = ~np.isfinite(D[:, fi, :]).all(axis=0) | ~np.isfinite(s2c[fi])
        bad = few | undefined
        flags[f] = bad
        missing[f] = float((~np.isfinite(X[f].to_numpy(float))).mean())
        xx = X[f].to_numpy(float)
        sigma2[f] = _apply_rule(s2c[fi], bad, few_rule, xx, veq)
        sigma2_unc[f] = _apply_rule(s2u[fi], bad, few_rule, xx, veq)
    usable = pd.Series({f: (1.0 - missing[f] >= MIN_VALUED) and (flags[f].mean() <= MAX_FLAGGED) for f in indicators})
    used = [f for f in indicators if usable[f]]
    if not used:
        raise ValueError('no indicator is usable for this period (too many flagged units or units without a value)')

    # step 6: Eq. (3), split-half reliability, predicted coefficient ratios and unit reliability
    Xf = X[used].copy()
    for f in used:
        Xf[f] = Xf[f].fillna(np.nanmedian(Xf[f]))
    Z = pd.concat([design.loc[uid], Xf], axis=1)
    Zm = Z.to_numpy(float)
    cols = list(Z.columns)
    SU = np.zeros((len(cols), len(cols)))
    for f in used:
        SU[cols.index(f), cols.index(f)] = np.sum(w * sigma2[f].to_numpy())
    if sigma_u == 'full' and len(used) > 1:
        for a_, b_ in [(a_, b_) for i, a_ in enumerate(used) for b_ in used[i + 1:]]:
            ia, ib = indicators.index(a_), indicators.index(b_)
            with np.errstate(all='ignore'):
                cab = np.nanmean(Dc[:, ia, :] * Dc[:, ib, :], axis=0) / 4.0 if divisor == '4' else \
                    np.nanmean(Dc[:, ia, :] * Dc[:, ib, :] * np.sqrt(S[:, ia, :] * S[:, ib, :]), axis=0)
            sa, sb = sigma2[a_].to_numpy(), sigma2[b_].to_numpy()
            good = ~(flags[a_].to_numpy() | flags[b_].to_numpy()) & np.isfinite(cab)
            rho = np.average(cab[good] / np.sqrt(s2c[ia][good] * s2c[ib][good]), weights=w[good]) if good.any() else 0.0
            cab = np.where(good, cab, rho * np.sqrt(sa * sb))
            SU[cols.index(a_), cols.index(b_)] = SU[cols.index(b_), cols.index(a_)] = np.sum(w * cab)
    Lam = attenuation_matrix(Zm, w, SU)
    Lambda = pd.DataFrame(Lam, index=cols, columns=cols)
    lam = pd.Series({f: Lam[cols.index(f), cols.index(f)] for f in used})
    unit_rel = pd.DataFrame(index=uid, columns=used, dtype=float)
    sw = np.sqrt(w)
    for f in used:
        jj = cols.index(f)
        others = [c for c in range(len(cols)) if c != jj]
        cf, *_ = np.linalg.lstsq(Zm[:, others] * sw[:, None], Zm[:, jj] * sw, rcond=None)
        res = Zm[:, jj] - Zm[:, others] @ cf
        vc = np.average(res ** 2, weights=w) - np.average(sigma2[f], weights=w)
        unit_rel[f] = vc / (vc + sigma2[f])
    pred = None
    if coef is not None:
        cvec = coef.reindex(cols)
        if cvec.isna().any():
            raise ValueError('coef must contain every design column and used indicator: missing %s'
                             % list(cvec[cvec.isna()].index))
        bh = np.linalg.solve(Lam, cvec.to_numpy(float))
        pred = pd.Series({f: cvec[f] / bh[cols.index(f)] for f in used})

    # step 7: Eq. (4) and the observation period needed at the same penetration
    nbar = volume_summary(veq, w)
    k = pd.Series({f: k_hat(nbar, lam[f]) for f in used})
    n_req = pd.Series({f: n_required(k[f], lam_req) for f in used})
    weeks_req = pd.Series({f: weeks_needed(n_req[f], nbar, period_weeks) for f in used})

    # step 8: site lists, Eq. (5). The risk score keeps the centred variances (route effects absorb shared error);
    # lists ranked by raw values use uncentred variances, because errors shared along a route reorder their units.
    def list_quantities(sel):
        out = {}
        if coef is not None:
            score = Zm[sel] @ coef.reindex(cols).to_numpy(float) + off[sel]
            ev = sum(coef[f] ** 2 * sigma2[f].to_numpy(float)[sel] for f in used)
            out['risk score'] = (score, ev)
        if 'HDR' in used:
            hdr = X['HDR'].to_numpy(float)[sel]
            e = sigma2_unc['HDR'].to_numpy(float)[sel]
            out['HDR'] = (hdr, e)
            hs = veq[sel] / 100.0 / days                     # probe exposure taken as observed
            out['HDC'] = (hdr * hs, hs ** 2 * e)
            if 'Q' in units.columns:
                qs = units['Q'].to_numpy(float)[sel] / 100.0
                out['EHC'] = (hdr * qs, qs ** 2 * e)
        return out
    allsel = np.ones(n, bool)
    lists = pd.Series({nm: predicted_overlap(ranking_reliability(x, e), q)
                       for nm, (x, e) in list_quantities(allsel).items()}, dtype=float)
    by_route = None
    if 'route' in units.columns:
        rows = []
        for r, idx in units.groupby('route').indices.items():
            sel = np.zeros(n, bool)
            sel[idx] = True
            kr = list_size(int(sel.sum()), top_share)
            row = {'route': r, 'units': int(sel.sum()), 'list size': kr}
            for nm, (x, e) in list_quantities(sel).items():
                row[nm] = predicted_overlap(ranking_reliability(x, e), kr / sel.sum())
            rows.append(row)
        by_route = pd.DataFrame(rows).set_index('route')

    with np.errstate(all='ignore'):
        prod = np.nanmedian(S[:, indicators.index(used[0]), :], axis=0)
        balance = pd.Series(0.5 - np.sqrt(np.maximum(0.25 - prod, 0.0)), index=uid)

    notes = ['vehicle identifiers must stay the same over the observation period; splitting passes, trips or daily '
             'identifiers misses the error shared by the passes of a vehicle.',
             'the check registers sampling error only; distortions shared by all probes of a unit (sampling interval, '
             'probe exposure behind counts, fleet mix) need the separate checks of step 9.',
             'Eq. (5) treats the errors of units as independent; HDC list predictions can be optimistic.']
    if sampling_interval_s > 1:
        notes.append('sampling interval %g s > 1 s: calibrate the scale of acceleration-based indicators on a 1 Hz '
                     'validation subset (calibration.calibration_slope) before using them.' % sampling_interval_s)
    for f in indicators:
        if not usable[f]:
            notes.append('%s is not used for this period: more than %d%% of units are flagged or more than %d%% lack a '
                         'value.' % (f, round(100 * MAX_FLAGGED), round(100 * (1 - MIN_VALUED))))
    if divisor == '4' and np.nanmin(balance) < 0.4:
        notes.append("some units split their indicator weight unevenly between halves (smallest share %.2f); "
                     "divisor='size' removes the resulting upward bias of Eq. (2)." % np.nanmin(balance))
    settings = dict(period_weeks=float(period_weeks), J=J, divisor=divisor, few_rule=few_rule, lam_req=lam_req,
                    top_k=top_k, n_units=n, seed=seed, missing_share=missing, n_vehicles=nv, used=used,
                    sigma_u=sigma_u, coef=coef)
    return SplitHalfResult(sigma2=sigma2, sigma2_uncentred=sigma2_unc, Lambda=Lambda, lam=lam, unit_rel=unit_rel,
                           nbar=nbar, k=k, n_req=n_req, weeks_req=weeks_req, pred_ratio=pred, coef_se=coef_se,
                           lists=lists,
                           lists_by_route=by_route, flags=flags, usable=usable, balance=balance,
                           indicators=X.assign(veq=veq), weights=pd.Series(w, index=uid),
                           half_diff=Dc[:, [indicators.index(f) for f in used], :], settings=settings, notes=notes)

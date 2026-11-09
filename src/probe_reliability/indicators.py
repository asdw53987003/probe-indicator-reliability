# -*- coding: utf-8 -*-
"""Unit indicators from 1 Hz probe records (Table S1 of the paper).

Indicators
----------
MSP  mean probe speed (m/s): pass mean speeds weighted by vehicle-kilometres.
VSD  between-vehicle speed SD (m/s): SD of pass mean speeds about MSP, weighted by vehicle-kilometres, so that the
     variance within and between hours is pooled.
ASD  SD of accelerations between records (m/s^2): root of the vehicle-kilometre-weighted mean of the pass mean
     squared accelerations.
HDR  harsh decelerations per 100 probe vehicle-kilometres: 100 x episodes / vehicle-kilometres.
HDC  harsh decelerations per km and day: HDR x vehicle-equivalents / 100 / days (mixes behaviour with probe traffic).
EHC  volume-expanded harsh-deceleration count per km and day: HDR x daily volume / 100.

Processing rules for raw records: accelerations are speed differences over the time gap between kept records,
differenced only across gaps up to max(2 tau, 60) s; a harsh deceleration is an acceleration of -2.0 m/s^2 or less,
and consecutive exceedances count once.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HARSH = -2.0                    # harsh-deceleration threshold, m/s^2
INDICATORS = ('MSP', 'VSD', 'ASD', 'HDR')
SUM_COLUMNS = ('km', 'kmv', 'kmv2', 'kma2', 'km_acc', 'hd_events')


def gap_limit(tau: float) -> float:
    """Longest gap (s) across which successive kept records are differenced."""
    return max(2.0 * float(tau), 60.0)


def thin_records(points: pd.DataFrame, tau: int, seed: Optional[int] = None) -> pd.DataFrame:
    """Keep every tau-th record of each vehicle from a random offset (a coarser reporting interval).

    `points` needs columns vehicle and t (s, integer seconds of a 1 Hz record). The offset is drawn once per vehicle,
    so all passes of a vehicle share it."""
    tau = int(tau)
    if tau <= 1:
        return points.copy()
    rng = np.random.default_rng(seed)
    veh = np.unique(points['vehicle'].to_numpy())
    offset = pd.Series(rng.integers(0, tau, len(veh)), index=veh)
    t = np.round(points['t'].to_numpy(float)).astype(np.int64)
    keep = ((t + offset.reindex(points['vehicle']).to_numpy()) % tau) == 0
    return points.loc[keep].copy()


def sums_from_points(points: pd.DataFrame, tau: float = 1.0) -> pd.DataFrame:
    """Sums per vehicle x unit from records with columns vehicle, unit, pass, t (s) and speed (m/s).

    A pass is one traversal of the unit. Vehicle-kilometres of a pass integrate speed over the gaps between its
    records (gaps longer than max(2 tau, 60) s are not bridged), and its mean speed is distance over time.
    Accelerations are differences of successive kept records over gaps up to max(2 tau, 60) s; runs of accelerations
    at or below -2.0 m/s^2 count once.

    Returns one row per vehicle x unit with columns vehicle, unit, km, kmv, kmv2, kma2, km_acc, hd_events."""
    need = {'vehicle', 'unit', 'pass', 't', 'speed'}
    missing = need - set(points.columns)
    if missing:
        raise ValueError('points lack columns %s' % sorted(missing))
    gmax = gap_limit(tau)
    p = points.sort_values(['vehicle', 'unit', 'pass', 't']).reset_index(drop=True)
    key = p['vehicle'].astype(str) + '\x1f' + p['unit'].astype(str) + '\x1f' + p['pass'].astype(str)
    same = key.eq(key.shift())
    dt = p['t'].diff().where(same)
    dv = p['speed'].diff().where(same)
    ok = (dt > 0) & (dt <= gmax)
    acc = (dv / dt).where(ok)
    dist = (0.5 * (p['speed'] + p['speed'].shift()) * dt).where(ok)          # trapezoid, m
    harsh = acc <= HARSH
    start = harsh & ~(harsh.shift(fill_value=False) & same & ok.shift(fill_value=False))
    g = pd.DataFrame({'vehicle': p['vehicle'], 'unit': p['unit'], 'pass': p['pass'], 'dist': dist.fillna(0.0),
                      'time': dt.where(ok).fillna(0.0), 'a2': (acc ** 2).fillna(0.0), 'na': ok.astype(int),
                      'hd': start.astype(int)})
    ps = g.groupby(['vehicle', 'unit', 'pass'], sort=False).sum().reset_index()
    ps = ps[ps['time'] > 0]
    km = ps['dist'] / 1000.0
    v = ps['dist'] / ps['time']
    ma2 = ps['a2'] / ps['na'].where(ps['na'] > 0)
    out = pd.DataFrame({'vehicle': ps['vehicle'], 'unit': ps['unit'], 'km': km, 'kmv': km * v, 'kmv2': km * v ** 2,
                        'kma2': (km * ma2).fillna(0.0), 'km_acc': km.where(ps['na'] > 0, 0.0), 'hd_events': ps['hd']})
    return out.groupby(['vehicle', 'unit'], sort=False).sum().reset_index()


def standard_columns(rec: pd.DataFrame) -> pd.DataFrame:
    """Accept per-vehicle sums with n_spd, spd_sum, n_acc and acc2_sum (each vehicle then counts as one pass)."""
    rec = rec.copy()
    if 'kmv' not in rec.columns and 'spd_sum' in rec.columns:
        mv = rec['spd_sum'].to_numpy(float) / rec['n_spd'].to_numpy(float)
        rec['kmv'] = rec['km'] * mv
        rec['kmv2'] = rec['km'] * mv ** 2
    if 'kma2' not in rec.columns and 'acc2_sum' in rec.columns:
        rec['kma2'] = (rec['km'] * rec['acc2_sum'] / rec['n_acc'].where(rec['n_acc'] > 0)).fillna(0.0)
        rec['km_acc'] = rec['km'].where(rec['n_acc'] > 0, 0.0)
    if 'km_acc' not in rec.columns:
        rec['km_acc'] = rec['km']
    return rec


def unit_indicators(rec: pd.DataFrame, n_units: int, uidx: np.ndarray, indicators: Sequence[str],
                    ratio_indicators: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, np.ndarray]:
    """Indicators of each unit from the vehicles in `rec` (rows aligned with `uidx`, the unit positions).

    Returns a dict with one array per indicator, 'km' (vehicle-kilometres per unit) and '_w', the weight sum behind each
    indicator (vehicle-kilometres, or the denominator of a ratio indicator)."""
    def s(col):
        return np.bincount(uidx, weights=rec[col].to_numpy(float), minlength=n_units)
    out, wsum = {}, {}
    with np.errstate(all='ignore'):
        skm = s('km')
        out['km'] = skm
        for f in indicators:
            if ratio_indicators and f in ratio_indicators:
                num, den = ratio_indicators[f]
                d = s(den)
                out[f] = s(num) / d
                wsum[f] = d
            elif f == 'MSP':
                out[f] = s('kmv') / skm
                wsum[f] = skm
            elif f == 'VSD':
                m = s('kmv') / skm
                out[f] = np.sqrt(np.maximum(s('kmv2') / skm - m ** 2, 0.0))
                wsum[f] = skm
            elif f == 'ASD':
                ka = s('km_acc')
                out[f] = np.sqrt(s('kma2') / ka)
                wsum[f] = ka
            elif f == 'HDR':
                out[f] = 100.0 * s('hd_events') / skm
                wsum[f] = skm
            else:
                raise ValueError('unknown indicator %s: pass it in ratio_indicators' % f)
            out[f] = np.where(wsum[f] > 0, out[f], np.nan)
    out['_w'] = wsum
    return out


def indicators_table(records: pd.DataFrame, units: pd.DataFrame, indicators: Sequence[str] = INDICATORS,
                     ratio_indicators: Optional[Dict[str, Tuple[str, str]]] = None) -> pd.DataFrame:
    """Unit x indicator table from per-vehicle sums, with vehicle-equivalents (veq = vehicle-km / unit length)."""
    uid = units.index
    rec = standard_columns(records[records['unit'].isin(uid)].reset_index(drop=True))
    uidx = uid.get_indexer(rec['unit'])
    d = unit_indicators(rec, len(uid), uidx, indicators, ratio_indicators)
    out = pd.DataFrame({f: d[f] for f in indicators}, index=uid)
    out['veq'] = d['km'] / units['length_km'].to_numpy(float)
    return out


def hdc(hdr: np.ndarray, veq: np.ndarray, days: float) -> np.ndarray:
    """Harsh decelerations per km and day: HDR x vehicle-equivalents / 100 / days."""
    return np.asarray(hdr, float) * np.asarray(veq, float) / 100.0 / float(days)


def ehc(hdr: np.ndarray, daily_volume: np.ndarray) -> np.ndarray:
    """Volume-expanded harsh-deceleration count per km and day: HDR x daily volume / 100."""
    return np.asarray(hdr, float) * np.asarray(daily_volume, float) / 100.0

# -*- coding: utf-8 -*-
"""Run the split-half check and its companion checks on the example data in examples/data/.

1. Steps 1-8 of Algorithm A.1 with all probe vehicles at 1 s (Eq. (1) fitted to the crash counts gives the working
   weights and the coefficients).
2. A reduced fleet: a quarter of the vehicles retained at random (coefficients refitted on the reduced data).
3. The check executed against held-out vehicles at 1 s and 2 s (executed.executed_check), with the diagonal Sigma_U of
   Eq. (3) and with the full Sigma_U.
4. Step 9 for a 2 s interval: calibration slope on a 1 Hz validation subset of units.
5. Step 7: SIMEX correction of the HDR coefficient for a quarter of the vehicles, averaged over ten draws.

Writes examples/example_output.txt, or the file given with --out.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from probe_reliability import (calibration_slope, executed_check, indicators_table, simex_nb2,  # noqa: E402
                               split_half_check, sums_from_points, thin_records)

DATA = os.path.join(HERE, 'data')
WEEKS = 13


def load():
    if not os.path.exists(os.path.join(DATA, 'points.csv.gz')):
        raise SystemExit('example data not found in examples/data/')
    pts = pd.read_csv(os.path.join(DATA, 'points.csv.gz'))
    units = pd.read_csv(os.path.join(DATA, 'units.csv'), index_col=0)
    design = pd.read_csv(os.path.join(DATA, 'design.csv'), index_col=0)
    crashes = pd.read_csv(os.path.join(DATA, 'crashes.csv'), index_col=0)['crashes']
    return pts, units, design, crashes


def main(out_path=None):
    pts, units, design, crashes = load()
    out = []
    fmt = lambda v: '%.4g' % v                                  # noqa: E731
    off = np.log(units['length_km'].to_numpy())

    # 1. all vehicles at 1 s
    rec = sums_from_points(pts, 1.0)
    res = split_half_check(rec, units, design, WEEKS, crash_counts=crashes, seed=1)
    out += ['== 1. All vehicles, 1 s ==', res.summary(), '']
    units_w = units.assign(weight=res.weights)
    coef_all = res.settings['coef']

    # 2. a quarter of the vehicles; naive coefficients refitted on the reduced data
    rng = np.random.default_rng(7)
    veh = np.unique(pts['vehicle'])
    keep = set(veh[rng.random(len(veh)) < 0.25])
    rec_q = rec[rec['vehicle'].isin(keep)]
    res_q = split_half_check(rec_q, units_w, design, WEEKS, crash_counts=crashes, seed=1)
    out += ['== 2. A quarter of the vehicles, 1 s ==', res_q.summary(), '']

    # 3. executed against held-out vehicles, diagonal and full Sigma_U
    for tau in (1, 2):
        tabs = []
        for su in ('diagonal', 'full'):
            ex = executed_check(pts, units_w, design, WEEKS, retain_share=0.25, tau=tau, draws=4, seed=11, sigma_u=su)
            t = ex.groupby(['quantity', 'indicator'])[['split_half', 'held_out']].mean()
            tabs.append(t.rename(columns={'split_half': 'split_half_' + su}))
        tab = pd.concat([tabs[0]['split_half_diagonal'], tabs[1]['split_half_full'], tabs[0]['held_out']], axis=1)
        out += ['== 3. Executed check, retained share 0.25, %d s (mean of 4 draws) ==' % tau,
                tab.to_string(float_format=fmt), '']
    out += ['  The full Sigma_U includes the error covariances of indicators computed from the same records; list',
            '  predictions do not use Sigma_U. Under 2 s, held-out values for acceleration-based indicators include',
            '  the rescaling that the halves cannot see (step 9).', '']

    # 4. step 9: calibration slope of the 2 s indicators on a 1 Hz validation subset
    val = units.index[np.random.default_rng(3).random(len(units)) < 0.3]
    ind1 = indicators_table(rec, units)
    rec2 = sums_from_points(thin_records(pts, 2, seed=5), 2.0)
    ind2 = indicators_table(rec2, units)
    res2 = split_half_check(rec2, units_w, design, WEEKS, coef=coef_all, seed=1, sampling_interval_s=2)
    out.append('== 4. Calibration slope at 2 s, %d validation units ==' % len(val))
    for f in ('MSP', 'VSD', 'ASD', 'HDR'):
        c = calibration_slope(ind2.loc[val, f], ind1.loc[val, f], res2.sigma2.loc[val, f])
        out.append('  %-4s gamma1 %.3f (SE %.3f)  identity rejected: %s  estimable: %s  sigma2_eps %.3g'
                   % (f, c.gamma1, c.se_gamma1, c.rejects_identity(), c.estimable, c.sigma2_eps))
    out.append('  split-half reliability at 2 s: ' + ', '.join('%s %.3f' % kv for kv in res2.lam.items()))
    out.append('')

    # 5. step 7: SIMEX for HDR with a quarter of the vehicles, ten draws
    rows = []
    for d in range(10):
        kd = set(veh[np.random.default_rng(100 + d).random(len(veh)) < 0.25])
        rd = split_half_check(rec[rec['vehicle'].isin(kd)], units_w, design, WEEKS, seed=d)
        used = rd.settings['used']
        if 'HDR' not in used:
            continue
        sx = simex_nb2(crashes, design, rd.indicators[used], rd.sigma2, ['HDR'], offset=off, half_diff=rd.half_diff,
                       half_diff_cols=used, flags=rd.flags, B=20, seed=d)
        rows.append(dict(naive=sx.naive_coef['HDR'], simex=sx.coef['HDR'], lam=rd.lam['HDR']))
    r = pd.DataFrame(rows)
    out += ['== 5. SIMEX for HDR, a quarter of the vehicles (mean over %d draws) ==' % len(r),
            '  split-half reliability %.3f   naive %.4f   SIMEX %.4f   all vehicles %.4f'
            % (r['lam'].mean(), r['naive'].mean(), r['simex'].mean(), coef_all['HDR']), '']
    text = '\n'.join(out)
    if out_path is None:
        out_path = os.path.join(HERE, 'example_output.txt')
    open(out_path, 'w', encoding='utf-8').write(text + '\n')
    print(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    main(ap.parse_args().out)

# -*- coding: utf-8 -*-
"""Command line: python -m probe_reliability RECORDS UNITS DESIGN PERIOD_WEEKS [options]

RECORDS is a CSV of 1 Hz records (vehicle, unit, pass, t, speed) or of per-vehicle sums (vehicle, unit, km, kmv, kmv2,
kma2, km_acc, hd_events); UNITS and DESIGN are CSVs indexed by unit (first column).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .indicators import sums_from_points
from .split_half import split_half_check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='probe_reliability', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('records')
    ap.add_argument('units')
    ap.add_argument('design')
    ap.add_argument('period_weeks', type=float)
    ap.add_argument('--coef', help='CSV of fitted coefficients (index = design columns and indicators)')
    ap.add_argument('--crashes', help='CSV of crash counts per unit (index = unit), used when UNITS has no weight')
    ap.add_argument('--tau', type=float, default=1.0, help='sampling interval of the records in s (default 1)')
    ap.add_argument('--J', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lam-req', type=float, default=0.9)
    ap.add_argument('--top-share', type=float, default=0.10)
    a = ap.parse_args(argv)
    rec = pd.read_csv(a.records)
    if 'speed' in rec.columns and 'km' not in rec.columns:
        rec = sums_from_points(rec, a.tau)
    units = pd.read_csv(a.units, index_col=0)
    design = pd.read_csv(a.design, index_col=0)
    coef = pd.read_csv(a.coef, index_col=0).iloc[:, 0] if a.coef else None
    crashes = pd.read_csv(a.crashes, index_col=0).iloc[:, 0] if a.crashes else None
    res = split_half_check(rec, units, design, a.period_weeks, coef=coef, crash_counts=crashes, J=a.J, seed=a.seed,
                           lam_req=a.lam_req, top_share=a.top_share, sampling_interval_s=a.tau)
    print(res.summary())
    print('\nsigma2 (Eq. 2, centred):')
    print(res.sigma2.to_string(float_format=lambda v: '%.6g' % v))
    print('\nunit reliability:')
    print(res.unit_rel.round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())

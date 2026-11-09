import numpy as np
import pandas as pd
import pytest

from probe_reliability import split_half_check


def _network(n_units=150, n_veh=40, s=2.0, seed=0, route_groups=3):
    """Each unit is travelled by its own vehicles; a vehicle's pass mean speed is unit mean + N(0, s^2)."""
    rng = np.random.default_rng(seed)
    uid = pd.Index(['u%03d' % i for i in range(n_units)], name='unit')
    mean = rng.normal(25, 1.5, n_units)
    rows = []
    for i, u in enumerate(uid):
        v = mean[i] + rng.normal(0, s, n_veh)
        for k in range(n_veh):
            rows.append(dict(vehicle='%s-%d' % (u, k), unit=u, km=1.0, kmv=v[k], kmv2=v[k] ** 2, kma2=float(rng.uniform(0.05, 0.2)),
                             km_acc=1.0, hd_events=int(rng.poisson(0.5))))
    units = pd.DataFrame({'length_km': 1.0, 'group': ['g%d' % (i % route_groups) for i in range(n_units)],
                          'weight': 1.0}, index=uid)
    design = pd.DataFrame({'const': 1.0}, index=uid)
    return pd.DataFrame(rows), units, design, mean


def test_eq2_estimates_the_sampling_variance():
    rec, units, design, _ = _network()
    res = split_half_check(rec, units, design, 13, J=40, seed=1, indicators=['MSP'])
    ratio = res.sigma2['MSP'].mean() / (2.0 ** 2 / 40)
    assert 0.85 < ratio < 1.15


def test_reliability_one_without_error():
    rng = np.random.default_rng(3)
    uid = pd.Index(['u%02d' % i for i in range(30)], name='unit')
    mean = rng.normal(25, 1.5, 30)
    rec = pd.DataFrame([dict(vehicle='v%d' % k, unit=u, km=1.0, kmv=mean[i], kmv2=mean[i] ** 2, kma2=0.1, km_acc=1.0,
                             hd_events=0) for i, u in enumerate(uid) for k in range(6)])
    units = pd.DataFrame({'length_km': 1.0, 'group': 'g', 'weight': 1.0}, index=uid)
    design = pd.DataFrame({'const': 1.0}, index=uid)
    res = split_half_check(rec, units, design, 4, seed=1, indicators=['MSP'], coef=pd.Series({'const': 0.0, 'MSP': 0.1}))
    assert abs(res.lam['MSP'] - 1.0) < 1e-12
    assert abs(res.pred_ratio['MSP'] - 1.0) < 1e-9


def test_reliability_matches_variance_share():
    rec, units, design, mean = _network(seed=4)
    res = split_half_check(rec, units, design, 13, seed=2, indicators=['MSP'])
    expect = np.var(mean, ddof=1) / (np.var(mean, ddof=1) + 2.0 ** 2 / 40)
    assert abs(res.lam['MSP'] - expect) < 0.03


def test_seed_reproducible_and_assignment():
    rec, units, design, _ = _network(n_units=20, seed=5)
    a = split_half_check(rec, units, design, 13, seed=9, indicators=['MSP'])
    b = split_half_check(rec, units, design, 13, seed=9, indicators=['MSP'])
    pd.testing.assert_frame_equal(a.sigma2, b.sigma2)
    veh = np.unique(rec['vehicle'])
    assign = np.zeros((20, len(veh)), bool)
    assign[:, ::2] = True
    c = split_half_check(rec, units, design, 13, assignment=assign, J=20, indicators=['MSP'])
    assert np.isfinite(c.lam['MSP'])


def test_few_vehicle_units_flagged_and_rule_f():
    rec, units, design, _ = _network(n_units=60, seed=6)
    rec = rec[~((rec['unit'] == 'u000') & (rec['vehicle'] != 'u000-0') & (rec['vehicle'] != 'u000-1')
                & (rec['vehicle'] != 'u000-2'))]
    res = split_half_check(rec, units, design, 13, seed=1, indicators=['MSP'])
    assert bool(res.flags.loc['u000', 'MSP'])
    assert res.sigma2.loc['u000', 'MSP'] == pytest.approx(res.sigma2['MSP'].drop('u000').max())
    assert res.usable['MSP']


def test_indicator_dropped_when_too_many_units_flagged():
    rec, units, design, _ = _network(n_units=40, seed=7)
    few = units.index[:5]
    rec = rec[~(rec['unit'].isin(few) & ~rec['vehicle'].str.endswith(('-0', '-1')))]
    with pytest.raises(ValueError):
        split_half_check(rec, units, design, 13, seed=1, indicators=['MSP'])


def test_lists_and_working_weights_from_crashes():
    rec, units, design, mean = _network(n_units=80, seed=8)
    rng = np.random.default_rng(1)
    crashes = pd.Series(rng.poisson(np.exp(1 + 0.1 * (mean - 25))), index=units.index)
    res = split_half_check(rec, units.drop(columns='weight').assign(Q=10000.0), design, 13, seed=1,
                           crash_counts=crashes)
    assert {'risk score', 'HDR', 'HDC', 'EHC'} <= set(res.lists.index)
    assert ((res.lists >= 0) & (res.lists <= 1)).all()
    assert (res.weights > 0).all()
    # lists ranked by raw values use uncentred variances, which are not smaller on average
    assert res.sigma2_uncentred['HDR'].mean() >= res.sigma2['HDR'].mean() * 0.99


def test_requirement_consistency():
    rec, units, design, _ = _network(seed=10)
    res = split_half_check(rec, units, design, 13, seed=1, indicators=['MSP'])
    lam, k, nbar = res.lam['MSP'], res.k['MSP'], res.nbar
    assert abs(nbar / (nbar + k) - lam) < 1e-9
    assert abs(res.n_req['MSP'] / (res.n_req['MSP'] + k) - 0.9) < 1e-9


def test_full_sigma_u_equals_diagonal_for_one_indicator_and_runs():
    rec, units, design, _ = _network(n_units=60, seed=11)
    a = split_half_check(rec, units, design, 13, seed=1, indicators=['MSP'])
    b = split_half_check(rec, units, design, 13, seed=1, indicators=['MSP'], sigma_u='full')
    assert abs(a.lam['MSP'] - b.lam['MSP']) < 1e-12
    c = split_half_check(rec, units, design, 13, seed=1, sigma_u='full')
    assert all(np.isfinite(c.lam))

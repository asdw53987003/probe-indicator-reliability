import numpy as np
import pandas as pd

from probe_reliability import (calibration_slope, held_out_reliability, simex_nb2, slope_heterogeneity,
                               unit_error_variances)


def test_calibration_slope_detects_rescaling():
    rng = np.random.default_rng(0)
    idx = pd.Index(range(60))
    x = pd.Series(rng.normal(10, 3, 60), index=idx)
    y = 0.5 * x + rng.normal(0, 0.2, 60)
    c = calibration_slope(y, x)
    assert abs(c.gamma1 - 0.5) < 0.05
    assert c.rejects_identity()
    assert c.estimable
    assert abs(np.mean(c.rescale(y) - x)) < 0.1
    c2 = calibration_slope(x + rng.normal(0, 0.2, 60), x)
    assert c2.p_identity > 0.001


def test_excess_residual_variance():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(10, 3, 200))
    y = x + rng.normal(0, 1.0, 200)
    small = pd.Series(0.25, index=x.index)
    c = calibration_slope(y, x, split_half_var=small)
    assert c.sigma2_eps > 0.5
    c0 = calibration_slope(y, x, split_half_var=pd.Series(1.0, index=x.index))
    assert c0.sigma2_eps == 0.0


def test_slope_heterogeneity():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(10, 3, 120))
    g = pd.Series(np.repeat(['a', 'b'], 60))
    y = np.where(g == 'a', 0.5, 0.9) * x + rng.normal(0, 0.2, 120)
    r = slope_heterogeneity(pd.Series(y), x, g)
    assert r['p_value'] < 0.001
    assert abs(r['slopes']['a'] - 0.5) < 0.05


def test_held_out_reliability_single_indicator():
    rng = np.random.default_rng(3)
    n = 4000
    idx = pd.Index(range(n))
    t = rng.normal(0, 1, n)
    deg = pd.DataFrame({'X': t + rng.normal(0, 0.5, n)}, index=idx)
    ho = pd.DataFrame({'X': t + rng.normal(0, 0.8, n)}, index=idx)
    design = pd.DataFrame({'const': 1.0}, index=idx)
    r = held_out_reliability(deg, ho, design, pd.Series(1.0, index=idx))
    assert abs(r['X'] - 1 / 1.25) < 0.03


def test_unit_error_variances():
    s = pd.Series([1.0, 2.0])
    assert np.allclose(unit_error_variances(s, 0.5, 1.0), [8.0, 12.0])


def test_simex_reduces_attenuation():
    rng = np.random.default_rng(4)
    n = 1500
    idx = pd.Index(range(n))
    t = rng.normal(0, 1, n)
    s2 = 0.6
    w = t + rng.normal(0, np.sqrt(s2), n)
    mu = np.exp(1.0 + 0.5 * t)
    a = 0.2
    y = pd.Series(rng.negative_binomial(1 / a, 1 / (1 + a * mu)), index=idx)
    design = pd.DataFrame({'const': 1.0}, index=idx)
    ind = pd.DataFrame({'W': w}, index=idx)
    sig = pd.DataFrame({'W': s2}, index=idx)
    r = simex_nb2(y, design, ind, sig, ['W'], B=20, seed=1)
    naive, corr = r.naive_coef['W'], r.coef['W']
    assert naive < 0.5 * 0.8
    assert abs(corr - 0.5) < abs(naive - 0.5)
    hd = rng.normal(0, 2 * np.sqrt(s2), (20, 1, n))           # half differences with variance 4 sigma^2
    r2 = simex_nb2(y, design, ind, sig, ['W'], half_diff=hd, half_diff_cols=['W'], B=20, seed=1)
    assert abs(r2.coef['W'] - 0.5) < abs(naive - 0.5)

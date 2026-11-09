import numpy as np
import pandas as pd

from probe_reliability import (capture, fit_nb2, k_hat, list_overlap, n_required, predicted_overlap,
                               ranking_reliability, reliability_at, volume_summary, weeks_needed)


def test_predicted_overlap_limits_and_monotone():
    q = 0.1
    assert abs(predicted_overlap(1.0, q) - 1.0) < 1e-3
    assert abs(predicted_overlap(0.0, q) - q) < 1e-6
    vals = [predicted_overlap(l, q) for l in np.linspace(0, 0.99, 12)]
    assert all(np.diff(vals) > 0)


def test_predicted_overlap_matches_monte_carlo():
    rng = np.random.default_rng(0)
    lam, q, n = 0.7, 0.1, 400000
    x = rng.normal(size=n)
    xs = np.sqrt(lam) * x + np.sqrt(1 - lam) * rng.normal(size=n)
    z = np.quantile(x, 1 - q)
    zs = np.quantile(xs, 1 - q)
    mc = np.mean((x > z) & (xs > zs)) / q
    assert abs(predicted_overlap(lam, q) - mc) < 0.01


def test_ranking_reliability_and_overlap():
    x = np.arange(10, dtype=float)
    assert abs(ranking_reliability(x, np.zeros(10)) - 1.0) < 1e-12
    assert list_overlap(x, x, 3) == 1.0
    assert list_overlap(x, -x, 3) == 0.0
    assert capture({9, 8}, x, 2) == 1.0


def test_requirements_round_trip():
    k = k_hat(500.0, 0.6)
    assert abs(reliability_at(500.0, k) - 0.6) < 1e-12
    n = n_required(k, 0.9)
    assert abs(reliability_at(n, k) - 0.9) < 1e-12
    assert abs(weeks_needed(n, 500.0, 13) - n / (500.0 / 13)) < 1e-9
    w = np.array([1.0, 1.0, 2.0])
    v = np.array([100.0, 400.0, 5.0])
    assert abs(volume_summary(v, w) - 2.0 / (1 / 100 + 1 / 400)) < 1e-9


def test_nb2_recovers_coefficients():
    rng = np.random.default_rng(3)
    n = 3000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    off = np.log(rng.uniform(0.5, 1.5, n))
    mu = np.exp(off + 1.0 + 0.4 * X[:, 1])
    a = 0.3
    y = rng.negative_binomial(1 / a, 1 / (1 + a * mu))
    f = fit_nb2(y, X, off)
    assert f.converged
    assert abs(f.coef[1] - 0.4) < 4 * f.se[1]
    assert abs(f.alpha - a) < 0.08
    g = fit_nb2(y, X, off, alpha=a)
    assert np.allclose(g.working_weights, g.mu / (1 + a * g.mu))

# -*- coding: utf-8 -*-
"""Negative binomial (NB2) regression with a log link and an offset, Eq. (1) of the paper.

Var(y) = mu + alpha mu^2. With alpha fixed, the coefficients are fitted by iteratively reweighted least squares (IRLS);
alpha is estimated by maximising the profile log-likelihood. The working weights mu / (1 + alpha mu) enter the
attenuation matrix of Eq. (3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize, special


@dataclass
class NB2Fit:
    coef: np.ndarray
    se: np.ndarray
    alpha: float
    mu: np.ndarray
    loglik: float
    converged: bool

    @property
    def working_weights(self) -> np.ndarray:
        return self.mu / (1.0 + self.alpha * self.mu)


def nb2_loglik(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    y = np.asarray(y, float)
    if alpha <= 0:
        return float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu - special.gammaln(y + 1)))
    r = 1.0 / alpha
    return float(np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
                        + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def _irls(y, X, offset, alpha, beta0=None, max_iter=100, tol=1e-10):
    y = np.asarray(y, float)
    if beta0 is None:
        mu = y + 0.5
        eta = np.log(mu)
    else:
        eta = X @ beta0 + offset
        mu = np.exp(eta)
    beta = beta0
    converged = False
    for _ in range(max_iter):
        w = mu / (1.0 + alpha * mu)
        z = (eta - offset) + (y - mu) / mu
        sw = np.sqrt(w)
        new, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
        eta = X @ new + offset
        eta = np.clip(eta, -50, 50)
        mu = np.exp(eta)
        if beta is not None and np.max(np.abs(new - beta)) < tol * (1 + np.max(np.abs(new))):
            beta = new
            converged = True
            break
        beta = new
    return beta, mu, converged


def fit_nb2(y: np.ndarray, X: np.ndarray, offset: Optional[np.ndarray] = None, alpha: Optional[float] = None) -> NB2Fit:
    """Fit y ~ NB2(mu, alpha) with ln mu = offset + X beta. If `alpha` is None it is estimated."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    if alpha is None:
        beta_p, mu_p, _ = _irls(y, X, off, 0.0)

        def nll(log_a):
            b, mu, _ = _irls(y, X, off, float(np.exp(log_a)), beta_p, max_iter=50)
            return -nb2_loglik(y, mu, float(np.exp(log_a)))
        res = optimize.minimize_scalar(nll, bounds=(-12.0, 4.0), method='bounded', options=dict(xatol=1e-6))
        alpha = float(np.exp(res.x))
        beta, mu, conv = _irls(y, X, off, alpha, beta_p)
    else:
        beta, mu, conv = _irls(y, X, off, float(alpha))
    w = mu / (1.0 + alpha * mu)
    cov = np.linalg.pinv((X * w[:, None]).T @ X)
    return NB2Fit(coef=beta, se=np.sqrt(np.maximum(np.diag(cov), 0.0)), alpha=float(alpha), mu=mu,
                  loglik=nb2_loglik(y, mu, alpha), converged=conv)

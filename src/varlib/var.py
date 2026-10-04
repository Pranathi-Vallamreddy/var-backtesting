"""1-day VaR and Expected Shortfall.

Every function returns (var, es) in currency units, losses positive.
`alpha` is the confidence level (0.99), so the tail probability is 1 - alpha.
"""
from functools import lru_cache

import numpy as np
from scipy import stats


def var_hist(r_p, alpha, value):
    """Historical simulation on the empirical portfolio return distribution.

    Quantile uses linear interpolation between order statistics
    (numpy method="linear", Hyndman-Fan type 7).
    """
    r_p = np.asarray(r_p)
    q = np.quantile(r_p, 1 - alpha, method="linear")
    es = -r_p[r_p <= q].mean() * value
    return -q * value, es


def _moments(returns, w):
    mu = np.asarray(returns.mean())
    cov = np.atleast_2d(np.cov(np.asarray(returns), rowvar=False))
    return w @ mu, np.sqrt(w @ cov @ w)


def var_param(returns, w, alpha, value):
    """Gaussian variance-covariance VaR with mu_p = w'mu, sigma_p^2 = w'Sigma w."""
    mu_p, sigma_p = _moments(returns, w)
    z = stats.norm.ppf(alpha)
    var = (z * sigma_p - mu_p) * value
    es = (sigma_p * stats.norm.pdf(z) / (1 - alpha) - mu_p) * value
    return var, es


@lru_cache(maxsize=4)
def _normal_draws(n_sims, n_assets, seed):
    # cached so a backtest draws once instead of on every day and level;
    # callers must not modify the array
    return np.random.default_rng(seed).standard_normal((n_sims, n_assets))


def var_mc(returns, w, alpha, value, n_sims=50_000, seed=42):
    """Monte Carlo VaR from a multivariate normal fitted to the window.

    Draws x = mu + L z with L the Cholesky factor of the covariance matrix
    (not the correlation matrix, which would lose the asset volatilities),
    then reads VaR/ES off the simulated portfolio returns as in var_hist.
    """
    mu = np.asarray(returns.mean())
    cov = np.atleast_2d(np.cov(np.asarray(returns), rowvar=False))
    L = np.linalg.cholesky(cov)
    z = _normal_draws(n_sims, len(mu), seed)
    x = mu + z @ L.T
    return var_hist(x @ w, alpha, value)


def var_t(returns, w, alpha, value, nu=5.0):
    """Student-t variance-covariance VaR.

    The t is scaled so its variance matches sigma_p^2, i.e. scale
    s = sigma_p * sqrt((nu - 2) / nu); fatter tails at the same volatility.
    Pass nu=None to fit the degrees of freedom by MLE on the portfolio returns.
    """
    mu_p, sigma_p = _moments(returns, w)
    if nu is None:
        nu = stats.t.fit(np.asarray(returns) @ w)[0]
    if nu <= 2:
        raise ValueError(f"nu={nu:.2f}: the t has no finite variance for nu <= 2")
    s = sigma_p * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    var = (s * q - mu_p) * value
    # closed-form ES of a standard t, McNeil, Frey & Embrechts (2015), ch. 2
    es_std = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    es = (s * es_std - mu_p) * value
    return var, es


def ewma_sigma(r_p, lam=0.94):
    """RiskMetrics EWMA volatility, sigma_t^2 = lam sigma_{t-1}^2 + (1 - lam) r_{t-1}^2.

    Returns n + 1 values: sigma[t] is the forecast for day t made with returns
    up to t - 1, so sigma[-1] is the forecast for the day after the window.
    Seeded with the window's sample variance; with lam = 0.94 the seed's
    weight after 500 days is 0.94^500, about 4e-14.
    """
    r_p = np.asarray(r_p)
    var = np.empty(r_p.size + 1)
    var[0] = r_p.var()
    for t, r in enumerate(r_p):
        var[t + 1] = lam * var[t] + (1 - lam) * r * r
    return np.sqrt(var)


def var_ewma(r_p, alpha, value, lam=0.94):
    """Gaussian VaR with RiskMetrics EWMA volatility and zero mean.

    EWMA on the portfolio return gives the same sigma_p as an EWMA covariance
    matrix, w' Sigma_ewma w, since both are the same weighted sum of (w'r)^2.
    """
    sigma = ewma_sigma(r_p, lam)[-1]
    z = stats.norm.ppf(alpha)
    return z * sigma * value, sigma * stats.norm.pdf(z) / (1 - alpha) * value


def var_fhs(r_p, alpha, value, lam=0.94):
    """Filtered historical simulation (Hull and White 1998).

    Each past return is divided by the EWMA volatility at the time and
    multiplied by today's forecast, so the scenarios keep their empirical
    shape (fat tails, skew) but are scaled to current volatility.
    """
    sigma = ewma_sigma(r_p, lam)
    scenarios = np.asarray(r_p) / sigma[:-1] * sigma[-1]
    return var_hist(scenarios, alpha, value)


def var_all(returns, w, alpha, value, n_sims=50_000, seed=42):
    """(var, es) for every method on the same estimation window."""
    r_p = np.asarray(returns) @ w
    return {
        "hist": var_hist(r_p, alpha, value),
        "param": var_param(returns, w, alpha, value),
        "t": var_t(returns, w, alpha, value),
        # a fixed seed reuses the same draws every call (common random
        # numbers), so in a backtest day-to-day changes in MC VaR come from
        # the data rather than simulation noise
        "mc": var_mc(returns, w, alpha, value, n_sims, seed),
        "ewma": var_ewma(r_p, alpha, value),
        "fhs": var_fhs(r_p, alpha, value),
    }

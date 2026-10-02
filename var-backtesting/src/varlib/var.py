"""1-day VaR and Expected Shortfall.

Every function returns (var, es) in currency units, losses positive.
`alpha` is the confidence level (0.99), so the tail probability is 1 - alpha.
"""
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


def var_t(returns, w, alpha, value, nu=5.0):
    """Student-t variance-covariance VaR.

    The t is scaled so its variance matches sigma_p^2, i.e. scale
    s = sigma_p * sqrt((nu - 2) / nu); fatter tails at the same volatility.
    Pass nu=None to fit the degrees of freedom by MLE on the portfolio returns.
    """
    mu_p, sigma_p = _moments(returns, w)
    if nu is None:
        nu = stats.t.fit(np.asarray(returns) @ w)[0]
    s = sigma_p * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    var = (s * q - mu_p) * value
    # closed-form ES of a standard t, see McNeil, Frey & Embrechts (2015) ex. 2.15
    es_std = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    es = (s * es_std - mu_p) * value
    return var, es

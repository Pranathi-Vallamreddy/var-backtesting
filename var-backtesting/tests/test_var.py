import numpy as np
import pandas as pd
import pytest
from scipy import stats

from varlib.var import var_hist, var_mc, var_param, var_t

V = 1e6


def gaussian_returns(n, sigma=0.01, seed=0):
    """Single-asset normal returns rescaled to sample mean 0 and sample sd sigma exactly."""
    x = np.random.default_rng(seed).standard_normal(n)
    x = (x - x.mean()) / x.std(ddof=1) * sigma
    return pd.DataFrame({"a": x})


def test_param_var_matches_closed_form():
    r = gaussian_returns(10_000)
    var, es = var_param(r, np.array([1.0]), 0.99, V)
    assert var == pytest.approx(23_263.48, abs=0.01)  # 2.326348 * 0.01 * 1e6
    # Gaussian ES at 99%: phi(2.326) / 0.01 * sigma * V
    assert es == pytest.approx(26_652.14, abs=0.01)


def test_param_var_uses_portfolio_covariance():
    # two perfectly negatively correlated assets, equal weight: sigma_p = 0
    x = gaussian_returns(1_000)["a"]
    r = pd.DataFrame({"a": x, "b": -x})
    var, _ = var_param(r, np.array([0.5, 0.5]), 0.99, V)
    assert var == pytest.approx(0.0, abs=1e-6)


def test_hist_var_on_known_sample():
    # 100 returns -5.0%, -4.9%, ..., +4.9%. Linear interpolation puts the 1%
    # quantile 0.99 of the way from -5.0% to -4.9%: -4.901%.
    r = np.arange(-50, 50) / 1000
    var, es = var_hist(r, 0.99, V)
    assert var == pytest.approx(49_010)
    assert es == pytest.approx(50_000)  # only -5.0% lies at or below the quantile


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_methods_agree_on_gaussian_data(alpha):
    r = gaussian_returns(200_000, seed=1)
    w = np.array([1.0])
    var_p, es_p = var_param(r, w, alpha, V)
    var_h, es_h = var_hist(r["a"], alpha, V)
    var_m, es_m = var_mc(r, w, alpha, V, n_sims=200_000, seed=7)
    for var in (var_h, var_m):
        assert var == pytest.approx(var_p, rel=0.02)
    for es in (es_h, es_m):
        assert es == pytest.approx(es_p, rel=0.02)


def test_mc_converges_to_parametric():
    rng = np.random.default_rng(3)
    cov = np.array([[1.0, 0.6, 0.2], [0.6, 1.5, 0.3], [0.2, 0.3, 0.8]]) * 1e-4
    r = pd.DataFrame(rng.multivariate_normal([0.0005, 0.0002, 0.0], cov, size=2_000))
    w = np.array([0.5, 0.3, 0.2])
    var_p, _ = var_param(r, w, 0.99, V)
    errors = [abs(var_mc(r, w, 0.99, V, n_sims=n, seed=42)[0] / var_p - 1) for n in (1_000, 1_000_000)]
    assert errors[1] < 0.005
    assert errors[1] < errors[0]


def test_mc_is_reproducible_with_seed():
    r = gaussian_returns(500)
    w = np.array([1.0])
    assert var_mc(r, w, 0.99, V, seed=5) == var_mc(r, w, 0.99, V, seed=5)


def test_t_es_matches_numerical_integration():
    r = gaussian_returns(10_000)
    nu = 5.0
    var, es = var_t(r, np.array([1.0]), 0.99, V, nu=nu)
    s = 0.01 * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(0.99, nu)
    assert var == pytest.approx(s * q * V)
    tail_mean = stats.t.expect(lambda y: y, args=(nu,), lb=q, conditional=True)
    assert es == pytest.approx(s * tail_mean * V, rel=1e-6)


def test_t_var_exceeds_gaussian_at_99():
    # same variance, fatter tails: the t should give the larger 99% VaR and ES
    r = gaussian_returns(10_000)
    w = np.array([1.0])
    var_g, es_g = var_param(r, w, 0.99, V)
    var_t5, es_t5 = var_t(r, w, 0.99, V, nu=5)
    assert var_t5 > var_g and es_t5 > es_g

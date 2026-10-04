import numpy as np
import pandas as pd
import pytest

from varlib.data import log_returns
from varlib.portfolio import pnl, portfolio_returns, resolve_weights


def test_equal_weights_sum_to_one():
    w = resolve_weights("equal", ["A", "B", "C"])
    assert w == pytest.approx([1 / 3] * 3)
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("weights", [[0.5, 0.4, 0.2], [0.5, 0.5]])
def test_bad_weights_raise(weights):
    with pytest.raises(ValueError):
        resolve_weights(weights, ["A", "B", "C"])


def test_portfolio_return_is_weighted_sum():
    r = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.00]})
    r_p = portfolio_returns(r, np.array([0.25, 0.75]))
    assert r_p.tolist() == pytest.approx([0.25 * 0.01 + 0.75 * 0.03, 0.25 * -0.02])
    assert pnl(r_p, 1e6).tolist() == pytest.approx([25_000, -5_000])


def test_log_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    assert log_returns(prices)["A"].tolist() == pytest.approx([np.log(1.1), np.log(0.9)])

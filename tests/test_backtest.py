import numpy as np
import pandas as pd

from varlib.backtest import run_backtest
from varlib.stats_tests import coverage_table
from varlib.var import var_hist


def toy_returns(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=idx, columns=["A", "B"])


def test_forecast_uses_only_prior_window():
    r = toy_returns()
    w = np.array([0.5, 0.5])
    window = 60
    r.iloc[80] = -0.2  # a crash on day 80 must not be in day 80's forecast
    bt = run_backtest(r, w, 1e6, [0.99], window=window, n_sims=1_000)
    hist = bt[bt.method == "hist"].set_index("date")

    day = r.index[80]
    expected, _ = var_hist(r.iloc[80 - window:80].to_numpy() @ w, 0.99, 1e6)
    assert hist.loc[day, "var"] == expected
    assert hist.loc[day, "exception"]
    # the crash enters the estimation window from the next day on
    assert hist.loc[r.index[81], "var"] > 5 * hist.loc[day, "var"]


def test_backtest_shape_and_coverage_table():
    r = toy_returns()
    bt = run_backtest(r, np.array([0.5, 0.5]), 1e6, [0.95, 0.99], window=60, n_sims=1_000)
    assert len(bt) == (120 - 60) * 2 * bt.method.nunique()
    assert (bt.exception == (-bt.pnl > bt["var"])).all()

    table = coverage_table(bt)
    assert len(table) == 2 * bt.method.nunique()
    assert set(table.loc[table.alpha == 0.99, "basel_zone"]) <= {"green", "yellow", "red"}

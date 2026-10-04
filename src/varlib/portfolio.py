"""Portfolio aggregation.

Convention: the book is held at a constant notional V with fixed weights,
i.e. rebalanced daily back to target. Portfolio return is the weighted sum of
asset log returns, r_p = w'r, and daily P&L is V * r_p. Weighting log returns
is a first-order approximation of the true portfolio return; at daily horizons
the error is negligible next to the estimation error in the VaR itself.
"""
import numpy as np
import pandas as pd


def resolve_weights(weights, tickers) -> np.ndarray:
    n = len(tickers)
    if weights == "equal" or weights == ["equal"]:
        return np.full(n, 1.0 / n)

    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"got {w.size} weights for {n} tickers")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights sum to {w.sum():.6f}, expected 1")
    return w


def portfolio_returns(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    return pd.Series(returns.to_numpy() @ w, index=returns.index, name="r_p")


def pnl(r_p: pd.Series, value: float) -> pd.Series:
    return (value * r_p).rename("pnl")

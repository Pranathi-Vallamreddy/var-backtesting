"""Rolling out-of-sample backtest.

For each day t the models are estimated on returns [t-W, t-1] only, and the
forecast is compared with the realised loss on day t. An exception is a day
where the realised loss exceeds the VaR forecast.
"""
import pandas as pd

from .var import var_all


def run_backtest(returns, w, value, alphas, window=500, n_sims=50_000, seed=42):
    r_p = returns.to_numpy() @ w
    rows = []
    for t in range(window, len(returns)):
        est = returns.iloc[t - window:t]
        pnl = value * r_p[t]
        for alpha in alphas:
            for method, (var, es) in var_all(est, w, alpha, value, n_sims, seed).items():
                rows.append((returns.index[t], method, alpha, var, es, pnl, -pnl > var))

    return pd.DataFrame(rows, columns=["date", "method", "alpha", "var", "es", "pnl", "exception"])

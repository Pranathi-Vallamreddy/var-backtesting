"""Rolling out-of-sample backtest.

For each day t the models are estimated on returns [t-W, t-1] only, and the
forecast is compared with the realised loss on day t. An exception is a day
where the realised loss exceeds the VaR forecast.
"""
import pandas as pd

from .var import var_hist, var_mc, var_param, var_t

METHODS = ["hist", "param", "t", "mc"]


def run_backtest(returns, w, value, alphas, window=500, n_sims=50_000, seed=42):
    r_p = pd.Series(returns.to_numpy() @ w, index=returns.index)
    rows = []
    for t in range(window, len(returns)):
        est = returns.iloc[t - window:t]
        est_p = r_p.iloc[t - window:t]
        pnl = value * r_p.iloc[t]
        for alpha in alphas:
            forecasts = {
                "hist": var_hist(est_p, alpha, value),
                "param": var_param(est, w, alpha, value),
                "t": var_t(est, w, alpha, value),
                # same seed every day: common random numbers, so day-to-day
                # changes in MC VaR come from the data, not simulation noise
                "mc": var_mc(est, w, alpha, value, n_sims, seed),
            }
            for method, (var, es) in forecasts.items():
                rows.append((returns.index[t], method, alpha, var, es, pnl, -pnl > var))

    return pd.DataFrame(rows, columns=["date", "method", "alpha", "var", "es", "pnl", "exception"])

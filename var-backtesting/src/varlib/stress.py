"""Stressed VaR: the same models calibrated to a fixed crisis window.

Current VaR uses the most recent `window` days; stressed VaR uses only the
stress window (default Sep 2008 - Mar 2009). The GFC window is ~145 trading
days, so 99% historical VaR there rests on one or two tail observations and
should be read as indicative. The COVID alternative is only ~50 days, where
99% historical VaR is essentially the worst day in the sample.
"""
import pandas as pd

from .var import var_all


def stressed_var(returns, stress_returns, w, value, alphas, window=500, n_sims=50_000, seed=42):
    current = returns.iloc[-window:]
    rows = []
    for alpha in alphas:
        cur = var_all(current, w, alpha, value, n_sims, seed)
        stressed = var_all(stress_returns, w, alpha, value, n_sims, seed)
        for method in cur:
            rows.append({
                "alpha": alpha, "method": method,
                "var": cur[method][0], "svar": stressed[method][0],
                "es": cur[method][1], "ses": stressed[method][1],
            })
    df = pd.DataFrame(rows)
    df["svar_ratio"] = df["svar"] / df["var"]
    return df

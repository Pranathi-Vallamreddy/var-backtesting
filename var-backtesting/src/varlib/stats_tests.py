"""Coverage tests for a VaR exception series.

Kupiec (1995) proportion-of-failures, Christoffersen (1998) independence and
conditional coverage, and the Basel traffic light.

Degenerate counts (x = 0, x = N, n11 = 0, ...) are handled by computing the
log-likelihoods with xlogy, which defines 0 * log(0) = 0. This is the limit of
the likelihood as the count goes to zero, so the statistics stay finite and
no special-case branches are needed.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def _bernoulli_ll(n0, n1, p):
    return xlogy(n0, 1 - p) + xlogy(n1, p)


def kupiec(exceptions, alpha):
    """LR_uc = -2 ln[L(p) / L(x/N)], chi2(1) under correct coverage."""
    e = np.asarray(exceptions, dtype=bool)
    n, x = e.size, int(e.sum())
    p = 1 - alpha
    # clipped at 0: rounding can give -1e-14 when x/N equals p exactly
    lr = max(-2 * (_bernoulli_ll(n - x, x, p) - _bernoulli_ll(n - x, x, x / n)), 0.0)
    return lr, stats.chi2.sf(lr, 1)


def transition_counts(exceptions):
    e = np.asarray(exceptions, dtype=int)
    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    return n00, n01, n10, n11


def christoffersen(exceptions):
    """LR_ind: first-order Markov alternative vs independence, chi2(1)."""
    n00, n01, n10, n11 = transition_counts(exceptions)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    # an empty row (e.g. no exceptions, so n10 + n11 = 0) contributes nothing
    # to the likelihood, so its transition probability can be set to anything
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    ll_null = _bernoulli_ll(n00 + n10, n01 + n11, pi)
    ll_alt = _bernoulli_ll(n00, n01, pi01) + _bernoulli_ll(n10, n11, pi11)
    lr = max(-2 * (ll_null - ll_alt), 0.0)
    return lr, stats.chi2.sf(lr, 1)


def conditional_coverage(exceptions, alpha):
    lr = kupiec(exceptions, alpha)[0] + christoffersen(exceptions)[0]
    return lr, stats.chi2.sf(lr, 2)


def basel_zone(x):
    """Zone for an exception count over 250 days at 99%."""
    if x <= 4:
        return "green"
    if x <= 9:
        return "yellow"
    return "red"


def coverage_table(bt, test_size=0.05):
    """One row per (alpha, method) from a run_backtest frame.

    loss_exc / es_exc compare the mean realised loss on exception days with
    the mean ES forecast on the same days; a ratio well above 1 means ES is
    too low. It is a diagnostic, not a formal ES backtest.

    The Basel zone follows the regulatory convention: 99% VaR, exceptions
    counted over the most recent 250 days. Because our backtest is much
    longer than 250 days we also report the worst rolling 250-day count.
    """
    rows = []
    for (alpha, method), g in bt.sort_values("date").groupby(["alpha", "method"]):
        e = g["exception"].to_numpy()
        lr_uc, p_uc = kupiec(e, alpha)
        lr_ind, p_ind = christoffersen(e)
        lr_cc, p_cc = conditional_coverage(e, alpha)
        row = {
            "alpha": alpha, "method": method, "n": e.size, "x": int(e.sum()),
            "expected": round(e.size * (1 - alpha), 1),
            "lr_uc": lr_uc, "p_uc": p_uc,
            "lr_ind": lr_ind, "p_ind": p_ind,
            "lr_cc": lr_cc, "p_cc": p_cc,
            "reject_cc": p_cc < test_size,
            # ES check: if ES is right, the average loss on exception days
            # should be close to the average ES forecast on those days
            "loss_exc": -g.loc[g["exception"], "pnl"].mean(),
            "es_exc": g.loc[g["exception"], "es"].mean(),
        }
        if np.isclose(alpha, 0.99):
            rolling = pd.Series(e).rolling(250).sum()
            row["x_last250"] = int(e[-250:].sum())
            row["basel_zone"] = basel_zone(row["x_last250"])
            row["x_worst250"] = int(rolling.max())
            row["basel_zone_worst"] = basel_zone(row["x_worst250"])
        rows.append(row)
    return pd.DataFrame(rows)

import argparse
from pathlib import Path

import pandas as pd
import yaml

from . import plotting
from .backtest import run_backtest
from .data import load_prices, log_returns
from .portfolio import portfolio_returns, resolve_weights
from .stats_tests import coverage_table
from .stress import stressed_var
from .var import var_all

RESULTS = Path("results")


def load(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg["var"]["horizon_days"] != 1:
        raise SystemExit("only a 1-day horizon is implemented (var.horizon_days: 1)")
    tickers = cfg["portfolio"]["tickers"]
    w = resolve_weights(cfg["portfolio"]["weights"], tickers)
    returns = log_returns(load_prices(tickers, cfg["start"], cfg["end"]))
    return cfg, tickers, w, returns


def cmd_var(cfg, tickers, w, returns):
    window = cfg["backtest"]["window"]
    est = returns.iloc[-window:]
    rows = []
    for alpha in cfg["var"]["confidence_levels"]:
        for method, (var, es) in var_all(est, w, alpha, cfg["portfolio"]["value"],
                                         cfg["monte_carlo"]["n_sims"], cfg["monte_carlo"]["seed"]).items():
            rows.append({"alpha": alpha, "method": method, "var": var, "es": es})
    print(f"1-day VaR/ES, USD, estimated on {est.index[0].date()} to {est.index[-1].date()} ({window} days)\n")
    print(pd.DataFrame(rows).to_string(index=False, formatters={"var": "{:,.0f}".format, "es": "{:,.0f}".format}))


def cmd_backtest(cfg, tickers, w, returns):
    value = cfg["portfolio"]["value"]
    bt = run_backtest(returns, w, value, cfg["var"]["confidence_levels"], cfg["backtest"]["window"],
                      cfg["monte_carlo"]["n_sims"], cfg["monte_carlo"]["seed"])
    table = coverage_table(bt)

    RESULTS.mkdir(exist_ok=True)
    bt.to_csv(RESULTS / "backtest.csv", index=False)
    table.to_csv(RESULTS / "coverage.csv", index=False)
    plotting.plot_var_vs_loss(bt, RESULTS / "var_vs_loss_99.png")
    plotting.plot_rolling_exceptions(bt, RESULTS / "rolling_exceptions_99.png")
    plotting.plot_return_tails(portfolio_returns(returns, w), RESULTS / "return_tails.png")

    print(f"Backtest {bt.date.min().date()} to {bt.date.max().date()}, "
          f"{bt.date.nunique()} forecasts, window {cfg['backtest']['window']} days\n")
    count = lambda v: "-" if pd.isna(v) else f"{v:.0f}"
    print(table.to_string(index=False, na_rep="-", float_format="{:.3f}".format,
                          formatters={"alpha": "{:.2f}".format, "x_last250": count, "x_worst250": count}))
    print(f"\nBasel zones: 99% VaR, 250-day convention; backtest is {bt.date.nunique()} days, "
          "so both the last 250 days and the worst rolling 250 days are shown.")
    print(f"Figures and CSVs written to {RESULTS}/")


def cmd_stress(cfg, tickers, w, returns):
    start, end = cfg["stress"]["window"]
    stress_returns = log_returns(load_prices(tickers, start, end))
    df = stressed_var(returns, stress_returns, w, cfg["portfolio"]["value"], cfg["var"]["confidence_levels"],
                      cfg["backtest"]["window"], cfg["monte_carlo"]["n_sims"], cfg["monte_carlo"]["seed"])
    print(f"Current: last {cfg['backtest']['window']} days to {returns.index[-1].date()}. "
          f"Stressed: {start} to {end} ({len(stress_returns)} days).\n")
    print(df.to_string(index=False, formatters={
        **{c: "{:,.0f}".format for c in ("var", "svar", "es", "ses")},
        "svar_ratio": "{:.2f}".format,
    }))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="varlib", description="1-day VaR/ES, backtests and stressed VaR")
    parser.add_argument("--config", default="config/portfolio.yaml")
    parser.add_argument("command", choices=["var", "backtest", "stress"])
    args = parser.parse_args(argv)

    commands = {"var": cmd_var, "backtest": cmd_backtest, "stress": cmd_stress}
    commands[args.command](*load(args.config))


if __name__ == "__main__":
    main()

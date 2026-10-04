# var-backtesting

1-day Value-at-Risk and Expected Shortfall for a six-stock equity portfolio,
using historical simulation, variance-covariance (Gaussian and Student-t) and
Monte Carlo, with a rolling out-of-sample backtest and stressed VaR.

The portfolio is $1m equally weighted in AAPL, MSFT, JPM, XOM, JNJ and PG
(tech, banks, energy, healthcare, staples), 2015-2024. Everything is set in
`config/portfolio.yaml`. Formulas and references are in
[docs/methodology.md](docs/methodology.md).

## Install

```
pip install -e ".[test]"
pytest
```

Prices come from Yahoo Finance via `yfinance` and are cached in `data/`.

## Usage

```
python -m varlib.cli --config config/portfolio.yaml var        # current VaR/ES, all methods
python -m varlib.cli --config config/portfolio.yaml backtest   # coverage tests, figures to results/
python -m varlib.cli --config config/portfolio.yaml stress     # stressed vs current VaR
```

The backtest re-estimates every model on each of ~2,000 days and takes about a
minute, almost all of it Monte Carlo.

## Results

Rolling 500-day estimation window, forecasts from 2016-12-28 to 2024-12-31
(2,015 days). Expected exceptions: 100.8 at 95%, 20.2 at 99%.

| 99% VaR | Exceptions | Kupiec p | Christoffersen p | Cond. coverage p | Basel, last 250d | Basel, worst 250d |
|---|---|---|---|---|---|---|
| Historical | 29 | 0.063 | 0.007 | 0.005 | green (2) | red (12) |
| Student-t (nu=5) | 45 | <0.001 | <0.001 | <0.001 | green (2) | red (17) |
| Gaussian | 50 | <0.001 | <0.001 | <0.001 | green (3) | red (18) |
| Monte Carlo | 50 | <0.001 | <0.001 | <0.001 | green (3) | red (18) |

At 95% all four methods pass Kupiec (99-107 exceptions, p between 0.53 and
0.86) and all four fail independence (p < 0.001).

No method passes conditional coverage at 99%. Historical simulation is the
only one whose exception count is not rejected, and only just. The Gaussian
model has 2.5 times the expected exceptions, and on the days it is breached
the average loss is $34k against a forecast ES of $24k, so it understates
both the quantile and the tail beyond it. Monte Carlo matches Gaussian almost
exactly, as it should, since it samples from the same fitted normal. The
Student-t sits between the two. On 99% historical exception days the average
loss ($41k) is close to its ES ($40k).

![Realised loss vs 99% VaR](results/var_vs_loss_99.png)

Exceptions cluster in four episodes: February-April 2018, October-December
2018, February-March 2020 (10 of the Gaussian model's 50 fall in March 2020
alone), and April-October 2022. In each, volatility rose faster than a 500-day
equally weighted window can follow, so breaches arrive in runs and every
method fails the independence test. Historical VaR also shows the window's
ghosting effect: it steps up when March 2020 enters the sample and stays
there until those days roll out two years later, regardless of conditions in
2021.

![Rolling 250-day exception count](results/rolling_exceptions_99.png)

Over the last 250 days every method is in the Basel green zone, but each spent
long stretches in the red, so the last-250-day reading on its own overstates
how well these models perform.

![Return distribution vs normal](results/return_tails.png)

Stressed VaR, calibrated to Sep 2008 - Mar 2009 (145 days), compared with the
most recent 500 days:

| 99% | Current VaR | Stressed VaR | Ratio |
|---|---|---|---|
| Historical | 17,097 | 81,960 | 4.8 |
| Gaussian | 14,814 | 84,284 | 5.7 |
| Student-t | 16,678 | 94,156 | 5.6 |
| Monte Carlo | 14,704 | 83,993 | 5.7 |

The ratios are high partly because 2023-24 was a calm period. With 145
observations the 99% historical figure rests on one or two days.

## Assumptions

- Returns are i.i.d. within each estimation window; all observations get equal
  weight.
- Constant notional and fixed weights, rebalanced daily. Portfolio return is
  the weighted sum of log returns, a first-order approximation.
- Adjusted closes, so dividends are treated as reinvested.
- 1-day horizon only; no square-root-of-time scaling.
- Parametric and Monte Carlo use the sample mean. Over 500 days it is small
  next to sigma but not zero, and it slightly lowers VaR.

## Limitations

- No volatility model (EWMA or GARCH). This is the main reason the backtests
  fail independence, and the obvious next step.
- Equity only, linear positions. No options, so no gamma or vega, and
  delta-normal is exact here only because the book is linear.
- Gaussian and Monte Carlo VaR understate tail risk, as the results show. The
  Monte Carlo draws are multivariate normal, so it adds nothing beyond the
  Gaussian model here; it would matter with non-linear positions.
- The Student-t uses a fixed nu=5. An MLE fit on the full sample gives about
  2.8, close to the point where the variance is undefined, so the fixed value
  is a pragmatic choice rather than an estimate.
- No liquidity horizon, transaction costs, FX, or intraday risk.
- Stress windows are short (145 days for the GFC, about 50 for COVID), so
  stressed historical VaR is indicative only.
- Six names on one data source. Yahoo adjusted prices are fine for this but
  are not a vetted market data feed.

## References

See [docs/methodology.md](docs/methodology.md).

# Methodology

## Conventions

- Losses are positive. Confidence level alpha (0.95 or 0.99); tail probability
  p = 1 - alpha. Horizon one trading day.
- Log returns: r_t = ln(P_t / P_{t-1}), on adjusted closes.
- Portfolio return r_p,t = sum_i w_i r_i,t with fixed weights w (sum to 1).
- Portfolio value V is held constant (daily rebalancing to target weights).
  Daily P&L is V * r_p,t and realised loss is L_t = -V * r_p,t.
- Weighting log returns is a first-order approximation of the portfolio
  return, which is exactly linear in simple returns. At a one-day horizon the
  difference is well inside estimation error.

## Historical simulation

    VaR = -Q_p(r_p) * V
    ES  = -mean(r_p | r_p <= Q_p(r_p)) * V

Q_p is the empirical p-quantile with linear interpolation between order
statistics (`numpy.quantile(method="linear")`, type 7 in Hyndman and Fan 1996).

## Variance-covariance

From the window, mu (vector of mean returns) and Sigma (sample covariance,
ddof = 1):

    mu_p    = w' mu
    sigma_p = sqrt(w' Sigma w)
    z       = Phi^-1(alpha)                        (1.645 at 95%, 2.326 at 99%)
    VaR     = (z * sigma_p - mu_p) * V
    ES      = (sigma_p * phi(z) / (1 - alpha) - mu_p) * V

phi and Phi are the standard normal pdf and cdf (Jorion 2007).

**Student-t variant.** Returns are modelled as mu_p + s * T_nu with T_nu a
standard t. The scale is chosen so the variance matches the sample:
s = sigma_p * sqrt((nu - 2) / nu). With q = t_nu^-1(alpha) and f_nu the t pdf:

    VaR = (s * q - mu_p) * V
    ES  = (s * f_nu(q) / (1 - alpha) * (nu + q^2) / (nu - 1) - mu_p) * V

The ES expression is the standard result for the t (McNeil, Frey and
Embrechts 2015, ch. 2). Default nu = 5; `nu=None` fits it by maximum
likelihood on the window's portfolio returns.

## Monte Carlo

1. Estimate mu and Sigma from the window as above.
2. L = chol(Sigma), lower triangular, so L L' = Sigma. The covariance matrix
   is factored, not the correlation matrix, so the draws carry the asset
   volatilities.
3. Draw z ~ N(0, I) of size N x n (default N = 50,000) and set x = mu + L z.
4. Simulated portfolio returns w'x; VaR and ES as in historical simulation.

The generator is `numpy.random.default_rng(seed)` with the seed from config.
In the backtest the same seed is used every day (common random numbers), so
changes in MC VaR from one day to the next reflect the data, not sampling
noise. Since the model is multivariate normal, MC VaR converges to the
Gaussian variance-covariance figure as N grows; the tests check this.

## EWMA (RiskMetrics)

The methods above weight every day in the window equally. EWMA lets the
volatility estimate react to recent returns (J.P. Morgan/Reuters 1996):

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_p,{t-1}^2,   lambda = 0.94
    VaR = z * sigma_T * V
    ES  = sigma_T * phi(z) / (1 - alpha) * V

sigma_T is the forecast for the day after the window. The recursion is seeded
with the window's sample variance; after 500 days the seed carries weight
0.94^500, about 4e-14. Mean is taken as zero, as in RiskMetrics. Running the
recursion on the portfolio return gives the same sigma_p as an EWMA
covariance matrix, w' Sigma_ewma w, since both are the same weighted sum of
(w'r)^2; the tests check this.

## Filtered historical simulation

Hull and White (1998). Each return in the window is divided by the EWMA
volatility forecast for its own day and rescaled to the current forecast:

    r*_t = r_p,t * sigma_T / sigma_t

VaR and ES are then read off {r*_t} as in historical simulation. The
scenarios keep the empirical shape of the returns (fat tails, skew) but are
scaled to today's volatility, which addresses both failures of the plain
methods: equal weighting and the Gaussian tail.

## Backtest

For each day t from W to T - 1 (W = 500): estimate on returns [t - W, t - 1],
forecast VaR_t, observe L_t. An exception is L_t > VaR_t. Indicator I_t = 1
on exception days. N = number of forecasts, x = sum of I_t.

## Coverage tests

**Kupiec (1995) proportion of failures.** Under correct unconditional coverage
x ~ Binomial(N, p):

    LR_uc = -2 ln[ (1-p)^(N-x) p^x / ( (1-x/N)^(N-x) (x/N)^x ) ]  ~ chi2(1)

**Christoffersen (1998) independence.** With n_ij the number of days where
I_{t-1} = i and I_t = j:

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11)

    LR_ind = -2 ln[ (1-pi)^(n00+n10) pi^(n01+n11)
                    / ( (1-pi01)^n00 pi01^n01 (1-pi11)^n10 pi11^n11 ) ]  ~ chi2(1)

**Conditional coverage.** LR_cc = LR_uc + LR_ind ~ chi2(2).

p-values are chi2 survival functions; rejection at 5%.

**Degenerate cases.** When x = 0, x = N, n11 = 0 or a row of the transition
matrix is empty, some terms are 0 * ln(0). These are evaluated as 0
(`scipy.special.xlogy`), the limit of the likelihood as the count goes to
zero, so the statistics stay finite. An empty row contributes nothing to the
likelihood, so its transition probability is set to 0 without effect.
Statistics are clipped at 0 to remove floating-point residue of order 1e-14.

## ES diagnostic

For each method the coverage table reports the mean realised loss on
exception days and the mean ES forecast on the same days. If ES is right the
two should be close; a ratio well above 1 means ES understates the tail. This
is a sanity check in the spirit of Acerbi and Szekely (2014), not their formal
test, which needs a simulated null distribution.

## Basel traffic light

Basel Committee (1996): 99% VaR, 250 trading days of exceptions.

| Zone | Exceptions |
|---|---|
| green | 0-4 |
| yellow | 5-9 |
| red | 10 or more |

This backtest runs about 2,000 days, so the zone is reported both for the most
recent 250 days (the regulatory convention) and for the worst rolling 250-day
window.

## Stressed VaR

The four unconditional models (historical, Gaussian, Student-t, Monte
Carlo), estimated only on a fixed stress window (default
2008-09-01 to 2009-03-31; alternative 2020-02-15 to 2020-04-30), compared
with VaR estimated on the most recent W days. EWMA and FHS are left out:
they condition on volatility at the end of the window, a point-in-time
reading rather than a calibration to the period. This follows the idea of the
Basel 2.5 stressed VaR charge, though not its 10-day, 12-month calibration.

## References

- Acerbi, C. and Szekely, B. (2014). Backtesting Expected Shortfall.
  *Risk*, December 2014.
- Basel Committee on Banking Supervision (1996). *Supervisory Framework for
  the Use of "Backtesting" in Conjunction with the Internal Models Approach
  to Market Risk Capital Requirements.* Bank for International Settlements.
- Christoffersen, P. F. (1998). Evaluating Interval Forecasts. *International
  Economic Review*, 39(4), 841-862.
- Hyndman, R. J. and Fan, Y. (1996). Sample Quantiles in Statistical
  Packages. *The American Statistician*, 50(4), 361-365.
- Hull, J. and White, A. (1998). Incorporating Volatility Updating into the
  Historical Simulation Method for Value-at-Risk. *Journal of Risk*, 1(1),
  5-19.
- J.P. Morgan/Reuters (1996). *RiskMetrics - Technical Document*, 4th ed.
- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial
  Risk*, 3rd ed. McGraw-Hill.
- Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk
  Measurement Models. *Journal of Derivatives*, 3(2), 73-84.
- McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk
  Management: Concepts, Techniques and Tools*, revised ed. Princeton
  University Press.

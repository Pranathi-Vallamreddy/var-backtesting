from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data")


def load_prices(tickers, start, end, cache_dir=CACHE_DIR) -> pd.DataFrame:
    """Daily adjusted closes, one column per ticker, cached to CSV.

    Adjusted for splits and dividends so that log returns are total returns.
    Rows with any missing price are dropped rather than forward-filled, since
    a filled price would show up as a spurious zero return.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{'_'.join(tickers)}_{start}_{end}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)[list(tickers)]

    # yfinance treats `end` as exclusive; the config dates are meant inclusively
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(list(tickers), start=start, end=end_excl, auto_adjust=True, progress=False)
    prices = raw["Close"][list(tickers)].dropna()
    if prices.empty:
        raise ValueError(f"no price data for {tickers} between {start} and {end}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).iloc[1:]

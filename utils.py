"""
utils.py
--------
Shared utilities: historical data loading/caching, universe loading,
sector mapping, and small numeric helpers used across the backtester.
"""

import os
import time
import pandas as pd
import numpy as np

import config
from scanner_bridge import load_hardcoded_stocks

try:
    import yfinance as yf
except ImportError:
    yf = None

CACHE_DIR = os.path.join(config.OUTPUT_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def safe_val(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def load_universe():
    """
    Returns a DataFrame with columns: instrument_key, tradingsymbol[, sector]

    Priority:
      1. config.CUSTOM_UNIVERSE_CSV if provided
      2. load_hardcoded_stocks() reused from the live scanner
    """
    if config.CUSTOM_UNIVERSE_CSV and os.path.exists(config.CUSTOM_UNIVERSE_CSV):
        df = pd.read_csv(config.CUSTOM_UNIVERSE_CSV)
    elif load_hardcoded_stocks is not None:
        df = load_hardcoded_stocks()
    else:
        raise RuntimeError(
            "No universe available: live scanner has no load_hardcoded_stocks() "
            "and no CUSTOM_UNIVERSE_CSV was provided."
        )
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "sector" not in df.columns:
        df["sector"] = load_sector_map(df["tradingsymbol"].tolist())
    return df


def load_sector_map(symbols):
    """
    Returns a list of sector labels aligned with `symbols`.
    Uses SECTOR_MAP_CSV if available, otherwise tags everything 'Unknown'
    so sector analysis still runs without failing.
    """
    sector_lookup = {}
    if config.SECTOR_MAP_CSV and os.path.exists(config.SECTOR_MAP_CSV):
        m = pd.read_csv(config.SECTOR_MAP_CSV)
        sector_lookup = dict(zip(m["symbol"], m["sector"]))
    return [sector_lookup.get(s, "Unknown") for s in symbols]


def _cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}.csv")


def get_historical_data(symbol, start=None, end=None, retries=3, sleep_between=0.3):
    """
    Fetches daily OHLCV for `symbol` from `start` (inclusive, with warmup
    buffer already applied by caller) to `end`. Caches to disk so repeated
    backtest runs don't re-hit the network.

    Returns a DataFrame indexed by Date with columns Open, High, Low, Close, Volume
    (ascending date order), or None if no data could be fetched.
    """
    start = start or config.BACKTEST_START_DATE
    end = end or config.BACKTEST_END_DATE

    cache_file = _cache_path(symbol)
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                # Re-fetch only if the cache doesn't already cover the requested window
                if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
                    return df
        except Exception:
            pass

    if yf is None:
        raise ImportError("yfinance is required to fetch historical data. pip install yfinance")

    ticker = f"{symbol}.NS"
    last_err = None
    for attempt in range(retries):
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if data is None or data.empty:
                return None
            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data = data.sort_index()
            data.to_csv(cache_file)
            return data
        except Exception as e:
            last_err = e
            time.sleep(sleep_between * (attempt + 1))
    print(f"[utils] Failed to fetch {symbol}: {last_err}")
    return None


def warmup_start_date():
    return (pd.Timestamp(config.BACKTEST_START_DATE) -
            pd.Timedelta(days=config.WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")


def trading_days_between(date_index, start, end):
    """Returns the sorted slice of a DatetimeIndex between start and end inclusive."""
    return date_index[(date_index >= pd.Timestamp(start)) & (date_index <= pd.Timestamp(end))]

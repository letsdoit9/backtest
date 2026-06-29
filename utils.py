"""
utils.py
--------
Shared utilities: historical data loading/caching, universe loading,
sector mapping, and small numeric helpers used across the backtester.

OPTIMIZATION CHANGES vs original:
1. Disk cache uses Parquet format instead of CSV — 3-5x faster read/write,
   smaller file size, preserves dtypes (no re-parsing needed).
2. Cache validity check uses a lightweight metadata file instead of reading
   the full data file just to check date range.
3. Falls back to CSV cache automatically if pyarrow is not installed.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import config
from scanner_bridge import load_hardcoded_stocks

try:
    import yfinance as yf
except ImportError:
    yf = None

# Check if parquet (pyarrow) is available for faster caching
try:
    import pyarrow  # noqa: F401
    _USE_PARQUET = True
except ImportError:
    _USE_PARQUET = False
    print("[utils] pyarrow not found — using CSV cache (slower). "
          "Run: pip install pyarrow  for faster caching.")

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
    Uses SECTOR_MAP_CSV if available, otherwise tags everything 'Unknown'.
    """
    sector_lookup = {}
    if config.SECTOR_MAP_CSV and os.path.exists(config.SECTOR_MAP_CSV):
        m = pd.read_csv(config.SECTOR_MAP_CSV)
        sector_lookup = dict(zip(m["symbol"], m["sector"]))
    return [sector_lookup.get(s, "Unknown") for s in symbols]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(symbol):
    """Returns the cache file path for a symbol (parquet preferred, else csv)."""
    ext = "parquet" if _USE_PARQUET else "csv"
    return os.path.join(CACHE_DIR, f"{symbol}.{ext}")


def _meta_path(symbol):
    """Lightweight JSON file that stores only the date range of cached data."""
    return os.path.join(CACHE_DIR, f"{symbol}.meta.json")


def _write_cache(symbol, df):
    """Write OHLCV DataFrame to disk cache."""
    path = _cache_path(symbol)
    try:
        if _USE_PARQUET:
            df.to_parquet(path)
        else:
            df.to_csv(path)
        # Write lightweight metadata so we can check date range without reading the file
        meta = {
            "min_date": str(df.index.min().date()),
            "max_date": str(df.index.max().date()),
        }
        with open(_meta_path(symbol), "w") as f:
            json.dump(meta, f)
    except Exception as e:
        print(f"[utils] Cache write failed for {symbol}: {e}")


def _read_cache(symbol):
    """Read OHLCV DataFrame from disk cache. Returns None on failure."""
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        if _USE_PARQUET:
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None


def _cache_covers(symbol, start, end):
    """
    Fast check: does the cache cover the requested date window?
    Reads only the tiny metadata JSON, not the full data file.
    """
    meta_file = _meta_path(symbol)
    if not os.path.exists(meta_file):
        # No metadata — check if legacy cache file exists (migrating from old version)
        if os.path.exists(_cache_path(symbol)):
            return False  # Force a re-read to generate metadata
        return False
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        cache_min = pd.Timestamp(meta["min_date"])
        cache_max = pd.Timestamp(meta["max_date"])
        return (cache_min <= pd.Timestamp(start) and
                cache_max >= pd.Timestamp(end) - pd.Timedelta(days=5))
    except Exception:
        return False


# ── Main data fetch function ──────────────────────────────────────────────────

def get_historical_data(symbol, start=None, end=None, retries=3, sleep_between=0.3):
    """
    Fetches daily OHLCV for `symbol` from `start` to `end`.
    Caches to disk (Parquet if pyarrow available, else CSV) so repeated
    backtest runs don't re-hit the network.

    Returns a DataFrame indexed by Date with columns Open, High, Low, Close, Volume
    (ascending date order), or None if no data could be fetched.
    """
    start = start or config.BACKTEST_START_DATE
    end   = end   or config.BACKTEST_END_DATE

    # ── Fast cache check: read metadata only, not full file ──────────────────
    if _cache_covers(symbol, start, end):
        df = _read_cache(symbol)
        if df is not None and not df.empty:
            return df

    # ── Cache miss — fetch from yfinance ─────────────────────────────────────
    if yf is None:
        raise ImportError("yfinance is required. pip install yfinance")

    ticker   = f"{symbol}.NS"
    last_err = None

    for attempt in range(retries):
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if data is None or data.empty:
                return None
            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data = data.sort_index()
            _write_cache(symbol, data)
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

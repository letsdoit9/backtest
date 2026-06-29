"""
engine.py
---------
Signal generation engine. Reuses the live scanner's indicator and condition
functions EXACTLY as implemented (via scanner_bridge) for every historical
trading day of every stock, with strict no-look-ahead guarantees:

* Indicators are computed once per stock over the full causal history
  (EMA/SMA/RSI/ATR/StochRSI are all backward-looking by construction in
  the live scanner, so row i only ever reflects data up to and including
  day i).
* The condition check for day i only ever reads row i (today) and row i-1
  (yesterday) — never anything from the future.
* A signal generated on day i can only be acted on at day i+1's Open
  (handled downstream in trade_executor.py).

OPTIMIZATION CHANGES vs original:
1. NumPy array extraction done ONCE per stock outside the loop (5-10x faster
   than repeated .iloc[] calls inside the loop).
2. generate_all_signals() uses ProcessPoolExecutor for true parallel CPU work
   across stocks (4-8x faster on multi-core machines).
3. backtest_mask converted to integer indices array so we only iterate over
   qualifying rows instead of every row.
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from scanner_bridge import calculate_indicators_ultra_fast, check_conditions_vectorized
from utils import safe_val, warmup_start_date


def _condition_flags_for_reporting(row_vals):
    """
    Mirrors, for REPORTING ONLY, the exact 16 boolean expressions already
    hard-coded inside check_conditions_vectorized in the live scanner.
    """
    (price, ema5, ema13, ema26, sma50, sma100, sma200, rsi, stochrsi,
     macd, macd_signal, volume, vol_sma50, atr, bb_upper, high_200,
     high_52w, open_price, low, prev_high, prev_close, high_current) = row_vals

    flags = np.zeros(16, dtype=bool)
    flags[0]  = price > ema5 > ema13 > ema26 > 0
    flags[1]  = price > sma50 > sma100 > sma200 > 0
    flags[2]  = rsi > 55
    flags[3]  = stochrsi > 50
    flags[4]  = macd > macd_signal
    flags[5]  = volume > 100000 and volume > vol_sma50
    flags[6]  = price > open_price
    flags[7]  = price >= bb_upper
    flags[8]  = price > (1.05 * high_200)
    flags[9]  = low > prev_high
    flags[10] = price >= (0.97 * high_current)
    flags[11] = price >= (0.95 * high_52w)
    flags[12] = (atr / price) < 0.06 if price > 0 else False
    flags[13] = volume > (2 * vol_sma50) and (prev_close and (price - prev_close) / prev_close > 0.02)
    flags[14] = price > prev_close * 1.01
    flags[15] = volume > vol_sma50 * 1.5
    return flags


def generate_signals_for_stock(symbol, hist_data, sector="Unknown",
                                min_conditions=None, weights=None):
    """
    Runs the live scanner's indicator + condition functions across the full
    causal history of one stock and returns a DataFrame of qualifying signal
    days.

    OPTIMIZATION: All DataFrame columns are extracted to NumPy arrays ONCE
    before the loop. This avoids slow repeated .iloc[] calls — NumPy array
    indexing is 5-10x faster.
    """
    min_conditions = min_conditions or config.DEFAULT_MIN_CONDITIONS
    if weights is None:
        weights = (np.array(config.CONDITION_WEIGHTS) if config.USE_WEIGHTED_SCORING
                   else np.ones(16))

    df = calculate_indicators_ultra_fast(hist_data)
    if df is None or len(df) < 60:
        return pd.DataFrame()

    dates = df.index
    backtest_mask = dates >= pd.Timestamp(config.BACKTEST_START_DATE)
    if not backtest_mask.any():
        return pd.DataFrame()

    # ── OPTIMIZATION: Extract all columns to NumPy arrays ONCE ──────────────
    def col(name, default=0.0):
        if name in df.columns:
            arr = df[name].to_numpy(dtype=float)
            arr = np.where(np.isfinite(arr), arr, default)
            return arr
        return np.full(len(df), default)

    close_arr      = col("Close")
    ema5_arr       = col("EMA5")
    ema13_arr      = col("EMA13")
    ema26_arr      = col("EMA26")
    sma50_arr      = col("SMA50")
    sma100_arr     = col("SMA100")
    sma200_arr     = col("SMA200")
    rsi_arr        = col("RSI", 50.0)
    stochrsi_arr   = col("StochRSI", 50.0)
    macd_arr       = col("MACD")
    macd_sig_arr   = col("MACD_Signal")
    volume_arr     = col("Volume")
    vol_sma50_arr  = col("Volume_SMA50")
    atr_arr        = col("ATR", 0.5)
    bb_upper_arr   = col("BB_Upper")  # fallback handled per-row below
    high200_arr    = col("High_200")
    high52w_arr    = col("High_52w")
    open_arr       = col("Open")
    low_arr        = col("Low")
    high_arr       = col("High")

    # Close array shifted by 1 for "prev_close" and "prev_high"
    prev_close_arr = np.empty_like(close_arr)
    prev_close_arr[0] = 0.0
    prev_close_arr[1:] = close_arr[:-1]

    prev_high_arr = np.empty_like(high_arr)
    prev_high_arr[0] = 0.0
    prev_high_arr[1:] = high_arr[:-1]

    # Indices where we actually need to run the condition check
    valid_indices = np.where(backtest_mask)[0]
    valid_indices = valid_indices[valid_indices > 0]  # need at least index 1 for prev row

    records = []
    for i in valid_indices:
        today_close = close_arr[i]
        today_atr   = atr_arr[i]

        if today_close <= 0 or today_atr <= 0:
            continue

        bb_upper_val = bb_upper_arr[i] if bb_upper_arr[i] > 0 else today_close

        row_vals = (
            today_close,
            ema5_arr[i], ema13_arr[i], ema26_arr[i],
            sma50_arr[i], sma100_arr[i], sma200_arr[i],
            rsi_arr[i], stochrsi_arr[i],
            macd_arr[i], macd_sig_arr[i],
            volume_arr[i], vol_sma50_arr[i],
            today_atr,
            bb_upper_val,
            high200_arr[i], high52w_arr[i],
            open_arr[i], low_arr[i],
            prev_high_arr[i], prev_close_arr[i],
            high_arr[i],
        )

        conditions_met, score = check_conditions_vectorized(*row_vals, weights)

        if conditions_met >= min_conditions:
            flags = _condition_flags_for_reporting(row_vals)
            rec = {
                "Symbol":        symbol,
                "Sector":        sector,
                "SignalDate":    dates[i],
                "ConditionsMet": int(conditions_met),
                "Score":         float(score),
                "ATR":           today_atr,
                "Close":         today_close,
                "RowIndex":      i,
            }
            for c_idx, name in enumerate(config.CONDITION_NAMES):
                rec[name] = bool(flags[c_idx])
            records.append(rec)

    return pd.DataFrame.from_records(records)


# ── Worker function for parallel execution ────────────────────────────────────
# Must be a module-level function (not a lambda/closure) for ProcessPoolExecutor

def _signal_worker(args):
    """Unpacks args tuple and calls generate_signals_for_stock."""
    symbol, hist_data, sector, min_conditions, weights = args
    try:
        return generate_signals_for_stock(symbol, hist_data, sector,
                                          min_conditions, weights)
    except Exception as e:
        print(f"[engine] {symbol}: signal error — {e}")
        return pd.DataFrame()


def generate_all_signals(stock_data_map, min_conditions=None, weights=None,
                         sector_map=None, max_workers=None):
    """
    stock_data_map: dict[symbol] -> raw OHLCV DataFrame (already fetched with
                    warmup buffer included).
    Returns a concatenated DataFrame of all qualifying signals across stocks.

    OPTIMIZATION: Uses ProcessPoolExecutor for parallel CPU-bound work.
    ThreadPoolExecutor won't help here because the GIL blocks CPU-bound Python.
    Set max_workers to number of CPU cores (default: os.cpu_count()).
    """
    sector_map  = sector_map  or {}
    max_workers = max_workers or min(os.cpu_count() or 4, 8)

    args_list = [
        (symbol, hist, sector_map.get(symbol, "Unknown"), min_conditions, weights)
        for symbol, hist in stock_data_map.items()
        if hist is not None and not hist.empty
    ]

    all_signals = []

    # Use ProcessPoolExecutor for true parallelism across CPU cores
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_signal_worker, args): args[0]
                       for args in args_list}
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    sig_df = future.result()
                    if sig_df is not None and not sig_df.empty:
                        all_signals.append(sig_df)
                except Exception as e:
                    print(f"[engine] {sym}: future error — {e}")
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"[engine] Signal generation: {done}/{total} stocks done")
    except Exception:
        # Fallback to serial if multiprocessing fails (e.g. Windows spawn issues)
        print("[engine] Parallel execution failed, falling back to serial mode...")
        for args in args_list:
            sig_df = _signal_worker(args)
            if sig_df is not None and not sig_df.empty:
                all_signals.append(sig_df)

    if not all_signals:
        return pd.DataFrame()

    return (pd.concat(all_signals, ignore_index=True)
              .sort_values("SignalDate")
              .reset_index(drop=True))

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

check_conditions_vectorized's return (conditions_met, weighted_score) is the
ONLY signal generation decision used to gate trades — exactly as the live
scanner computes it. Per-condition booleans (Condition1..16) are derived
separately, using the identical formulas already present in the live file,
purely so they can be stored/reported on per trade; they never influence
which days qualify as signals.
"""

import numpy as np
import pandas as pd

import config
from scanner_bridge import calculate_indicators_ultra_fast, check_conditions_vectorized
from utils import safe_val, warmup_start_date


def _condition_flags_for_reporting(row_vals):
    """
    Mirrors, for REPORTING ONLY, the exact 16 boolean expressions already
    hard-coded inside check_conditions_vectorized in the live scanner. This
    does not gate any trade -- it only lets the analytics layer break down
    *which* of the conditions that were already counted by the live function
    happened to be true, for the Condition Analysis report.
    """
    (price, ema5, ema13, ema26, sma50, sma100, sma200, rsi, stochrsi,
     macd, macd_signal, volume, vol_sma50, atr, bb_upper, high_200,
     high_52w, open_price, low, prev_high, prev_close, high_current) = row_vals

    flags = np.zeros(16, dtype=bool)
    flags[0] = price > ema5 > ema13 > ema26 > 0
    flags[1] = price > sma50 > sma100 > sma200 > 0
    flags[2] = rsi > 55
    flags[3] = stochrsi > 50
    flags[4] = macd > macd_signal
    flags[5] = volume > 100000 and volume > vol_sma50
    flags[6] = price > open_price
    flags[7] = price >= bb_upper
    flags[8] = price > (1.05 * high_200)
    flags[9] = low > prev_high
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
    days (Signal Date, Conditions Met, Score, ATR, per-condition booleans).

    `hist_data` must already include the warmup buffer (extra history before
    BACKTEST_START_DATE) so indicators like SMA200/High_52w are not starved.
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

    records = []
    n = len(df)
    for i in range(1, n):  # start at 1 -> need a "yesterday" row
        if not backtest_mask[i]:
            continue
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]

        today_close = safe_val(today["Close"])
        today_atr = safe_val(today["ATR"], 0.5)
        # Guard only: never feed a zero/negative price into the live scanner's
        # check_conditions_vectorized (which divides atr/price). This does not
        # alter any condition logic -- it just skips unusable data rows
        # (e.g. corrupted/missing bars) the same way a live quote feed would
        # never serve a zero price.
        if today_close <= 0 or today_atr <= 0:
            continue

        row_vals = (
            today_close, safe_val(today["EMA5"]), safe_val(today["EMA13"]),
            safe_val(today["EMA26"]), safe_val(today["SMA50"]), safe_val(today["SMA100"]),
            safe_val(today["SMA200"]), safe_val(today["RSI"], 50), safe_val(today["StochRSI"], 50),
            safe_val(today["MACD"]), safe_val(today["MACD_Signal"]), safe_val(today["Volume"]),
            safe_val(today["Volume_SMA50"]), today_atr,
            safe_val(today["BB_Upper"], today_close), safe_val(today["High_200"]),
            safe_val(today["High_52w"]), safe_val(today["Open"]), safe_val(today["Low"]),
            safe_val(yesterday["High"]), safe_val(yesterday["Close"]), safe_val(today["High"]),
        )

        conditions_met, score = check_conditions_vectorized(*row_vals, weights)

        if conditions_met >= min_conditions:
            flags = _condition_flags_for_reporting(row_vals)
            rec = {
                "Symbol": symbol,
                "Sector": sector,
                "SignalDate": dates[i],
                "ConditionsMet": int(conditions_met),
                "Score": float(score),
                "ATR": row_vals[13],
                "Close": row_vals[0],
                "RowIndex": i,
            }
            for c_idx, name in enumerate(config.CONDITION_NAMES):
                rec[name] = bool(flags[c_idx])
            records.append(rec)

    return pd.DataFrame.from_records(records)


def generate_all_signals(stock_data_map, min_conditions=None, weights=None, sector_map=None):
    """
    stock_data_map: dict[symbol] -> raw OHLCV DataFrame (already fetched with
                    warmup buffer included).
    Returns a concatenated DataFrame of all qualifying signals across stocks.
    """
    sector_map = sector_map or {}
    all_signals = []
    for symbol, hist in stock_data_map.items():
        if hist is None or hist.empty:
            continue
        sig_df = generate_signals_for_stock(
            symbol, hist, sector=sector_map.get(symbol, "Unknown"),
            min_conditions=min_conditions, weights=weights
        )
        if not sig_df.empty:
            all_signals.append(sig_df)

    if not all_signals:
        return pd.DataFrame()
    return pd.concat(all_signals, ignore_index=True).sort_values("SignalDate").reset_index(drop=True)

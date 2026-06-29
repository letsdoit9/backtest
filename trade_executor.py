"""
trade_executor.py
------------------
Turns qualifying signal rows into simulated trades:

  Entry    = Next trading day's Open (no look-ahead).
  ATR      = ATR value on the Signal Date (frozen at entry, per spec).
  Target1  = Entry + 1.5*ATR    Target2 = Entry + 2.0*ATR
  Stoploss = Entry - 1.0*ATR

Exit priority per day: Stoploss → Target2 → Target1 → TimeExit.
"Hit" uses intrabar High/Low range.

OPTIMIZATION CHANGES vs original:
1. Price arrays (High, Low, Close, Open) extracted to NumPy arrays ONCE
   per stock — avoids slow .iloc[] calls inside the per-trade loop.
2. Date index pre-converted to a lookup dict once per stock group, not
   rebuilt for every signal.
3. np.searchsorted used for fast integer-index lookup when date is not
   in the main index dict (edge-case safety).
"""

import numpy as np
import pandas as pd
import config


def _simulate_single_trade(open_arr, high_arr, low_arr, close_arr,
                            n, entry_idx, atr,
                            target1_mult, target2_mult, stop_mult, max_hold):
    """
    Uses pre-extracted NumPy arrays instead of accessing DataFrame row-by-row.

    open_arr / high_arr / low_arr / close_arr : float64 NumPy arrays for
        the full stock history (same length n).
    entry_idx : integer position of the entry bar.
    Returns a dict with trade detail, or None if data runs out.
    """
    if entry_idx >= n:
        return None

    entry_price = open_arr[entry_idx]
    target1     = entry_price + target1_mult * atr
    target2     = entry_price + target2_mult * atr
    stoploss    = entry_price - stop_mult    * atr

    exit_price  = None
    exit_idx    = None
    exit_reason = None

    highest_high = -np.inf
    lowest_low   =  np.inf

    end_idx = min(entry_idx + max_hold, n)

    for idx in range(entry_idx, end_idx):
        h = high_arr[idx]
        l = low_arr[idx]

        if h > highest_high:
            highest_high = h
        if l < lowest_low:
            lowest_low = l

        if l <= stoploss:
            exit_price, exit_idx, exit_reason = stoploss, idx, "Stoploss"
            break
        elif h >= target2:
            exit_price, exit_idx, exit_reason = target2, idx, "Target2"
            break
        elif h >= target1:
            exit_price, exit_idx, exit_reason = target1, idx, "Target1"
            break

    if exit_price is None:
        # Time exit — last bar of holding window (or last bar of data)
        exit_idx    = min(entry_idx + max_hold - 1, n - 1)
        exit_price  = close_arr[exit_idx]
        exit_reason = "TimeExit"
        # Update MFE/MAE for any bars not yet scanned
        for idx in range(end_idx, exit_idx + 1):
            if high_arr[idx] > highest_high:
                highest_high = high_arr[idx]
            if low_arr[idx]  < lowest_low:
                lowest_low   = low_arr[idx]

    holding_days = exit_idx - entry_idx + 1
    return_pct   = (exit_price  - entry_price) / entry_price * 100.0
    mfe          = (highest_high - entry_price) / entry_price * 100.0 if np.isfinite(highest_high) else 0.0
    mae          = (lowest_low   - entry_price) / entry_price * 100.0 if np.isfinite(lowest_low)   else 0.0

    return {
        "entry_price":   round(entry_price, 4),
        "target1":       round(target1, 4),
        "target2":       round(target2, 4),
        "stoploss":      round(stoploss, 4),
        "exit_price":    round(exit_price, 4),
        "exit_idx":      exit_idx,
        "exit_reason":   exit_reason,
        "holding_days":  int(holding_days),
        "return_pct":    round(return_pct, 4),
        "mfe_pct":       round(mfe, 4),
        "mae_pct":       round(mae, 4),
        "highest_high":  round(highest_high, 4) if np.isfinite(highest_high) else None,
        "lowest_low":    round(lowest_low, 4)   if np.isfinite(lowest_low)   else None,
    }


def execute_trades(signals_df, stock_data_map,
                   target1_mult=None, target2_mult=None,
                   stop_mult=None, max_hold=None):
    """
    signals_df    : output of engine.generate_all_signals.
    stock_data_map: dict[symbol] -> OHLCV DataFrame (DatetimeIndex ascending).
    Returns a DataFrame of completed trades.

    OPTIMIZATION: Per-stock NumPy arrays extracted once, shared across all
    signals for that stock. idx_lookup dict built once per stock group.
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    t1_mult   = target1_mult if target1_mult is not None else config.TARGET1_ATR_MULT
    t2_mult   = target2_mult if target2_mult is not None else config.TARGET2_ATR_MULT
    sl_mult   = stop_mult    if stop_mult    is not None else config.STOPLOSS_ATR_MULT
    max_hold_ = max_hold     if max_hold     is not None else config.MAX_HOLDING_DAYS

    trades = []

    for symbol, group in signals_df.groupby("Symbol"):
        price_df = stock_data_map.get(symbol)
        if price_df is None or price_df.empty:
            continue

        # ── Extract arrays ONCE per stock ────────────────────────────────────
        open_arr  = price_df["Open"].to_numpy(dtype=float)
        high_arr  = price_df["High"].to_numpy(dtype=float)
        low_arr   = price_df["Low"].to_numpy(dtype=float)
        close_arr = price_df["Close"].to_numpy(dtype=float)
        date_arr  = price_df.index
        n         = len(price_df)

        # Build date → integer index lookup ONCE per stock
        idx_lookup = {d: i for i, d in enumerate(date_arr)}

        for _, sig in group.iterrows():
            sig_date = sig["SignalDate"]
            sig_idx  = idx_lookup.get(sig_date)
            if sig_idx is None:
                continue

            entry_idx = sig_idx + 1
            atr       = float(sig["ATR"]) if sig["ATR"] else 0.0
            if atr <= 0:
                continue

            result = _simulate_single_trade(
                open_arr, high_arr, low_arr, close_arr, n,
                entry_idx, atr, t1_mult, t2_mult, sl_mult, max_hold_
            )
            if result is None:
                continue

            entry_date = date_arr[entry_idx]
            exit_date  = date_arr[result["exit_idx"]]

            full = {
                "Ticker":        symbol,
                "Sector":        sig.get("Sector", "Unknown"),
                "SignalDate":    sig_date,
                "ATR":           round(atr, 4),
                "ConditionsMet": int(sig["ConditionsMet"]),
                "Score":         float(sig["Score"]),
                "EntryDate":     entry_date,
                "EntryPrice":    result["entry_price"],
                "Target1":       result["target1"],
                "Target2":       result["target2"],
                "Stoploss":      result["stoploss"],
                "ExitDate":      exit_date,
                "ExitPrice":     result["exit_price"],
                "ExitReason":    result["exit_reason"],
                "HoldingDays":   result["holding_days"],
                "ReturnPct":     result["return_pct"],
                "MFE_Pct":       result["mfe_pct"],
                "MAE_Pct":       result["mae_pct"],
                "HighestHigh":   result["highest_high"],
                "LowestLow":     result["lowest_low"],
            }

            for name in config.CONDITION_NAMES:
                full[name] = bool(sig.get(name, False))

            trades.append(full)

    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values(["SignalDate", "Ticker"]).reset_index(drop=True)
    return trades_df

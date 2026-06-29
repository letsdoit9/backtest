"""
trade_executor.py
------------------
Turns qualifying signal rows into simulated trades:

  Entry  = Next trading day's Open (no look-ahead: the signal is known at
           the close of the signal day, so the earliest actionable price is
           the following day's open).
  ATR    = ATR value on the Signal Date (frozen at entry, per spec).
  Target1 = Entry + 1.5*ATR     Target2 = Entry + 2.0*ATR
  Stoploss = Entry - 1.0*ATR

  Each subsequent candle is checked in order; exit priority:
      1. Stoploss hit first         -> exit at Stoploss
      2. else Target2 hit           -> exit at Target2
      3. else Target1 hit           -> exit at Target1
      4. else after MAX_HOLDING_DAYS sessions -> exit at Close of that day

  "Hit" is evaluated using the day's High/Low range (intrabar), consistent
  with how swing target/stop levels are actually triggered.
"""

import numpy as np
import pandas as pd

import config


def _simulate_single_trade(price_df, row_idx_signal, entry_idx, atr,
                            target1_mult=None, target2_mult=None, stop_mult=None,
                            max_hold=None):
    """
    price_df: full OHLCV DataFrame (DatetimeIndex, ascending) for one stock.
    row_idx_signal: integer position of the signal day within price_df.
    entry_idx: integer position of the entry day (signal_idx + 1).
    Returns a dict with full trade detail, or None if data runs out.
    """
    target1_mult = target1_mult if target1_mult is not None else config.TARGET1_ATR_MULT
    target2_mult = target2_mult if target2_mult is not None else config.TARGET2_ATR_MULT
    stop_mult = stop_mult if stop_mult is not None else config.STOPLOSS_ATR_MULT
    max_hold = max_hold if max_hold is not None else config.MAX_HOLDING_DAYS

    n = len(price_df)
    if entry_idx >= n:
        return None  # not enough future data to even enter

    entry_price = float(price_df["Open"].iloc[entry_idx])
    entry_date = price_df.index[entry_idx]

    target1 = entry_price + target1_mult * atr
    target2 = entry_price + target2_mult * atr
    stoploss = entry_price - stop_mult * atr

    exit_price = None
    exit_date = None
    exit_reason = None
    holding_days = 0

    highest_high = -np.inf
    lowest_low = np.inf

    last_checked_idx = entry_idx
    for offset in range(1, max_hold + 1):
        idx = entry_idx + offset - 1  # day 1 of holding == entry day itself can also move
        if idx >= n:
            break
        day = price_df.iloc[idx]
        last_checked_idx = idx
        holding_days = offset
        highest_high = max(highest_high, float(day["High"]))
        lowest_low = min(lowest_low, float(day["Low"]))

        hit_stop = day["Low"] <= stoploss
        hit_t2 = day["High"] >= target2
        hit_t1 = day["High"] >= target1

        if hit_stop:
            exit_price, exit_date, exit_reason = stoploss, price_df.index[idx], "Stoploss"
            break
        elif hit_t2:
            exit_price, exit_date, exit_reason = target2, price_df.index[idx], "Target2"
            break
        elif hit_t1:
            exit_price, exit_date, exit_reason = target1, price_df.index[idx], "Target1"
            break

    if exit_price is None:
        # Exit after max_hold sessions using Close of that day
        idx = entry_idx + max_hold - 1
        if idx >= n:
            idx = n - 1  # data ran out before max_hold sessions elapsed
        day = price_df.iloc[idx]
        exit_price = float(day["Close"])
        exit_date = price_df.index[idx]
        exit_reason = "TimeExit"
        holding_days = idx - entry_idx + 1
        highest_high = max(highest_high, float(day["High"])) if np.isfinite(highest_high) else float(day["High"])
        lowest_low = min(lowest_low, float(day["Low"])) if np.isfinite(lowest_low) else float(day["Low"])

    return_pct = (exit_price - entry_price) / entry_price * 100.0
    mfe = (highest_high - entry_price) / entry_price * 100.0 if np.isfinite(highest_high) else 0.0
    mae = (lowest_low - entry_price) / entry_price * 100.0 if np.isfinite(lowest_low) else 0.0

    return {
        "EntryDate": entry_date,
        "EntryPrice": round(entry_price, 4),
        "Target1": round(target1, 4),
        "Target2": round(target2, 4),
        "Stoploss": round(stoploss, 4),
        "ExitDate": exit_date,
        "ExitPrice": round(exit_price, 4),
        "ExitReason": exit_reason,
        "HoldingDays": int(holding_days),
        "ReturnPct": round(return_pct, 4),
        "MFE_Pct": round(mfe, 4),
        "MAE_Pct": round(mae, 4),
        "HighestHigh": round(highest_high, 4) if np.isfinite(highest_high) else None,
        "LowestLow": round(lowest_low, 4) if np.isfinite(lowest_low) else None,
    }


def execute_trades(signals_df, stock_data_map, target1_mult=None, target2_mult=None,
                    stop_mult=None, max_hold=None):
    """
    signals_df: output of engine.generate_all_signals (one row per qualifying
                signal day, with Symbol, SignalDate, ConditionsMet, Score, ATR,
                Condition1..16 booleans).
    stock_data_map: dict[symbol] -> OHLCV DataFrame (DatetimeIndex ascending).

    Returns a DataFrame of completed trades with full trade-level detail
    plus the original signal/condition columns carried through.
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    trades = []
    for symbol, group in signals_df.groupby("Symbol"):
        price_df = stock_data_map.get(symbol)
        if price_df is None or price_df.empty:
            continue
        idx_lookup = {d: i for i, d in enumerate(price_df.index)}

        for _, sig in group.iterrows():
            sig_date = sig["SignalDate"]
            if sig_date not in idx_lookup:
                continue
            sig_idx = idx_lookup[sig_date]
            entry_idx = sig_idx + 1
            atr = float(sig["ATR"]) if sig["ATR"] else 0.0
            if atr <= 0:
                continue

            trade = _simulate_single_trade(
                price_df, sig_idx, entry_idx, atr,
                target1_mult=target1_mult, target2_mult=target2_mult,
                stop_mult=stop_mult, max_hold=max_hold
            )
            if trade is None:
                continue

            full = {
                "Ticker": symbol,
                "Sector": sig.get("Sector", "Unknown"),
                "SignalDate": sig_date,
                "ATR": round(atr, 4),
                "ConditionsMet": int(sig["ConditionsMet"]),
                "Score": float(sig["Score"]),
            }
            full.update(trade)
            for name in config.CONDITION_NAMES:
                full[name] = bool(sig.get(name, False))
            trades.append(full)

    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values(["SignalDate", "Ticker"]).reset_index(drop=True)
    return trades_df

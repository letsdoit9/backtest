"""
analytics.py
------------
Higher-level analyses built on top of a trade log: sector breakdown,
per-condition contribution, threshold comparison, ATR multiplier
optimization, holding-period optimization, and full sensitivity sweeps.
"""

import pandas as pd
import numpy as np

import config
from metrics import performance_summary, build_equity_curve
from trade_executor import execute_trades


def sector_analysis(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    rows = []
    for sector, g in trades_df.groupby("Sector"):
        wins = g[g["ReturnPct"] > 0]
        losses = g[g["ReturnPct"] < 0]
        gross_profit = wins["ReturnPct"].sum() if len(wins) else 0.0
        gross_loss = abs(losses["ReturnPct"].sum()) if len(losses) else 0.0
        rows.append({
            "Sector": sector,
            "Trades": len(g),
            "WinRate_%": round(len(wins) / len(g) * 100.0, 2),
            "AverageReturn_%": round(g["ReturnPct"].mean(), 3),
            "ProfitFactor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else np.inf,
            "TotalReturn_%": round(g["ReturnPct"].sum(), 3),
        })
    return pd.DataFrame(rows).sort_values("TotalReturn_%", ascending=False).reset_index(drop=True)


def condition_analysis(trades_df):
    """
    For each of the 16 conditions, computes how many trades satisfied it,
    win rate, average return, average holding days, and a simple
    contribution score (win-rate-weighted average return relative to the
    overall baseline) -- then ranks strongest to weakest.
    """
    if trades_df.empty:
        return pd.DataFrame()

    baseline_avg_return = trades_df["ReturnPct"].mean()
    rows = []
    for name in config.CONDITION_NAMES:
        if name not in trades_df.columns:
            continue
        subset = trades_df[trades_df[name] == True]
        if subset.empty:
            rows.append({
                "Condition": name, "TradesSatisfied": 0, "WinRate_%": 0.0,
                "AvgReturn_%": 0.0, "AvgHoldingDays": 0.0, "ContributionScore": 0.0
            })
            continue
        wins = subset[subset["ReturnPct"] > 0]
        avg_ret = subset["ReturnPct"].mean()
        contribution = (avg_ret - baseline_avg_return) * (len(subset) / len(trades_df))
        rows.append({
            "Condition": name,
            "TradesSatisfied": len(subset),
            "WinRate_%": round(len(wins) / len(subset) * 100.0, 2),
            "AvgReturn_%": round(avg_ret, 3),
            "AvgHoldingDays": round(subset["HoldingDays"].mean(), 2),
            "ContributionScore": round(contribution, 4),
        })

    result = pd.DataFrame(rows).sort_values("ContributionScore", ascending=False).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


def threshold_analysis(signals_df, stock_data_map, thresholds=None):
    """
    Re-executes trades for each candidate minimum-conditions threshold
    (filtering the already-generated signals, which were produced with the
    lowest threshold in the sweep so every higher threshold is a strict
    subset) and reports headline metrics per threshold.
    """
    thresholds = thresholds or config.THRESHOLD_SWEEP
    rows = []
    best_threshold, best_score = None, -np.inf

    for th in sorted(thresholds):
        filtered = signals_df[signals_df["ConditionsMet"] >= th]
        trades = execute_trades(filtered, stock_data_map)
        if trades.empty:
            rows.append({"Threshold": f"{th}/16", "Trades": 0})
            continue
        eq = build_equity_curve(trades)
        summary = performance_summary(trades, eq)
        rows.append({
            "Threshold": f"{th}/16",
            "Trades": summary.get("TotalTrades", 0),
            "WinRate_%": summary.get("WinRate_%", 0),
            "ProfitFactor": summary.get("ProfitFactor", 0),
            "Expectancy_%": summary.get("Expectancy_%", 0),
            "CAGR_%": summary.get("CAGR_%", 0),
            "SharpeRatio": summary.get("SharpeRatio", 0),
            "MaxDrawdown_%": summary.get("MaxDrawdown_%", 0),
            "AverageReturn_%": summary.get("AverageReturn_%", 0),
        })
        # simple composite: prioritize risk-adjusted return with enough sample size
        composite = summary.get("SharpeRatio", 0) * np.log1p(max(summary.get("TotalTrades", 0), 1))
        if composite > best_score:
            best_score, best_threshold = composite, th

    df = pd.DataFrame(rows)
    return df, best_threshold


def atr_optimization(signals_df, stock_data_map, atr_mults=None):
    atr_mults = atr_mults or config.ATR_MULT_SWEEP
    rows = []
    best_mult, best_score = None, -np.inf
    for mult in atr_mults:
        # Target1 multiplier varies; Target2 stays proportionally higher, stop fixed at 1.0
        trades = execute_trades(signals_df, stock_data_map, target1_mult=mult,
                                 target2_mult=mult + 0.5, stop_mult=config.STOPLOSS_ATR_MULT)
        if trades.empty:
            rows.append({"Target1_ATR_Mult": mult, "Trades": 0})
            continue
        eq = build_equity_curve(trades)
        summary = performance_summary(trades, eq)
        rows.append({
            "Target1_ATR_Mult": mult,
            "Trades": summary.get("TotalTrades", 0),
            "WinRate_%": summary.get("WinRate_%", 0),
            "ProfitFactor": summary.get("ProfitFactor", 0),
            "Expectancy_%": summary.get("Expectancy_%", 0),
            "AverageReturn_%": summary.get("AverageReturn_%", 0),
            "SharpeRatio": summary.get("SharpeRatio", 0),
        })
        if summary.get("Expectancy_%", -np.inf) > best_score:
            best_score, best_mult = summary.get("Expectancy_%", -np.inf), mult
    return pd.DataFrame(rows), best_mult


def holding_period_optimization(signals_df, stock_data_map, holding_days_list=None):
    holding_days_list = holding_days_list or config.HOLDING_DAYS_SWEEP
    rows = []
    best_days, best_score = None, -np.inf
    for days in holding_days_list:
        trades = execute_trades(signals_df, stock_data_map, max_hold=days)
        if trades.empty:
            rows.append({"MaxHoldingDays": days, "Trades": 0})
            continue
        eq = build_equity_curve(trades)
        summary = performance_summary(trades, eq)
        rows.append({
            "MaxHoldingDays": days,
            "Trades": summary.get("TotalTrades", 0),
            "WinRate_%": summary.get("WinRate_%", 0),
            "ProfitFactor": summary.get("ProfitFactor", 0),
            "Expectancy_%": summary.get("Expectancy_%", 0),
            "AverageReturn_%": summary.get("AverageReturn_%", 0),
            "SharpeRatio": summary.get("SharpeRatio", 0),
        })
        if summary.get("Expectancy_%", -np.inf) > best_score:
            best_score, best_days = summary.get("Expectancy_%", -np.inf), days
    return pd.DataFrame(rows), best_days


def sensitivity_analysis(signals_df, stock_data_map,
                          thresholds=None, atr_mults=None, holding_days_list=None):
    """
    Full parameter sweep over (min_conditions, ATR target1 multiplier,
    holding days). Returns a single comparison table; can be large for wide
    sweeps, so defaults are kept modest.
    """
    thresholds = thresholds or config.THRESHOLD_SWEEP
    atr_mults = atr_mults or config.ATR_MULT_SWEEP
    holding_days_list = holding_days_list or config.HOLDING_DAYS_SWEEP

    rows = []
    for th in thresholds:
        filtered = signals_df[signals_df["ConditionsMet"] >= th]
        if filtered.empty:
            continue
        for mult in atr_mults:
            for days in holding_days_list:
                trades = execute_trades(filtered, stock_data_map, target1_mult=mult,
                                         target2_mult=mult + 0.5,
                                         stop_mult=config.STOPLOSS_ATR_MULT, max_hold=days)
                if trades.empty:
                    continue
                eq = build_equity_curve(trades)
                summary = performance_summary(trades, eq)
                rows.append({
                    "MinConditions": th,
                    "ATR_Target1_Mult": mult,
                    "MaxHoldingDays": days,
                    "Trades": summary.get("TotalTrades", 0),
                    "WinRate_%": summary.get("WinRate_%", 0),
                    "ProfitFactor": summary.get("ProfitFactor", 0),
                    "Expectancy_%": summary.get("Expectancy_%", 0),
                    "CAGR_%": summary.get("CAGR_%", 0),
                    "SharpeRatio": summary.get("SharpeRatio", 0),
                    "MaxDrawdown_%": summary.get("MaxDrawdown_%", 0),
                })
    return pd.DataFrame(rows)

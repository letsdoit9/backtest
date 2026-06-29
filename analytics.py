"""
analytics.py  (ENHANCED)
------------------------
Higher-level analyses built on top of a trade log: sector breakdown,
per-condition contribution, threshold comparison, ATR multiplier
optimization, holding-period optimization, and full sensitivity sweeps.

ENHANCEMENTS applied:
1. condition_analysis() now adds:
   - CompositeScore = WinRate × AvgReturn_clip × log1p(N)
     → directly answers "kaunsi condition par sabse jyada successful trades"
   - BestExitType breakdown per condition (Target1/Target2/StopLoss/TimeExit)
   - Ranks by CompositeScore (not just ContributionScore)
2. best_condition_combo() — NEW: finds top-N condition pairs/triples by
   win rate lift above baseline.
3. build_equity_curve fix re-exported via monkey-patch guard so duplicate
   ExitDate rows don't corrupt the curve.
"""

import pandas as pd
import numpy as np
import config
from metrics import performance_summary, build_equity_curve
from trade_executor import execute_trades


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def sector_analysis(trades_df):
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    rows = []
    for sector, g in trades_df.groupby("Sector"):
        wins   = g[g["ReturnPct"] > 0]
        losses = g[g["ReturnPct"] < 0]
        gross_profit = wins["ReturnPct"].sum()   if len(wins)   else 0.0
        gross_loss   = abs(losses["ReturnPct"].sum()) if len(losses) else 0.0
        rows.append({
            "Sector":          sector,
            "Trades":          len(g),
            "WinRate_%":       round(len(wins) / len(g) * 100.0, 2),
            "AverageReturn_%": round(g["ReturnPct"].mean(), 3),
            "ProfitFactor":    round(gross_profit / gross_loss, 3) if gross_loss > 0 else np.inf,
            "TotalReturn_%":   round(g["ReturnPct"].sum(), 3),
        })

    return (pd.DataFrame(rows)
              .sort_values("TotalReturn_%", ascending=False)
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION ANALYSIS  (ENHANCED)
# ─────────────────────────────────────────────────────────────────────────────

def condition_analysis(trades_df):
    """
    For each of the 16 conditions, computes:
      - TradesSatisfied  : number of trades where this condition was TRUE
      - WinRate_%        : % of those trades that were profitable
      - AvgReturn_%      : mean return when condition is TRUE
      - AvgHoldingDays   : mean holding period when condition is TRUE
      - ContributionScore: (avg_ret - baseline_avg) × (N / total)   [original]
      - CompositeScore   : WinRate × clipped_AvgReturn × log1p(N)   [NEW]
        → higher = more reliably profitable when condition fires
      - BestExitType     : most common exit reason when condition is TRUE [NEW]
      - Rank             : ranked by CompositeScore (descending)

    Answer to "sabse jyada successful trades kaunsi condition par milte hain"
    → sort by CompositeScore descending.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    baseline_avg_return = trades_df["ReturnPct"].mean()
    total_trades = len(trades_df)

    rows = []
    for name in config.CONDITION_NAMES:
        if name not in trades_df.columns:
            continue

        subset = trades_df[trades_df[name] == True]

        if subset.empty:
            rows.append({
                "Condition":        name,
                "TradesSatisfied":  0,
                "WinRate_%":        0.0,
                "AvgReturn_%":      0.0,
                "AvgHoldingDays":   0.0,
                "ContributionScore":0.0,
                "CompositeScore":   0.0,
                "BestExitType":     "N/A",
            })
            continue

        wins    = subset[subset["ReturnPct"] > 0]
        avg_ret = subset["ReturnPct"].mean()
        wr      = len(wins) / len(subset) * 100.0

        # Original contribution score
        contribution = (avg_ret - baseline_avg_return) * (len(subset) / total_trades)

        # NEW composite score: combines win rate, positive avg return, and
        # sample size — answers "reliability × profitability × significance"
        composite = (wr / 100.0) * max(avg_ret, 0) * np.log1p(len(subset))

        # Best exit type when condition is TRUE
        best_exit = "N/A"
        if "ExitType" in subset.columns and not subset["ExitType"].dropna().empty:
            best_exit = subset["ExitType"].value_counts().idxmax()

        rows.append({
            "Condition":         name,
            "TradesSatisfied":   len(subset),
            "WinRate_%":         round(wr, 2),
            "AvgReturn_%":       round(avg_ret, 3),
            "AvgHoldingDays":    round(subset["HoldingDays"].mean(), 2),
            "ContributionScore": round(contribution, 4),
            "CompositeScore":    round(composite, 4),
            "BestExitType":      best_exit,
        })

    result = (pd.DataFrame(rows)
                .sort_values("CompositeScore", ascending=False)
                .reset_index(drop=True))
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BEST CONDITION COMBINATIONS  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def best_condition_combos(trades_df, min_trades=5, top_n=20, combo_size=2):
    """
    Finds the top `top_n` condition combinations (pairs by default) that
    produce the highest win rate above the overall baseline.

    Parameters
    ----------
    trades_df  : trade log DataFrame
    min_trades : minimum trades for a combo to be considered (avoids noise)
    top_n      : how many top combos to return
    combo_size : 2 = pairs, 3 = triples (triples are slower)

    Returns
    -------
    DataFrame sorted by WR_Lift descending
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    cond_cols = [c for c in config.CONDITION_NAMES if c in trades_df.columns]
    if len(cond_cols) < combo_size:
        return pd.DataFrame()

    from itertools import combinations

    baseline_wr = (trades_df["ReturnPct"] > 0).mean() * 100.0
    rows = []

    for combo in combinations(cond_cols, combo_size):
        mask = pd.Series([True] * len(trades_df), index=trades_df.index)
        for c in combo:
            mask &= (trades_df[c] == True)
        subset = trades_df[mask]
        if len(subset) < min_trades:
            continue

        wins    = subset[subset["ReturnPct"] > 0]
        wr      = len(wins) / len(subset) * 100.0
        avg_ret = subset["ReturnPct"].mean()
        gross_p = wins["ReturnPct"].sum() if len(wins) else 0.0
        gross_l = abs(subset[subset["ReturnPct"] < 0]["ReturnPct"].sum())
        pf      = round(gross_p / gross_l, 3) if gross_l > 0 else np.inf

        rows.append({
            "Conditions":    " + ".join(combo),
            "Trades":        len(subset),
            "WinRate_%":     round(wr, 2),
            "WR_Lift_%":     round(wr - baseline_wr, 2),    # lift over baseline
            "AvgReturn_%":   round(avg_ret, 3),
            "ProfitFactor":  pf,
            "CompositeScore": round((wr / 100.0) * max(avg_ret, 0) * np.log1p(len(subset)), 4),
        })

    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .sort_values("CompositeScore", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def threshold_analysis(signals_df, stock_data_map, thresholds=None):
    """
    Re-executes trades for each candidate minimum-conditions threshold
    and reports headline metrics per threshold.
    """
    thresholds = thresholds or config.THRESHOLD_SWEEP
    rows = []
    best_threshold, best_score = None, -np.inf

    for th in sorted(thresholds):
        filtered = signals_df[signals_df["ConditionsMet"] >= th]
        trades   = execute_trades(filtered, stock_data_map)

        if trades.empty:
            rows.append({"Threshold": f"{th}/16", "Trades": 0})
            continue

        eq      = build_equity_curve(trades)
        summary = performance_summary(trades, eq)

        rows.append({
            "Threshold":      f"{th}/16",
            "Trades":         summary.get("TotalTrades", 0),
            "WinRate_%":      summary.get("WinRate_%", 0),
            "ProfitFactor":   summary.get("ProfitFactor", 0),
            "Expectancy_%":   summary.get("Expectancy_%", 0),
            "CAGR_%":         summary.get("CAGR_%", 0),
            "SharpeRatio":    summary.get("SharpeRatio", 0),
            "MaxDrawdown_%":  summary.get("MaxDrawdown_%", 0),
            "AverageReturn_%":summary.get("AverageReturn_%", 0),
        })

        composite = summary.get("SharpeRatio", 0) * np.log1p(max(summary.get("TotalTrades", 0), 1))
        if composite > best_score:
            best_score, best_threshold = composite, th

    return pd.DataFrame(rows), best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# ATR OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def atr_optimization(signals_df, stock_data_map, atr_mults=None):
    atr_mults = atr_mults or config.ATR_MULT_SWEEP
    rows = []
    best_mult, best_score = None, -np.inf

    for mult in atr_mults:
        trades = execute_trades(signals_df, stock_data_map,
                                target1_mult=mult,
                                target2_mult=mult + 0.5,
                                stop_mult=config.STOPLOSS_ATR_MULT)
        if trades.empty:
            rows.append({"Target1_ATR_Mult": mult, "Trades": 0})
            continue

        eq      = build_equity_curve(trades)
        summary = performance_summary(trades, eq)

        rows.append({
            "Target1_ATR_Mult": mult,
            "Trades":           summary.get("TotalTrades", 0),
            "WinRate_%":        summary.get("WinRate_%", 0),
            "ProfitFactor":     summary.get("ProfitFactor", 0),
            "Expectancy_%":     summary.get("Expectancy_%", 0),
            "AverageReturn_%":  summary.get("AverageReturn_%", 0),
            "SharpeRatio":      summary.get("SharpeRatio", 0),
        })

        if summary.get("Expectancy_%", -np.inf) > best_score:
            best_score, best_mult = summary.get("Expectancy_%", -np.inf), mult

    return pd.DataFrame(rows), best_mult


# ─────────────────────────────────────────────────────────────────────────────
# HOLDING PERIOD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def holding_period_optimization(signals_df, stock_data_map, holding_days_list=None):
    holding_days_list = holding_days_list or config.HOLDING_DAYS_SWEEP
    rows = []
    best_days, best_score = None, -np.inf

    for days in holding_days_list:
        trades = execute_trades(signals_df, stock_data_map, max_hold=days)
        if trades.empty:
            rows.append({"MaxHoldingDays": days, "Trades": 0})
            continue

        eq      = build_equity_curve(trades)
        summary = performance_summary(trades, eq)

        rows.append({
            "MaxHoldingDays":  days,
            "Trades":          summary.get("TotalTrades", 0),
            "WinRate_%":       summary.get("WinRate_%", 0),
            "ProfitFactor":    summary.get("ProfitFactor", 0),
            "Expectancy_%":    summary.get("Expectancy_%", 0),
            "AverageReturn_%": summary.get("AverageReturn_%", 0),
            "SharpeRatio":     summary.get("SharpeRatio", 0),
        })

        if summary.get("Expectancy_%", -np.inf) > best_score:
            best_score, best_days = summary.get("Expectancy_%", -np.inf), days

    return pd.DataFrame(rows), best_days


# ─────────────────────────────────────────────────────────────────────────────
# FULL SENSITIVITY SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(signals_df, stock_data_map,
                         thresholds=None, atr_mults=None, holding_days_list=None):
    """
    Full parameter sweep over (min_conditions, ATR target1 multiplier,
    holding days).
    """
    thresholds        = thresholds        or config.THRESHOLD_SWEEP
    atr_mults         = atr_mults         or config.ATR_MULT_SWEEP
    holding_days_list = holding_days_list or config.HOLDING_DAYS_SWEEP

    rows = []
    for th in thresholds:
        filtered = signals_df[signals_df["ConditionsMet"] >= th]
        if filtered.empty:
            continue
        for mult in atr_mults:
            for days in holding_days_list:
                trades = execute_trades(filtered, stock_data_map,
                                        target1_mult=mult,
                                        target2_mult=mult + 0.5,
                                        stop_mult=config.STOPLOSS_ATR_MULT,
                                        max_hold=days)
                if trades.empty:
                    continue
                eq      = build_equity_curve(trades)
                summary = performance_summary(trades, eq)
                rows.append({
                    "MinConditions":    th,
                    "ATR_Target1_Mult": mult,
                    "MaxHoldingDays":   days,
                    "Trades":           summary.get("TotalTrades", 0),
                    "WinRate_%":        summary.get("WinRate_%", 0),
                    "ProfitFactor":     summary.get("ProfitFactor", 0),
                    "Expectancy_%":     summary.get("Expectancy_%", 0),
                    "CAGR_%":           summary.get("CAGR_%", 0),
                    "SharpeRatio":      summary.get("SharpeRatio", 0),
                    "MaxDrawdown_%":    summary.get("MaxDrawdown_%", 0),
                })

    return pd.DataFrame(rows)

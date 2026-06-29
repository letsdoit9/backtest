"""
metrics.py
----------
All performance-metric calculations on a completed trade log.
Pure functions -- no I/O, no plotting (see charts.py / excel_export.py).
"""

import numpy as np
import pandas as pd

import config


def _annualization_factor(holding_days_avg):
    return config.TRADING_DAYS_PER_YEAR / max(holding_days_avg, 1)


def build_equity_curve(trades_df, starting_capital=100000.0, risk_pct=1.0):
    """
    Builds a simple sequential equity curve assuming each trade risks
    `risk_pct`% of current capital (compounding). Trades are ordered by
    ExitDate to reflect when P&L is realized.

    Returns a DataFrame indexed by ExitDate with columns: ReturnPct, PnL, Equity
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["ReturnPct", "PnL", "Equity"])

    df = trades_df.sort_values("ExitDate").copy()
    equity = starting_capital
    rows = []
    for _, t in df.iterrows():
        risk_amount = equity * (risk_pct / 100.0)
        pnl = risk_amount * (t["ReturnPct"] / 100.0)
        equity += pnl
        rows.append({"ExitDate": t["ExitDate"], "ReturnPct": t["ReturnPct"], "PnL": pnl, "Equity": equity})

    eq_df = pd.DataFrame(rows).set_index("ExitDate")
    return eq_df


def compute_drawdown(equity_series):
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100.0
    return drawdown


def performance_summary(trades_df, equity_df=None, starting_capital=100000.0):
    """Returns a dict of all required performance metrics."""
    out = {}
    if trades_df is None or trades_df.empty:
        return {"TotalTrades": 0}

    rets = trades_df["ReturnPct"]
    wins = trades_df[rets > 0]
    losses = trades_df[rets < 0]
    breakeven = trades_df[rets == 0]

    out["TotalTrades"] = len(trades_df)
    out["WinningTrades"] = len(wins)
    out["LosingTrades"] = len(losses)
    out["BreakevenTrades"] = len(breakeven)
    out["WinRate_%"] = round(len(wins) / len(trades_df) * 100.0, 3)
    out["AverageWin_%"] = round(wins["ReturnPct"].mean(), 4) if len(wins) else 0.0
    out["AverageLoss_%"] = round(losses["ReturnPct"].mean(), 4) if len(losses) else 0.0
    out["AverageReturn_%"] = round(rets.mean(), 4)
    out["MedianReturn_%"] = round(rets.median(), 4)

    gross_profit = wins["ReturnPct"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["ReturnPct"].sum()) if len(losses) else 0.0
    out["ProfitFactor"] = round(gross_profit / gross_loss, 4) if gross_loss > 0 else np.inf

    win_rate = len(wins) / len(trades_df)
    avg_win = wins["ReturnPct"].mean() if len(wins) else 0.0
    avg_loss = abs(losses["ReturnPct"].mean()) if len(losses) else 0.0
    out["Expectancy_%"] = round(win_rate * avg_win - (1 - win_rate) * avg_loss, 4)

    out["AverageHoldingDays"] = round(trades_df["HoldingDays"].mean(), 3)
    out["LargestWinner_%"] = round(rets.max(), 4)
    out["LargestLoser_%"] = round(rets.min(), 4)

    # Equity-curve-dependent metrics
    if equity_df is None or equity_df.empty:
        from metrics import build_equity_curve  # local import avoids circularity issues
        equity_df = build_equity_curve(trades_df, starting_capital=starting_capital)

    if not equity_df.empty:
        equity = equity_df["Equity"]
        dd = compute_drawdown(equity)
        out["MaxDrawdown_%"] = round(dd.min(), 4)

        total_return_pct = (equity.iloc[-1] / starting_capital - 1) * 100.0
        years = max((equity_df.index[-1] - equity_df.index[0]).days / 365.25, 1e-6)
        cagr = ((equity.iloc[-1] / starting_capital) ** (1 / years) - 1) * 100.0 if equity.iloc[-1] > 0 else -100.0
        out["CAGR_%"] = round(cagr, 4)
        out["RecoveryFactor"] = round(total_return_pct / abs(out["MaxDrawdown_%"]), 4) if out["MaxDrawdown_%"] != 0 else np.inf
        out["CalmarRatio"] = round(cagr / abs(out["MaxDrawdown_%"]), 4) if out["MaxDrawdown_%"] != 0 else np.inf

        # Per-trade return series used as a proxy return stream for Sharpe/Sortino
        trade_rets = equity_df["ReturnPct"].values / 100.0
        ann_factor = _annualization_factor(out["AverageHoldingDays"])
        mean_r = np.mean(trade_rets)
        std_r = np.std(trade_rets, ddof=1) if len(trade_rets) > 1 else 0.0
        rf_per_trade = config.RISK_FREE_RATE_ANNUAL / ann_factor

        out["AnnualVolatility_%"] = round(std_r * np.sqrt(ann_factor) * 100.0, 4)
        out["SharpeRatio"] = round(((mean_r - rf_per_trade) / std_r) * np.sqrt(ann_factor), 4) if std_r > 0 else 0.0

        downside = trade_rets[trade_rets < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        out["SortinoRatio"] = round(((mean_r - rf_per_trade) / downside_std) * np.sqrt(ann_factor), 4) if downside_std > 0 else 0.0
    else:
        out.update({"MaxDrawdown_%": 0, "CAGR_%": 0, "RecoveryFactor": 0, "CalmarRatio": 0,
                    "AnnualVolatility_%": 0, "SharpeRatio": 0, "SortinoRatio": 0})

    return out


def monthly_analysis(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["Month"] = pd.to_datetime(df["ExitDate"]).dt.to_period("M")
    grouped = df.groupby("Month")
    rows = []
    for month, g in grouped:
        wins = g[g["ReturnPct"] > 0]
        rows.append({
            "Month": str(month),
            "Trades": len(g),
            "WinRate_%": round(len(wins) / len(g) * 100.0, 2),
            "PnL_SumReturn_%": round(g["ReturnPct"].sum(), 3),
            "AvgReturn_%": round(g["ReturnPct"].mean(), 3),
        })
    return pd.DataFrame(rows)


def yearly_analysis(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["Year"] = pd.to_datetime(df["ExitDate"]).dt.year
    grouped = df.groupby("Year")
    rows = []
    for year, g in grouped:
        wins = g[g["ReturnPct"] > 0]
        rows.append({
            "Year": int(year),
            "Trades": len(g),
            "WinRate_%": round(len(wins) / len(g) * 100.0, 2),
            "Return_%": round(g["ReturnPct"].sum(), 3),
        })
    return pd.DataFrame(rows)

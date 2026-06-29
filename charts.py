"""
charts.py
---------
Generates all required PNG charts using matplotlib (no seaborn dependency,
keeps requirements minimal).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from metrics import compute_drawdown

os.makedirs(config.CHARTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(config.CHARTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_curve(equity_df):
    if equity_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_df.index, equity_df["Equity"], color="#1f77b4", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    return _save(fig, "equity_curve")


def plot_drawdown(equity_df):
    if equity_df.empty:
        return None
    dd = compute_drawdown(equity_df["Equity"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    ax.set_title("Drawdown Curve (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    return _save(fig, "drawdown_curve")


def plot_monthly_heatmap(monthly_df):
    if monthly_df.empty:
        return None
    df = monthly_df.copy()
    df["Year"] = df["Month"].str.slice(0, 4)
    df["MonthNum"] = df["Month"].str.slice(5, 7)
    pivot = df.pivot(index="Year", columns="MonthNum", values="PnL_SumReturn_%").fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-abs(pivot.values).max() if pivot.values.size else -1,
                   vmax=abs(pivot.values).max() if pivot.values.size else 1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly Return Heatmap (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, "monthly_heatmap")


def plot_rolling_metric(equity_df, window_days=60, metric="cagr"):
    if equity_df.empty or len(equity_df) < window_days + 2:
        return None
    eq = equity_df["Equity"]
    rolling_vals = []
    dates = []
    for i in range(window_days, len(eq)):
        window = eq.iloc[i - window_days:i + 1]
        years = max((window.index[-1] - window.index[0]).days / 365.25, 1e-6)
        if metric == "cagr":
            val = ((window.iloc[-1] / window.iloc[0]) ** (1 / years) - 1) * 100.0
        else:  # sharpe
            rets = window.pct_change().dropna()
            val = (rets.mean() / rets.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
                   if rets.std() > 0 else 0)
        rolling_vals.append(val)
        dates.append(window.index[-1])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, rolling_vals, color="#2ca02c" if metric == "cagr" else "#9467bd")
    ax.set_title(f"Rolling {'CAGR (%)' if metric == 'cagr' else 'Sharpe Ratio'} ({window_days}d window)")
    ax.grid(alpha=0.3)
    return _save(fig, f"rolling_{metric}")


def plot_return_distribution(trades_df):
    if trades_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(trades_df["ReturnPct"], bins=40, color="#1f77b4", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Trade Return Distribution (%)")
    ax.set_xlabel("Return %")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    return _save(fig, "return_distribution")


def plot_sector_performance(sector_df):
    if sector_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(sector_df))))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in sector_df["TotalReturn_%"]]
    ax.barh(sector_df["Sector"], sector_df["TotalReturn_%"], color=colors)
    ax.set_title("Sector Performance (Total Return %)")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, "sector_performance")


def plot_condition_contribution(condition_df):
    if condition_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(condition_df))))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in condition_df["ContributionScore"]]
    ax.barh(condition_df["Condition"], condition_df["ContributionScore"], color=colors)
    ax.set_title("Condition Contribution Score")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, "condition_contribution")


def plot_threshold_comparison(threshold_df):
    if threshold_df.empty or "WinRate_%" not in threshold_df.columns:
        return None
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(threshold_df["Threshold"], threshold_df["Trades"], color="#aec7e8", alpha=0.7, label="Trades")
    ax1.set_ylabel("Trades")
    ax2 = ax1.twinx()
    ax2.plot(threshold_df["Threshold"], threshold_df["WinRate_%"], color="#d62728", marker="o", label="Win Rate %")
    ax2.set_ylabel("Win Rate %")
    ax1.set_title("Threshold Comparison: Trade Count vs Win Rate")
    fig.legend(loc="upper right")
    return _save(fig, "threshold_comparison")


def generate_all_charts(equity_df, trades_df, monthly_df, sector_df, condition_df, threshold_df):
    paths = {}
    paths["equity_curve"] = plot_equity_curve(equity_df)
    paths["drawdown_curve"] = plot_drawdown(equity_df)
    paths["monthly_heatmap"] = plot_monthly_heatmap(monthly_df)
    paths["rolling_cagr"] = plot_rolling_metric(equity_df, metric="cagr")
    paths["rolling_sharpe"] = plot_rolling_metric(equity_df, metric="sharpe")
    paths["return_distribution"] = plot_return_distribution(trades_df)
    paths["sector_performance"] = plot_sector_performance(sector_df)
    paths["condition_contribution"] = plot_condition_contribution(condition_df)
    paths["threshold_comparison"] = plot_threshold_comparison(threshold_df)
    return {k: v for k, v in paths.items() if v}

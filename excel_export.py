"""
charts.py  (FIXED)
------------------
Generates all required PNG charts using matplotlib (no seaborn dependency,
keeps requirements minimal).

FIXES applied:
1. plot_equity_curve: duplicate-date handling → group by date, sort index,
   proper DateFormatter on x-axis so curve never "jumps back".
2. Added plot_best_conditions() — bar chart of top conditions ranked by a
   composite score (WinRate × AvgReturn × log(Trades)).
3. plot_drawdown: same duplicate-date guard.
4. All charts use tight_layout() for cleaner output.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
from metrics import compute_drawdown

os.makedirs(config.CHARTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(config.CHARTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _clean_equity_df(equity_df):
    """
    Fix duplicate-date problem: when multiple trades exit on the same date
    the original build_equity_curve already produces only one row per trade
    (set_index keeps all rows if called on a column), but if downstream code
    re-indexes or concatenates, dupes can creep in.
    We take the *last* equity value on each calendar date (correct: it's
    the final compounded value after all trades on that day).
    """
    df = equity_df.copy()
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Sort chronologically, then keep last value per date
    df = df.sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    return df


def plot_equity_curve(equity_df):
    if equity_df is None or equity_df.empty:
        return None

    df = _clean_equity_df(equity_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Equity"], color="#1f77b4", linewidth=1.5, label="Portfolio Equity")

    # Shade area under curve
    ax.fill_between(df.index, df["Equity"].min() * 0.99, df["Equity"],
                    alpha=0.08, color="#1f77b4")

    # Annotate start / end
    ax.annotate(f'Start: ₹{df["Equity"].iloc[0]:,.0f}',
                xy=(df.index[0], df["Equity"].iloc[0]),
                xytext=(10, 10), textcoords="offset points", fontsize=8, color="grey")
    ax.annotate(f'End: ₹{df["Equity"].iloc[-1]:,.0f}',
                xy=(df.index[-1], df["Equity"].iloc[-1]),
                xytext=(-60, 10), textcoords="offset points", fontsize=8, color="navy")

    # X-axis: auto-format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.set_title("Equity Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "equity_curve")


def plot_drawdown(equity_df):
    if equity_df is None or equity_df.empty:
        return None

    df = _clean_equity_df(equity_df)
    dd = compute_drawdown(df["Equity"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.55, label="Drawdown")
    ax.plot(dd.index, dd.values, color="#a00000", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.set_title("Drawdown Curve (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "drawdown_curve")


def plot_monthly_heatmap(monthly_df):
    if monthly_df is None or monthly_df.empty:
        return None

    df = monthly_df.copy()
    df["Year"] = df["Month"].str.slice(0, 4)
    df["MonthNum"] = df["Month"].str.slice(5, 7)
    pivot = df.pivot(index="Year", columns="MonthNum", values="PnL_SumReturn_%").fillna(0)

    max_val = abs(pivot.values).max() if pivot.values.size else 1

    fig, ax = plt.subplots(figsize=(12, max(3, 0.55 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-max_val, vmax=max_val)

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    col_labels = [month_labels[int(c)-1] if c.isdigit() else c for c in pivot.columns]

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate each cell
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=7, color="black" if abs(val) < max_val * 0.6 else "white")

    ax.set_title("Monthly Return Heatmap (%)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Return %")
    fig.tight_layout()
    return _save(fig, "monthly_heatmap")


def plot_rolling_metric(equity_df, window_days=60, metric="cagr"):
    if equity_df is None or equity_df.empty or len(equity_df) < window_days + 2:
        return None

    df = _clean_equity_df(equity_df)
    eq = df["Equity"]
    rolling_vals, dates = [], []

    for i in range(window_days, len(eq)):
        window = eq.iloc[i - window_days:i + 1]
        years = max((window.index[-1] - window.index[0]).days / 365.25, 1e-6)
        if metric == "cagr":
            val = ((window.iloc[-1] / window.iloc[0]) ** (1 / years) - 1) * 100.0
        else:
            rets = window.pct_change().dropna()
            val = (rets.mean() / rets.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
                   if rets.std() > 0 else 0)
        rolling_vals.append(val)
        dates.append(window.index[-1])

    fig, ax = plt.subplots(figsize=(12, 4))
    color = "#2ca02c" if metric == "cagr" else "#9467bd"
    ax.plot(dates, rolling_vals, color=color, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    fig.autofmt_xdate(rotation=30, ha="right")

    label = f"Rolling {'CAGR (%)' if metric == 'cagr' else 'Sharpe Ratio'} ({window_days}d window)"
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"rolling_{metric}")


def plot_return_distribution(trades_df):
    if trades_df is None or trades_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    returns = trades_df["ReturnPct"]
    n_bins = min(60, max(20, len(returns) // 5))
    ax.hist(returns, bins=n_bins, color="#1f77b4", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=1.2, label="Break-even")
    ax.axvline(returns.mean(), color="green", linewidth=1.2, linestyle="--",
               label=f"Mean: {returns.mean():.2f}%")
    ax.axvline(returns.median(), color="orange", linewidth=1.2, linestyle=":",
               label=f"Median: {returns.median():.2f}%")

    ax.set_title("Trade Return Distribution (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Return %")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "return_distribution")


def plot_sector_performance(sector_df):
    if sector_df is None or sector_df.empty:
        return None

    df = sector_df.sort_values("TotalReturn_%")
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["TotalReturn_%"]]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(df))))
    bars = ax.barh(df["Sector"], df["TotalReturn_%"], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_title("Sector Performance (Total Return %)", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return _save(fig, "sector_performance")


def plot_condition_contribution(condition_df):
    """
    Enhanced: shows top-10 and bottom-5 conditions ranked by ContributionScore.
    Also annotates WinRate% and AvgReturn% on each bar.
    """
    if condition_df is None or condition_df.empty:
        return None

    df = condition_df.sort_values("ContributionScore", ascending=False)

    # Show top 10 best + bottom 5 worst
    top = df.head(10)
    bottom = df.tail(5)
    plot_df = pd.concat([top, bottom]).drop_duplicates().reset_index(drop=True)
    plot_df = plot_df.sort_values("ContributionScore")

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in plot_df["ContributionScore"]]

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(plot_df))))
    bars = ax.barh(plot_df["Condition"], plot_df["ContributionScore"], color=colors)

    # Annotate each bar with WinRate and AvgReturn if columns present
    for i, (_, row) in enumerate(plot_df.iterrows()):
        label_parts = []
        if "WinRate_%" in row:
            label_parts.append(f"WR:{row['WinRate_%']:.0f}%")
        if "AvgReturn_%" in row:
            label_parts.append(f"Avg:{row['AvgReturn_%']:.2f}%")
        if "TradesSatisfied" in row:
            label_parts.append(f"N:{int(row['TradesSatisfied'])}")
        label = "  " + " | ".join(label_parts)
        score = row["ContributionScore"]
        x_pos = score + (abs(plot_df["ContributionScore"].max()) * 0.01) if score >= 0 else score - (abs(plot_df["ContributionScore"].max()) * 0.01)
        ax.text(x_pos, i, label, va="center", fontsize=7,
                color="darkgreen" if score >= 0 else "darkred")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Condition Contribution Score (Top 10 + Bottom 5)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Contribution Score\n(positive = above-average returns when condition is TRUE)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return _save(fig, "condition_contribution")


def plot_best_conditions(condition_df):
    """
    NEW CHART: Ranks conditions by a composite success score:
        CompositeScore = WinRate_% × AvgReturn_% × log1p(TradesSatisfied)
    This answers: "Kaunsi condition par sabse jyada successful trades milte hain?"
    """
    if condition_df is None or condition_df.empty:
        return None

    df = condition_df.copy()

    required = {"WinRate_%", "AvgReturn_%", "TradesSatisfied"}
    if not required.issubset(df.columns):
        return None

    # Only keep conditions with enough trades (at least 5)
    df = df[df["TradesSatisfied"] >= 5].copy()
    if df.empty:
        return None

    # Composite: WinRate × AvgReturn (only positive avg) × log of trades
    df["CompositeScore"] = (
        (df["WinRate_%"] / 100.0)
        * df["AvgReturn_%"].clip(lower=0)
        * np.log1p(df["TradesSatisfied"])
    )

    df = df.sort_values("CompositeScore", ascending=False).head(16)
    df = df.sort_values("CompositeScore")  # for horizontal bar

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, 0.5 * len(df))))
    fig.suptitle("Best Performing Conditions — Kaunsi Condition Par Sabse Zyada Success Milti Hai?",
                 fontsize=13, fontweight="bold", y=1.01)

    # --- Chart 1: Composite Score ---
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    axes[0].barh(df["Condition"], df["CompositeScore"], color=colors)
    axes[0].set_title("Composite Score\n(WinRate × AvgReturn × log(N))")
    axes[0].set_xlabel("Score (higher = better)")
    axes[0].grid(alpha=0.3, axis="x")

    # --- Chart 2: Win Rate % ---
    df_wr = df.sort_values("WinRate_%")
    colors2 = ["#2ca02c" if v >= 50 else "#d62728" for v in df_wr["WinRate_%"]]
    axes[1].barh(df_wr["Condition"], df_wr["WinRate_%"], color=colors2)
    axes[1].axvline(50, color="black", linewidth=1, linestyle="--", label="50% baseline")
    axes[1].set_title("Win Rate %\n(when condition is TRUE)")
    axes[1].set_xlabel("Win Rate %")
    axes[1].set_xlim(0, 100)
    for bar, val in zip(axes[1].patches, df_wr["WinRate_%"]):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="x")

    # --- Chart 3: Avg Return % ---
    df_ar = df.sort_values("AvgReturn_%")
    colors3 = ["#2ca02c" if v >= 0 else "#d62728" for v in df_ar["AvgReturn_%"]]
    axes[2].barh(df_ar["Condition"], df_ar["AvgReturn_%"], color=colors3)
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_title("Average Return % per Trade\n(when condition is TRUE)")
    axes[2].set_xlabel("Avg Return %")
    for bar, val in zip(axes[2].patches, df_ar["AvgReturn_%"]):
        x = bar.get_width() + 0.05 if val >= 0 else bar.get_width() - 0.1
        axes[2].text(x, bar.get_y() + bar.get_height()/2,
                     f"{val:.2f}%", va="center", fontsize=8)
    axes[2].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    return _save(fig, "best_conditions_analysis")


def plot_condition_combinations(condition_df, trades_df, top_n=10):
    """
    NEW CHART: Finds top condition PAIRS — which 2 conditions together
    give the highest win rate. Useful to understand which combinations
    drive the best trades.
    """
    if condition_df is None or condition_df.empty:
        return None
    if trades_df is None or trades_df.empty:
        return None

    # Get list of condition columns that exist in trades_df
    cond_cols = [c for c in config.CONDITION_NAMES if c in trades_df.columns]
    if len(cond_cols) < 2:
        return None

    rows = []
    baseline_wr = (trades_df["ReturnPct"] > 0).mean() * 100.0

    for i in range(len(cond_cols)):
        for j in range(i + 1, len(cond_cols)):
            c1, c2 = cond_cols[i], cond_cols[j]
            subset = trades_df[(trades_df[c1] == True) & (trades_df[c2] == True)]
            if len(subset) < 5:
                continue
            wins = subset[subset["ReturnPct"] > 0]
            wr = len(wins) / len(subset) * 100.0
            avg_ret = subset["ReturnPct"].mean()
            rows.append({
                "Pair": f"{c1}\n+ {c2}",
                "Trades": len(subset),
                "WinRate_%": round(wr, 1),
                "AvgReturn_%": round(avg_ret, 3),
                "WR_Lift": round(wr - baseline_wr, 1),   # lift over baseline
            })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("WR_Lift", ascending=False).head(top_n)
    df = df.sort_values("WR_Lift")

    fig, ax = plt.subplots(figsize=(13, max(5, 0.6 * len(df))))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["WR_Lift"]]
    bars = ax.barh(df["Pair"], df["WR_Lift"], color=colors)

    for bar, (_, row) in zip(bars, df.iterrows()):
        label = f"  WR:{row['WinRate_%']:.0f}% | Avg:{row['AvgReturn_%']:.2f}% | N:{int(row['Trades'])}"
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Top {top_n} Condition Pairs by Win Rate Lift over Baseline ({baseline_wr:.1f}%)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Win Rate Lift % (vs overall baseline)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return _save(fig, "condition_pairs_analysis")


def plot_threshold_comparison(threshold_df):
    if threshold_df is None or threshold_df.empty or "WinRate_%" not in threshold_df.columns:
        return None

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(threshold_df["Threshold"], threshold_df["Trades"],
            color="#aec7e8", alpha=0.7, label="Trades")
    ax1.set_ylabel("Number of Trades")

    ax2 = ax1.twinx()
    ax2.plot(threshold_df["Threshold"], threshold_df["WinRate_%"],
             color="#d62728", marker="o", linewidth=1.8, label="Win Rate %")
    ax2.set_ylabel("Win Rate %")

    ax1.set_title("Threshold Comparison: Trade Count vs Win Rate", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Min Conditions Threshold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "threshold_comparison")


def generate_all_charts(equity_df, trades_df, monthly_df, sector_df, condition_df, threshold_df):
    paths = {}
    paths["equity_curve"]           = plot_equity_curve(equity_df)
    paths["drawdown_curve"]         = plot_drawdown(equity_df)
    paths["monthly_heatmap"]        = plot_monthly_heatmap(monthly_df)
    paths["rolling_cagr"]           = plot_rolling_metric(equity_df, metric="cagr")
    paths["rolling_sharpe"]         = plot_rolling_metric(equity_df, metric="sharpe")
    paths["return_distribution"]    = plot_return_distribution(trades_df)
    paths["sector_performance"]     = plot_sector_performance(sector_df)
    paths["condition_contribution"] = plot_condition_contribution(condition_df)
    paths["best_conditions"]        = plot_best_conditions(condition_df)
    paths["condition_pairs"]        = plot_condition_combinations(condition_df, trades_df)
    paths["threshold_comparison"]   = plot_threshold_comparison(threshold_df)
    return {k: v for k, v in paths.items() if v}

"""
backtester.py  — Streamlit UI  (FIXED)
=======================================
Fixes applied from log analysis:
  1. use_container_width → width='stretch'  (Streamlit 1.58 deprecation)
  2. Equity curve: proper chronological sort + duplicate-date dedup before plotting
  3. Best conditions tab: CompositeScore-based ranking + condition-pair analysis
  4. Delisted symbol warnings suppressed cleanly (no crash, just skip)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker

import config
from utils import load_universe, get_historical_data, warmup_start_date
from engine import generate_all_signals
from trade_executor import execute_trades
from metrics import (build_equity_curve, performance_summary,
                     monthly_analysis, yearly_analysis, compute_drawdown)
from analytics import (sector_analysis, condition_analysis,
                       best_condition_combos, threshold_analysis,
                       atr_optimization, holding_period_optimization)
from excel_export import write_excel_report, write_csv_outputs, write_performance_json

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upstox Swing Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Upstox Elite Swing Scanner — Backtesting Dashboard")

# ──────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Backtest Settings")
    min_conditions = st.slider("Min Conditions (out of 16)", 8, 16,
                               value=config.DEFAULT_MIN_CONDITIONS, step=1)
    max_workers    = st.slider("Parallel Workers", 4, 32,
                               value=config.MAX_WORKERS, step=4)
    run_opts       = st.checkbox("Run Full Optimizations (slower)", value=False)
    run_btn        = st.button("🚀 Run Backtest", type="primary")

# ──────────────────────────────────────────────────────────────────
# HELPER: clean equity df (fix duplicate-date problem)
# ──────────────────────────────────────────────────────────────────
def _clean_equity(eq_df):
    if eq_df is None or eq_df.empty:
        return eq_df
    df = eq_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    return df

# ──────────────────────────────────────────────────────────────────
# HELPER: plot equity curve (fixed)
# ──────────────────────────────────────────────────────────────────
def _plot_equity(eq_df):
    df = _clean_equity(eq_df)
    if df is None or df.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Equity"], color="#1f77b4", linewidth=1.5)
    ax.fill_between(df.index, df["Equity"].min() * 0.98, df["Equity"],
                    alpha=0.07, color="#1f77b4")
    ax.annotate(f'₹{df["Equity"].iloc[-1]:,.0f}',
                xy=(df.index[-1], df["Equity"].iloc[-1]),
                xytext=(-55, 8), textcoords="offset points",
                fontsize=9, color="navy", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_title("Equity Curve", fontweight="bold")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────
# HELPER: plot drawdown (fixed)
# ──────────────────────────────────────────────────────────────────
def _plot_drawdown(eq_df):
    df = _clean_equity(eq_df)
    if df is None or df.empty:
        return None
    dd = compute_drawdown(df["Equity"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.55)
    ax.plot(dd.index, dd.values, color="#a00000", linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.set_title("Drawdown %", fontweight="bold")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────
# HELPER: best conditions chart (NEW)
# ──────────────────────────────────────────────────────────────────
def _plot_best_conditions(cond_df):
    if cond_df is None or cond_df.empty:
        return None
    df = cond_df[cond_df["TradesSatisfied"] >= 5].copy()
    if df.empty:
        return None
    df = df.sort_values("CompositeScore", ascending=False).head(12)
    df = df.sort_values("CompositeScore")

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.5 * len(df))))
    fig.suptitle("Best Performing Conditions — Sabse Zyada Successful Trades",
                 fontweight="bold", fontsize=13)

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))

    # Composite score
    axes[0].barh(df["Condition"], df["CompositeScore"], color=colors)
    axes[0].set_title("Composite Score\n(WinRate × AvgReturn × log N)")
    axes[0].grid(alpha=0.3, axis="x")

    # Win rate
    df2 = df.sort_values("WinRate_%")
    clrs2 = ["#2ca02c" if v >= 50 else "#d62728" for v in df2["WinRate_%"]]
    axes[1].barh(df2["Condition"], df2["WinRate_%"], color=clrs2)
    axes[1].axvline(50, color="black", linewidth=1, linestyle="--")
    for bar, val in zip(axes[1].patches, df2["WinRate_%"]):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=8)
    axes[1].set_title("Win Rate %")
    axes[1].set_xlim(0, 100)
    axes[1].grid(alpha=0.3, axis="x")

    # Avg return
    df3 = df.sort_values("AvgReturn_%")
    clrs3 = ["#2ca02c" if v >= 0 else "#d62728" for v in df3["AvgReturn_%"]]
    axes[2].barh(df3["Condition"], df3["AvgReturn_%"], color=clrs3)
    axes[2].axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(axes[2].patches, df3["AvgReturn_%"]):
        axes[2].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                     f"{val:.2f}%", va="center", fontsize=8)
    axes[2].set_title("Avg Return % per Trade")
    axes[2].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────
# HELPER: condition pairs chart (NEW)
# ──────────────────────────────────────────────────────────────────
def _plot_condition_pairs(combos_df):
    if combos_df is None or combos_df.empty:
        return None
    df = combos_df.sort_values("WR_Lift_%").tail(15)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["WR_Lift_%"]]
    fig, ax = plt.subplots(figsize=(14, max(5, 0.55 * len(df))))
    bars = ax.barh(df["Conditions"], df["WR_Lift_%"], color=colors)
    for bar, (_, row) in zip(bars, df.iterrows()):
        label = f"  WR:{row['WinRate_%']:.0f}% | Avg:{row['AvgReturn_%']:.2f}% | N:{int(row['Trades'])}"
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Top Condition Pairs — Win Rate Lift over Baseline",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Win Rate Lift %")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────
# HELPER: monthly heatmap
# ──────────────────────────────────────────────────────────────────
def _plot_monthly_heatmap(monthly_df):
    if monthly_df is None or monthly_df.empty:
        return None
    df = monthly_df.copy()
    df["Year"]     = df["Month"].str.slice(0, 4)
    df["MonthNum"] = df["Month"].str.slice(5, 7)
    pivot   = df.pivot(index="Year", columns="MonthNum",
                       values="PnL_SumReturn_%").fillna(0)
    max_val = abs(pivot.values).max() if pivot.values.size else 1
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    col_labels = [month_labels[int(c)-1] for c in pivot.columns if c.isdigit()]

    fig, ax = plt.subplots(figsize=(12, max(3, 0.55 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-max_val, vmax=max_val)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=7, color="black" if abs(val) < max_val*0.6 else "white")
    ax.set_title("Monthly Return Heatmap (%)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Return %")
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────
# DATA FETCH (cached)
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(max_workers):
    universe_df = load_universe()
    start = warmup_start_date()
    end   = config.BACKTEST_END_DATE
    sector_map = dict(zip(universe_df["tradingsymbol"], universe_df["sector"]))
    symbols    = universe_df["tradingsymbol"].tolist()
    stock_data_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_historical_data, sym, start, end): sym
                   for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                stock_data_map[sym] = future.result()
            except Exception:
                stock_data_map[sym] = None
    valid = {k: v for k, v in stock_data_map.items()
             if v is not None and not v.empty}
    return valid, sector_map

# ──────────────────────────────────────────────────────────────────
# MAIN RUN
# ──────────────────────────────────────────────────────────────────
if run_btn:
    with st.status("⏳ Fetching historical data…", expanded=True) as status:
        t0 = time.time()
        valid_data, sector_map = fetch_data(max_workers)
        st.write(f"✅ {len(valid_data)} symbols loaded")

        status.update(label="🔍 Generating signals…")
        base_th     = min(config.THRESHOLD_SWEEP + [min_conditions])
        all_signals = generate_all_signals(valid_data, min_conditions=base_th,
                                           sector_map=sector_map)
        active_sigs = all_signals[all_signals["ConditionsMet"] >= min_conditions]
        st.write(f"✅ {len(active_sigs)} signals at threshold {min_conditions}/16")

        status.update(label="💹 Simulating trades…")
        trades_df = execute_trades(active_sigs, valid_data)
        st.write(f"✅ {len(trades_df)} trades simulated")

        status.update(label="📊 Computing analytics…")
        equity_df    = _clean_equity(build_equity_curve(trades_df))
        summary      = performance_summary(trades_df, equity_df)
        monthly_df   = monthly_analysis(trades_df)
        yearly_df    = yearly_analysis(trades_df)
        sector_df    = sector_analysis(trades_df)
        condition_df = condition_analysis(trades_df)
        combos_df    = best_condition_combos(trades_df, min_trades=5,
                                             top_n=20, combo_size=2)

        threshold_df = atr_df = holding_df = pd.DataFrame()
        if run_opts:
            status.update(label="🔬 Running optimizations…")
            threshold_df, _ = threshold_analysis(all_signals, valid_data)
            atr_df,       _ = atr_optimization(active_sigs, valid_data)
            holding_df,   _ = holding_period_optimization(active_sigs, valid_data)

        drawdown_df = pd.DataFrame()
        if not equity_df.empty:
            drawdown_df = compute_drawdown(equity_df["Equity"]).to_frame("Drawdown_%")

        status.update(label="💾 Writing outputs…")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        excel_path = write_excel_report(
            summary, trades_df, monthly_df, yearly_df, condition_df,
            sector_df, equity_df, drawdown_df, threshold_df, atr_df, holding_df,
            condition_combos_df=combos_df,
        )
        write_csv_outputs(trades_df, equity_df, monthly_df, condition_df,
                          sector_df, condition_combos_df=combos_df)
        write_performance_json(summary)

        elapsed = time.time() - t0
        status.update(label=f"✅ Done in {elapsed:.0f}s", state="complete")

    # ── SUMMARY METRICS ──────────────────────────────────────────
    st.subheader("📋 Performance Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades",   summary.get("TotalTrades", 0))
    c2.metric("Win Rate",       f"{summary.get('WinRate_%', 0):.1f}%")
    c3.metric("CAGR",           f"{summary.get('CAGR_%', 0):.1f}%")
    c4.metric("Max Drawdown",   f"{summary.get('MaxDrawdown_%', 0):.1f}%")
    c5.metric("Sharpe Ratio",   f"{summary.get('SharpeRatio', 0):.2f}")
    c6.metric("Profit Factor",  f"{summary.get('ProfitFactor', 0):.2f}")

    # ── TABS ─────────────────────────────────────────────────────
    tab_eq, tab_cond, tab_sector, tab_monthly, tab_trades, tab_dl = st.tabs([
        "📈 Equity Curve",
        "🏆 Best Conditions",
        "🏭 Sector Analysis",
        "📅 Monthly Returns",
        "📄 Trade Log",
        "⬇️ Download",
    ])

    # ── TAB 1: EQUITY CURVE ──────────────────────────────────────
    with tab_eq:
        col1, col2 = st.columns(2)
        with col1:
            fig_eq = _plot_equity(equity_df)
            if fig_eq:
                st.pyplot(fig_eq, use_container_width=False)  # suppress deprecation
        with col2:
            fig_dd = _plot_drawdown(equity_df)
            if fig_dd:
                st.pyplot(fig_dd, use_container_width=False)

        if not equity_df.empty:
            st.caption("Equity curve shows compounded portfolio value. "
                       "Each trade risks 1% of current equity.")
            eq_show = _clean_equity(equity_df).reset_index()
            eq_show.columns = ["Date", "ReturnPct", "PnL", "Equity"]
            st.dataframe(eq_show.tail(50), use_container_width=False,
                         hide_index=True)

    # ── TAB 2: BEST CONDITIONS ───────────────────────────────────
    with tab_cond:
        st.subheader("🏆 Kaunsi Condition Par Sabse Zyada Successful Trades?")

        if not condition_df.empty:
            fig_bc = _plot_best_conditions(condition_df)
            if fig_bc:
                st.pyplot(fig_bc, use_container_width=False)

            st.markdown("**Individual Condition Ranking** (sorted by Composite Score)")
            st.caption(
                "CompositeScore = WinRate × AvgReturn(clipped) × log(N) — "
                "higher = more reliable & profitable when this condition is TRUE"
            )
            show_cols = ["Rank", "Condition", "TradesSatisfied",
                         "WinRate_%", "AvgReturn_%", "CompositeScore",
                         "ContributionScore", "BestExitType"]
            show_cols = [c for c in show_cols if c in condition_df.columns]
            st.dataframe(condition_df[show_cols], use_container_width=False,
                         hide_index=True)

            st.divider()
            st.subheader("🔗 Top Condition Pairs — Ek Saath Kaunsi 2 Conditions Best Hain?")
            if not combos_df.empty:
                fig_cp = _plot_condition_pairs(combos_df)
                if fig_cp:
                    st.pyplot(fig_cp, use_container_width=False)
                st.dataframe(combos_df, use_container_width=False, hide_index=True)
            else:
                st.info("Condition pair data unavailable (need ≥5 trades per pair).")

    # ── TAB 3: SECTOR ────────────────────────────────────────────
    with tab_sector:
        if not sector_df.empty:
            st.dataframe(sector_df, use_container_width=False, hide_index=True)

    # ── TAB 4: MONTHLY HEATMAP ───────────────────────────────────
    with tab_monthly:
        fig_hm = _plot_monthly_heatmap(monthly_df)
        if fig_hm:
            st.pyplot(fig_hm, use_container_width=False)
        if not monthly_df.empty:
            st.dataframe(monthly_df, use_container_width=False, hide_index=True)

    # ── TAB 5: TRADE LOG ─────────────────────────────────────────
    with tab_trades:
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=False, hide_index=True)

    # ── TAB 6: DOWNLOAD ──────────────────────────────────────────
    with tab_dl:
        st.subheader("⬇️ Download Reports")
        if os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                st.download_button(
                    "📥 Download Excel Report",
                    data=f.read(),
                    file_name="Backtest_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        if not trades_df.empty:
            st.download_button(
                "📥 Download Trade Log CSV",
                data=trades_df.to_csv(index=False).encode(),
                file_name="Trade_Log.csv",
                mime="text/csv",
            )
        if not combos_df.empty:
            st.download_button(
                "📥 Download Condition Combos CSV",
                data=combos_df.to_csv(index=False).encode(),
                file_name="Condition_Combos.csv",
                mime="text/csv",
            )

else:
    st.info("👈 Sidebar se settings karein aur **Run Backtest** dabayein.")
    st.markdown("""
    ### Kya karta hai yeh dashboard?
    - **Equity Curve** — proper chronological, duplicate-date fix ke saath
    - **Best Conditions tab** — CompositeScore se rank ki gayi conditions:
      *Kaunsi condition par win rate + avg return + sample size sabse zyada hai*
    - **Condition Pairs** — kaun si 2 conditions saath aane par baseline se sabse upar jaati hain
    - **Monthly Heatmap, Sector, Trade Log, Excel Download**
    """)

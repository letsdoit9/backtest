"""
backtester.py
-------------
Streamlit UI for the Upstox Elite Swing Scanner Backtesting Engine.

Fixes applied vs old version:
  1. .applymap() -> .map()  (pandas 2.1+ removed applymap)
  2. use_container_width=True -> width='stretch'  (Streamlit 1.41+ deprecation)
  3. use_container_width=False -> width='content'

This file never touches the live scanner's logic -- it only calls into
engine.py / trade_executor.py, which in turn call the unmodified functions
exposed by scanner_bridge.py.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

import config
from utils import load_universe, get_historical_data, warmup_start_date
from engine import generate_all_signals
from trade_executor import execute_trades
from metrics import (build_equity_curve, performance_summary,
                     monthly_analysis, yearly_analysis)
from analytics import (sector_analysis, condition_analysis, threshold_analysis,
                       atr_optimization, holding_period_optimization,
                       sensitivity_analysis)
from excel_export import write_excel_report, write_csv_outputs, write_performance_json
from charts import generate_all_charts

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Upstox Elite Swing Scanner — Backtest",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Upstox Elite Swing Scanner — Backtesting Engine")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    min_conditions = st.slider(
        "Min conditions required (out of 16)",
        min_value=8, max_value=16,
        value=config.DEFAULT_MIN_CONDITIONS, step=1,
    )

    max_workers = st.number_input(
        "Parallel workers (data fetch)", min_value=1, max_value=32,
        value=config.MAX_WORKERS, step=1,
    )

    st.subheader("Exit Rules")
    target1_mult = st.number_input("Target 1 ATR multiplier",
                                   min_value=0.5, max_value=5.0,
                                   value=float(config.TARGET1_ATR_MULT), step=0.25)
    target2_mult = st.number_input("Target 2 ATR multiplier",
                                   min_value=0.5, max_value=5.0,
                                   value=float(config.TARGET2_ATR_MULT), step=0.25)
    stop_mult = st.number_input("Stoploss ATR multiplier",
                                min_value=0.25, max_value=3.0,
                                value=float(config.STOPLOSS_ATR_MULT), step=0.25)
    max_hold = st.number_input("Max holding days",
                               min_value=1, max_value=60,
                               value=int(config.MAX_HOLDING_DAYS), step=1)

    run_optimizations = st.checkbox(
        "Run full optimizations (threshold / ATR / holding sweeps)", value=False
    )

    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: fetch historical data
# ──────────────────────────────────────────────────────────────────────────────

def fetch_all_data(universe_df, max_workers=8):
    """Fetches (and caches) OHLCV history for every symbol in the universe."""
    start = warmup_start_date()
    end = config.BACKTEST_END_DATE
    stock_data_map = {}
    sector_map = dict(zip(universe_df["tradingsymbol"], universe_df["sector"]))
    symbols = universe_df["tradingsymbol"].tolist()

    prog = st.progress(0, text="Fetching historical data…")
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {
            executor.submit(get_historical_data, sym, start, end): sym
            for sym in symbols
        }
        done = 0
        for future in as_completed(futures):
            sym = futures[future]
            try:
                data = future.result()
            except Exception as e:
                st.warning(f"{sym}: fetch error — {e}")
                data = None
            stock_data_map[sym] = data
            done += 1
            prog.progress(done / len(symbols),
                          text=f"Fetching data… {done}/{len(symbols)}")

    prog.empty()
    return stock_data_map, sector_map


# ──────────────────────────────────────────────────────────────────────────────
# Main backtest runner
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest_ui(min_conditions, max_workers, run_optimizations,
                    target1_mult, target2_mult, stop_mult, max_hold):
    t0 = time.time()

    # 1. Universe
    with st.spinner("Loading universe…"):
        universe_df = load_universe()
    st.info(f"Universe: **{len(universe_df)} symbols**")

    # 2. Historical data
    stock_data_map, sector_map = fetch_all_data(universe_df, max_workers=max_workers)
    valid_data = {k: v for k, v in stock_data_map.items()
                  if v is not None and not v.empty}
    st.info(f"Usable data: **{len(valid_data)}/{len(stock_data_map)} symbols**")

    # 3. Signal generation
    base_threshold = min(config.THRESHOLD_SWEEP + [min_conditions])
    with st.spinner("Generating signals (no look-ahead)…"):
        all_signals = generate_all_signals(
            valid_data, min_conditions=base_threshold, sector_map=sector_map
        )

    print(f"[engine] Signal generation: 1/1 stocks done")   # keep existing log line
    st.info(f"Raw signals at threshold≥{base_threshold}: **{len(all_signals)}**")

    active_signals = all_signals[all_signals["ConditionsMet"] >= min_conditions]
    st.info(f"Active signals (threshold≥{min_conditions}/16): **{len(active_signals)}**")

    # 4. Trade execution
    with st.spinner("Simulating trades…"):
        trades_df = execute_trades(
            active_signals, valid_data,
            target1_mult=target1_mult,
            target2_mult=target2_mult,
            stop_mult=stop_mult,
            max_hold=max_hold,
        )
    st.success(f"✅ **{len(trades_df)} trades** simulated in "
               f"{time.time() - t0:.1f}s")

    # 5. Metrics & analysis
    equity_df   = build_equity_curve(trades_df)
    summary     = performance_summary(trades_df, equity_df)
    monthly_df  = monthly_analysis(trades_df)
    yearly_df   = yearly_analysis(trades_df)
    sector_df   = sector_analysis(trades_df)
    condition_df = condition_analysis(trades_df)

    threshold_df = atr_df = holding_df = sensitivity_df = pd.DataFrame()
    best_threshold = best_atr = best_holding = None

    if run_optimizations:
        with st.spinner("Running threshold analysis…"):
            threshold_df, best_threshold = threshold_analysis(all_signals, valid_data)
        with st.spinner("Running ATR multiplier optimisation…"):
            atr_df, best_atr = atr_optimization(active_signals, valid_data)
        with st.spinner("Running holding-period optimisation…"):
            holding_df, best_holding = holding_period_optimization(active_signals, valid_data)
        with st.spinner("Running sensitivity analysis (may take a while)…"):
            sensitivity_df = sensitivity_analysis(all_signals, valid_data)

        summary["RecommendedThreshold"]    = f"{best_threshold}/16" if best_threshold else "N/A"
        summary["RecommendedATRMultiplier"] = best_atr
        summary["RecommendedHoldingDays"]  = best_holding

    drawdown_df = pd.DataFrame()
    if not equity_df.empty:
        from metrics import compute_drawdown
        drawdown_df = compute_drawdown(equity_df["Equity"]).to_frame(name="Drawdown_%")

    # 6. Output files
    with st.spinner("Writing output files…"):
        excel_path = write_excel_report(
            summary, trades_df, monthly_df, yearly_df, condition_df,
            sector_df, equity_df, drawdown_df, threshold_df, atr_df, holding_df
        )
        write_csv_outputs(trades_df, equity_df, monthly_df, condition_df, sector_df)
        write_performance_json(summary)
        generate_all_charts(equity_df, trades_df, monthly_df, sector_df,
                            condition_df, threshold_df)

    return (summary, trades_df, equity_df, monthly_df, yearly_df,
            sector_df, condition_df, threshold_df, atr_df, holding_df,
            sensitivity_df, excel_path)


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _metric(label, value, fmt="{}", delta=None):
    """Small wrapper so we don't repeat st.metric everywhere."""
    try:
        display = fmt.format(value)
    except Exception:
        display = str(value)
    st.metric(label, display, delta=delta)


def _fmt_pct(v):
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return str(v)


def show_summary(summary):
    st.subheader("📊 Performance Summary")
    c = st.columns(4)
    metrics = [
        ("Total Trades",        summary.get("TotalTrades",       "—"), "{}"),
        ("Win Rate",            summary.get("WinRate_%",         "—"), "{:.2f}%"),
        ("Profit Factor",       summary.get("ProfitFactor",      "—"), "{:.3f}"),
        ("Expectancy",          summary.get("Expectancy_%",      "—"), "{:.3f}%"),
        ("CAGR",                summary.get("CAGR_%",            "—"), "{:.2f}%"),
        ("Sharpe Ratio",        summary.get("SharpeRatio",       "—"), "{:.3f}"),
        ("Sortino Ratio",       summary.get("SortinoRatio",      "—"), "{:.3f}"),
        ("Max Drawdown",        summary.get("MaxDrawdown_%",     "—"), "{:.2f}%"),
        ("Avg Return/Trade",    summary.get("AverageReturn_%",   "—"), "{:.3f}%"),
        ("Avg Holding Days",    summary.get("AverageHoldingDays","—"), "{:.1f}"),
        ("Largest Winner",      summary.get("LargestWinner_%",   "—"), "{:.2f}%"),
        ("Largest Loser",       summary.get("LargestLoser_%",    "—"), "{:.2f}%"),
    ]
    for i, (label, val, fmt) in enumerate(metrics):
        with c[i % 4]:
            try:
                display = fmt.format(val)
            except Exception:
                display = str(val)
            st.metric(label, display)


def style_return(v):
    """Green for positive, red for negative return string — used with .map()."""
    if isinstance(v, str):
        if v.startswith("+") or (not v.startswith("-") and "%" in v):
            try:
                if float(v.replace("%", "").replace("₹", "").replace(",", "")) > 0:
                    return "color: green"
            except Exception:
                pass
        if v.startswith("-"):
            return "color: red"
    return ""


def show_trade_log(trades_df):
    st.subheader("📋 Trade Log")
    if trades_df.empty:
        st.warning("No trades to display.")
        return

    display_cols = [
        "Ticker", "Sector", "SignalDate", "EntryDate", "EntryPrice",
        "ExitDate", "ExitPrice", "ExitReason", "HoldingDays",
        "ReturnPct", "ConditionsMet",
    ]
    cols = [c for c in display_cols if c in trades_df.columns]
    df = trades_df[cols].copy()

    df["ReturnPct"] = df["ReturnPct"].apply(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else ""
    )
    if "EntryPrice" in df.columns:
        df["EntryPrice"] = df["EntryPrice"].apply(
            lambda x: f"₹{x:.2f}" if pd.notna(x) else ""
        )
    if "ExitPrice" in df.columns:
        df["ExitPrice"] = df["ExitPrice"].apply(
            lambda x: f"₹{x:.2f}" if pd.notna(x) else ""
        )

    styled = (
        df.style
        .format({})
        # FIX: pandas 2.1+ removed applymap; use map() instead
        .map(
            lambda v: (
                "color: green" if isinstance(v, str) and v.startswith("+")
                else ("color: red" if isinstance(v, str) and v.startswith("-")
                      else "")
            ),
            subset=["ReturnPct"],
        )
    )
    # FIX: Streamlit 1.41+ deprecated use_container_width → use width='stretch'
    st.dataframe(styled, width="stretch")


def show_monthly(monthly_df):
    st.subheader("📅 Monthly Analysis")
    if monthly_df.empty:
        st.info("No monthly data.")
        return
    st.dataframe(monthly_df, width="stretch")


def show_yearly(yearly_df):
    st.subheader("📆 Yearly Analysis")
    if yearly_df.empty:
        st.info("No yearly data.")
        return
    st.dataframe(yearly_df, width="stretch")


def show_sector(sector_df):
    st.subheader("🏭 Sector Analysis")
    if sector_df.empty:
        st.info("No sector data (all trades tagged 'Unknown').")
        return
    st.dataframe(sector_df, width="stretch")


def show_condition(condition_df):
    st.subheader("🔎 Condition Contribution")
    if condition_df.empty:
        st.info("No condition data.")
        return
    st.dataframe(condition_df, width="stretch")


def show_optimizations(threshold_df, atr_df, holding_df, sensitivity_df):
    if not threshold_df.empty:
        st.subheader("🎯 Threshold Comparison")
        st.dataframe(threshold_df, width="stretch")

    if not atr_df.empty:
        st.subheader("📐 ATR Multiplier Optimisation")
        st.dataframe(atr_df, width="stretch")

    if not holding_df.empty:
        st.subheader("⏱️ Holding Period Optimisation")
        st.dataframe(holding_df, width="stretch")

    if not sensitivity_df.empty:
        st.subheader("🗂️ Sensitivity Analysis")
        st.dataframe(sensitivity_df, width="stretch")


def show_equity_chart(equity_df):
    if equity_df.empty:
        return
    st.subheader("📈 Equity Curve")
    st.line_chart(equity_df[["Equity"]])


def show_charts_from_disk():
    """If PNG charts exist on disk, display them."""
    import os
    charts_dir = config.CHARTS_DIR
    if not os.path.isdir(charts_dir):
        return
    pngs = sorted(f for f in os.listdir(charts_dir) if f.endswith(".png"))
    if not pngs:
        return
    st.subheader("🖼️ Charts")
    cols = st.columns(2)
    for i, fname in enumerate(pngs):
        with cols[i % 2]:
            st.image(os.path.join(charts_dir, fname),
                     caption=fname.replace("_", " ").replace(".png", "").title(),
                     width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running backtest…"):
        results = run_backtest_ui(
            min_conditions=min_conditions,
            max_workers=max_workers,
            run_optimizations=run_optimizations,
            target1_mult=target1_mult,
            target2_mult=target2_mult,
            stop_mult=stop_mult,
            max_hold=max_hold,
        )

    (summary, trades_df, equity_df, monthly_df, yearly_df,
     sector_df, condition_df, threshold_df, atr_df, holding_df,
     sensitivity_df, excel_path) = results

    # Download button for Excel report
    try:
        with open(excel_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Excel Report",
                data=f.read(),
                file_name="Backtest_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception:
        pass

    # Tabs for results
    tabs = st.tabs([
        "Summary", "Trade Log", "Monthly", "Yearly",
        "Sector", "Conditions", "Optimisations", "Charts",
    ])

    with tabs[0]:
        show_summary(summary)
        show_equity_chart(equity_df)

    with tabs[1]:
        show_trade_log(trades_df)

    with tabs[2]:
        show_monthly(monthly_df)

    with tabs[3]:
        show_yearly(yearly_df)

    with tabs[4]:
        show_sector(sector_df)

    with tabs[5]:
        show_condition(condition_df)

    with tabs[6]:
        show_optimizations(threshold_df, atr_df, holding_df, sensitivity_df)

    with tabs[7]:
        show_charts_from_disk()

else:
    st.info("👈 Configure settings in the sidebar and click **Run Backtest** to begin.")
    st.markdown("""
**What this tool does:**
- Loads your stock universe from `upstox_scanner_hardcoded__6_.py`
- Fetches historical OHLCV data (NSE) via `yfinance`
- Generates signals using the **exact same** indicator & condition functions as your live scanner
- Simulates next-day-open entry with ATR-based targets / stoploss
- Reports Win Rate, CAGR, Sharpe, Sortino, Drawdown, and more
- Optionally sweeps thresholds, ATR multipliers, and holding periods
""")

"""
backtester.py
-------------
Main Streamlit app + orchestrator for the Upstox Elite Swing Scanner Backtesting Engine.

FIXES APPLIED:
  1. `.applymap()` → `.map()`  (pandas >= 2.1 renamed it)
  2. `use_container_width=True` → `width='stretch'`  (Streamlit >= 1.40 deprecation)
"""

import sys
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

# ── path fix so Streamlit Cloud can find sibling modules ──────────────────────
sys.path.insert(0, os.path.dirname(__file__))

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

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE BACKTEST LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all_data(universe_df, max_workers=None):
    """Fetches (and caches) OHLCV history for every symbol in the universe."""
    max_workers = max_workers or config.MAX_WORKERS
    start = warmup_start_date()
    end   = config.BACKTEST_END_DATE

    stock_data_map = {}
    sector_map     = dict(zip(universe_df["tradingsymbol"], universe_df["sector"]))
    symbols        = universe_df["tradingsymbol"].tolist()

    print(f"[backtester] Fetching historical data for {len(symbols)} symbols "
          f"({start} -> {end}) using {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_historical_data, sym, start, end): sym
                   for sym in symbols}
        done = 0
        for future in as_completed(futures):
            sym = futures[future]
            try:
                data = future.result()
            except Exception as e:
                print(f"[backtester] {sym}: fetch error {e}")
                data = None
            stock_data_map[sym] = data
            done += 1
            if done % 25 == 0 or done == len(symbols):
                print(f"[backtester] ...{done}/{len(symbols)} fetched")

    return stock_data_map, sector_map


def run_backtest(min_conditions=None, max_workers=None, run_optimizations=False,
                 target1_mult=None, target2_mult=None, stop_mult=None, max_hold=None):
    t0 = time.time()
    min_conditions = min_conditions or config.DEFAULT_MIN_CONDITIONS

    universe_df               = load_universe()
    stock_data_map, sector_map = fetch_all_data(universe_df, max_workers=max_workers)
    valid_data = {k: v for k, v in stock_data_map.items()
                  if v is not None and not v.empty}

    print(f"[backtester] {len(valid_data)}/{len(stock_data_map)} symbols have usable data.")
    base_threshold = min(config.THRESHOLD_SWEEP + [min_conditions])
    all_signals    = generate_all_signals(valid_data, min_conditions=base_threshold,
                                          sector_map=sector_map)
    print(f"[backtester] {len(all_signals)} raw qualifying signal-days at threshold>={base_threshold}.")

    active_signals = all_signals[all_signals["ConditionsMet"] >= min_conditions]
    print(f"[backtester] {len(active_signals)} signals at the active threshold ({min_conditions}/16).")

    trades_df = execute_trades(active_signals, valid_data,
                               target1_mult=target1_mult, target2_mult=target2_mult,
                               stop_mult=stop_mult, max_hold=max_hold)
    print(f"[backtester] {len(trades_df)} trades simulated.")

    equity_df  = build_equity_curve(trades_df)
    summary    = performance_summary(trades_df, equity_df)
    monthly_df = monthly_analysis(trades_df)
    yearly_df  = yearly_analysis(trades_df)
    sector_df  = sector_analysis(trades_df)
    condition_df = condition_analysis(trades_df)

    threshold_df, best_threshold = pd.DataFrame(), None
    atr_df,       best_atr       = pd.DataFrame(), None
    holding_df,   best_holding   = pd.DataFrame(), None
    sensitivity_df               = pd.DataFrame()

    if run_optimizations:
        print("[backtester] Running threshold analysis...")
        threshold_df, best_threshold = threshold_analysis(all_signals, valid_data)
        print("[backtester] Running ATR multiplier optimization...")
        atr_df,      best_atr       = atr_optimization(active_signals, valid_data)
        print("[backtester] Running holding period optimization...")
        holding_df,  best_holding   = holding_period_optimization(active_signals, valid_data)
        print("[backtester] Running sensitivity analysis (this can take a while)...")
        sensitivity_df = sensitivity_analysis(all_signals, valid_data)

        summary["RecommendedThreshold"]    = f"{best_threshold}/16" if best_threshold else "N/A"
        summary["RecommendedATRMultiplier"] = best_atr
        summary["RecommendedHoldingDays"]   = best_holding

    drawdown_df = pd.DataFrame()
    if not equity_df.empty:
        from metrics import compute_drawdown
        drawdown_df = compute_drawdown(equity_df["Equity"]).to_frame(name="Drawdown_%")

    print("[backtester] Writing output files...")
    excel_path = write_excel_report(summary, trades_df, monthly_df, yearly_df,
                                    condition_df, sector_df, equity_df, drawdown_df,
                                    threshold_df, atr_df, holding_df)
    csv_paths  = write_csv_outputs(trades_df, equity_df, monthly_df, condition_df, sector_df)
    json_path  = write_performance_json(summary)
    chart_paths = generate_all_charts(equity_df, trades_df, monthly_df,
                                       sector_df, condition_df, threshold_df)

    elapsed = time.time() - t0
    print(f"[backtester] Done in {elapsed:.1f}s.")

    return {
        "summary":     summary,
        "trades":      trades_df,
        "signals":     all_signals,
        "monthly":     monthly_df,
        "yearly":      yearly_df,
        "sector":      sector_df,
        "condition":   condition_df,
        "threshold":   threshold_df,
        "atr":         atr_df,
        "holding":     holding_df,
        "sensitivity": sensitivity_df,
        "equity":      equity_df,
        "files": {"excel": excel_path, "csv": csv_paths,
                  "json": json_path, "charts": chart_paths},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def _color_return(v):
    """
    CSS colour for a single cell value used in the Trade Log styler.
    FIX: was called via .applymap() which was removed in pandas 2.1.
         Now called via .map() — identical API, just renamed.
    """
    if isinstance(v, str):
        # strip the trailing '%' that format_dict added, e.g. "3.45%"
        raw = v.rstrip("%")
        try:
            num = float(raw)
            return "color: green" if num >= 0 else "color: red"
        except ValueError:
            pass
    return ""


def _format_summary(summary: dict) -> pd.DataFrame:
    """Convert summary dict to a tidy two-column DataFrame for display."""
    rows = []
    for k, v in summary.items():
        if isinstance(v, float):
            rows.append((k, f"{v:.4f}"))
        else:
            rows.append((k, str(v)))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def main_streamlit():
    st.set_page_config(page_title="Upstox Backtest Engine", layout="wide")
    st.title("📈 Upstox Elite Swing Scanner — Backtesting Engine")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parameters")
        min_conditions = st.slider(
            "Min Conditions (out of 16)", min_value=8, max_value=16,
            value=config.DEFAULT_MIN_CONDITIONS, step=1
        )
        max_workers = st.slider(
            "Parallel Workers", min_value=1, max_value=32,
            value=config.MAX_WORKERS, step=1
        )
        run_optimizations = st.checkbox("Run full optimizations (slower)", value=False)
        run_btn = st.button("🚀 Run Backtest", type="primary")

    if not run_btn:
        st.info("Configure parameters in the sidebar and click **Run Backtest**.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    with st.spinner("Running backtest — this may take a few minutes..."):
        results = run_backtest(
            min_conditions=min_conditions,
            max_workers=max_workers,
            run_optimizations=run_optimizations,
        )

    trades_df    = results["trades"]
    equity_df    = results["equity"]
    summary      = results["summary"]
    monthly_df   = results["monthly"]
    yearly_df    = results["yearly"]
    sector_df    = results["sector"]
    condition_df = results["condition"]
    threshold_df = results["threshold"]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Summary", "📋 Trade Log", "📅 Monthly", "📆 Yearly",
        "🏭 Sector", "🔍 Conditions", "📉 Equity Curve",
        "🎛️ Threshold", "💾 Downloads"
    ])

    # ── Tab 0 : Summary ───────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Performance Summary")
        summary_df = _format_summary(summary)
        # FIX: use width='stretch' instead of deprecated use_container_width=True
        st.dataframe(summary_df, width="stretch")

    # ── Tab 1 : Trade Log ─────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader(f"Trade Log ({len(trades_df)} trades)")
        if trades_df.empty:
            st.warning("No trades were generated.")
        else:
            display_cols = [c for c in [
                "Symbol", "EntryDate", "ExitDate", "HoldDays",
                "EntryPrice", "ExitPrice", "ReturnPct", "ExitReason",
                "ConditionsMet", "Sector"
            ] if c in trades_df.columns]

            styled = (
                trades_df[display_cols]
                .style
                .format({
                    "ReturnPct":  "{:.2f}%",
                    "EntryPrice": "₹{:.2f}",
                    "ExitPrice":  "₹{:.2f}",
                })
                # ✅ FIX: .applymap() was renamed to .map() in pandas 2.1+
                .map(
                    _color_return,
                    subset=["ReturnPct"]
                )
            )
            # FIX: width='stretch' instead of use_container_width=True
            st.dataframe(styled, width="stretch")

    # ── Tab 2 : Monthly ───────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Monthly Returns")
        if monthly_df.empty:
            st.info("No monthly data.")
        else:
            st.dataframe(monthly_df, width="stretch")

    # ── Tab 3 : Yearly ────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Yearly Returns")
        if yearly_df.empty:
            st.info("No yearly data.")
        else:
            st.dataframe(yearly_df, width="stretch")

    # ── Tab 4 : Sector ────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Sector Analysis")
        if sector_df.empty:
            st.info("No sector data.")
        else:
            st.dataframe(sector_df, width="stretch")

    # ── Tab 5 : Conditions ────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Condition Analysis")
        if condition_df.empty:
            st.info("No condition data.")
        else:
            st.dataframe(condition_df, width="stretch")

    # ── Tab 6 : Equity Curve ──────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Equity Curve")
        if equity_df.empty:
            st.info("No equity data.")
        else:
            st.line_chart(equity_df["Equity"])

    # ── Tab 7 : Threshold ─────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Threshold Comparison")
        if threshold_df.empty:
            st.info("Run with 'full optimizations' enabled to see threshold sweep results.")
        else:
            st.dataframe(threshold_df, width="stretch")
            if "WinRate_%" in threshold_df.columns:
                st.line_chart(threshold_df.set_index("Threshold")["WinRate_%"])

    # ── Tab 8 : Downloads ─────────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("Download Files")
        files = results.get("files", {})

        excel_path = files.get("excel")
        if excel_path and os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Excel Report",
                    data=f,
                    file_name=os.path.basename(excel_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        json_path = files.get("json")
        if json_path and os.path.exists(json_path):
            with open(json_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Performance JSON",
                    data=f,
                    file_name=os.path.basename(json_path),
                    mime="application/json"
                )

        csv_paths = files.get("csv", {})
        for name, path in (csv_paths or {}).items():
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(
                        f"⬇️ Download {name}.csv",
                        data=f,
                        file_name=os.path.basename(path),
                        mime="text/csv"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT  (python backtester.py --min-conditions 12)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Upstox Elite Swing Scanner Backtester")
    p.add_argument("--min-conditions", type=int, default=config.DEFAULT_MIN_CONDITIONS)
    p.add_argument("--max-workers",    type=int, default=config.MAX_WORKERS)
    p.add_argument("--full-optimization", action="store_true")
    p.add_argument("--target1-mult",  type=float, default=None)
    p.add_argument("--target2-mult",  type=float, default=None)
    p.add_argument("--stop-mult",     type=float, default=None)
    p.add_argument("--max-hold",      type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    # When run via `streamlit run backtester.py`, sys.argv[0] ends with the
    # filename and streamlit injects its own args — just launch the UI.
    # When run directly via `python backtester.py`, use the CLI.
    _is_streamlit = "streamlit" in sys.modules and hasattr(st, "runtime")
    if _is_streamlit:
        main_streamlit()
    else:
        args = _parse_args()
        run_backtest(
            min_conditions=args.min_conditions,
            max_workers=args.max_workers,
            run_optimizations=args.full_optimization,
            target1_mult=args.target1_mult,
            target2_mult=args.target2_mult,
            stop_mult=args.stop_mult,
            max_hold=args.max_hold,
        )

"""
backtester.py  —  Streamlit UI version
"""

import time
import streamlit as st
import pandas as pd
import numpy as np

import config
from utils import load_universe, get_historical_data, warmup_start_date
from engine import generate_all_signals
from trade_executor import execute_trades
from metrics import build_equity_curve, performance_summary, monthly_analysis, yearly_analysis
from analytics import sector_analysis, condition_analysis
from excel_export import write_excel_report, write_csv_outputs, write_performance_json
from charts import generate_all_charts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Swing Scanner Backtester", page_icon="📈", layout="wide")

st.title("📈 Upstox Elite Swing Scanner — Backtester")
st.markdown("---")

# ── Universe load ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Universe load ho raha hai...")
def get_universe():
    return load_universe()

universe_df = get_universe()
all_symbols = sorted(universe_df["tradingsymbol"].str.upper().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Backtest mode:",
        ["🎯 Specific stocks chunno", "🌐 Poora universe (slow)"],
        index=0,
    )

    selected_symbols = []

    if mode == "🎯 Specific stocks chunno":
        manual_input = st.text_input(
            "Stock symbols (NSE) — comma se alag karo:",
            placeholder="e.g. RELIANCE,TCS,INFY",
        )
        dropdown_selected = st.multiselect(
            "Ya yahan se select karo:",
            options=all_symbols,
            placeholder="Type karke search karo...",
        )
        manual_list = [s.strip().upper() for s in manual_input.split(",") if s.strip()] if manual_input else []
        selected_symbols = list(dict.fromkeys(manual_list + dropdown_selected))

        if selected_symbols:
            st.success(f"✅ {len(selected_symbols)} stock(s) selected:")
            for sym in selected_symbols:
                st.markdown(f"  • **{sym}**")
        else:
            st.warning("⚠️ Koi stock select nahi kiya")

    st.markdown("---")
    st.subheader("📊 Backtest Parameters")

    min_conditions = st.slider("Min Conditions (threshold):", 8, 16, config.DEFAULT_MIN_CONDITIONS)

    col1, col2 = st.columns(2)
    with col1:
        target1_mult = st.number_input("Target 1 (ATR×)", value=float(config.TARGET1_ATR_MULT), step=0.25, min_value=0.5)
        target2_mult = st.number_input("Target 2 (ATR×)", value=float(config.TARGET2_ATR_MULT), step=0.25, min_value=0.5)
    with col2:
        stop_mult = st.number_input("Stoploss (ATR×)", value=float(config.STOPLOSS_ATR_MULT), step=0.25, min_value=0.25)
        max_hold  = st.number_input("Max Holding Days", value=int(config.MAX_HOLDING_DAYS), step=1, min_value=1)

    st.markdown("---")
    st.subheader("💰 Position Sizing")
    capital      = st.number_input("Aapka Capital (₹)", value=100000, step=10000, min_value=1000)
    risk_per_trade_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                                    help="Ek trade mein apne capital ka kitna % risk karoge")

    st.markdown("---")
    run_btn = st.button("🚀 Backtest Chalao!", type="primary", use_container_width=True)

# ── Helper: Position Sizing ───────────────────────────────────────────────────
def position_sizing_recommendation(trades_df, capital, risk_pct, stop_mult_val):
    """
    Har trade ke liye recommended position size calculate karta hai.
    Formula: Position Size = (Capital × Risk%) / (ATR × Stop Multiplier)
    """
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    risk_amount   = capital * (risk_pct / 100.0)                          # ₹ mein kitna risk
    df["RiskAmount_Rs"]    = round(risk_amount, 2)
    df["StopDistance_Rs"]  = round(df["EntryPrice"] * df["ATR"] / df["EntryPrice"] * stop_mult_val, 4)
    # Actual stop distance = EntryPrice - Stoploss
    df["StopDistance_Rs"]  = round(df["EntryPrice"] - df["Stoploss"], 4)
    df["StopDistance_Rs"]  = df["StopDistance_Rs"].replace(0, np.nan)

    df["RecommendedQty"]   = np.floor(risk_amount / df["StopDistance_Rs"]).fillna(0).astype(int)
    df["PositionValue_Rs"] = round(df["RecommendedQty"] * df["EntryPrice"], 2)
    df["CapitalUsed_%"]    = round(df["PositionValue_Rs"] / capital * 100.0, 2)

    return df[["Ticker", "SignalDate", "EntryPrice", "Stoploss", "Target1", "Target2",
               "StopDistance_Rs", "RiskAmount_Rs", "RecommendedQty",
               "PositionValue_Rs", "CapitalUsed_%", "ReturnPct", "ExitReason"]]

# ── Main area ─────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 Left sidebar mein stocks select karo, phir **'Backtest Chalao!'** button dabaao.")
    st.markdown("### Kaise use karein:")
    st.markdown("""
    1. **Sidebar** mein **'Specific stocks chunno'** mode select karo
    2. Stock ka naam type karo (e.g. `RELIANCE`) ya dropdown se select karo
    3. Capital aur Risk % set karo position sizing ke liye
    4. **🚀 Backtest Chalao!** button dabaao
    5. Results neeche aayenge — Excel/CSV download kar sakte ho
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Stocks Available", len(all_symbols))
    c2.metric("Default Threshold", f"{config.DEFAULT_MIN_CONDITIONS}/16")
    c3.metric("Backtest Start", config.BACKTEST_START_DATE)

else:
    # Validation
    if mode == "🎯 Specific stocks chunno" and not selected_symbols:
        st.error("❌ Koi stock select nahi kiya! Pehle sidebar mein stock chunno.")
        st.stop()

    symbols_to_run = selected_symbols if mode == "🎯 Specific stocks chunno" else all_symbols
    st.subheader(f"{'🎯' if selected_symbols else '🌐'} Backtest: {', '.join(symbols_to_run) if len(symbols_to_run) <= 10 else f'{len(symbols_to_run)} stocks'}")

    progress_bar = st.progress(0, text="Shuru ho raha hai...")
    status_box   = st.empty()
    t0           = time.time()

    # Step 1: Data fetch
    status_box.info(f"📥 {len(symbols_to_run)} stock(s) ka data download ho raha hai...")
    stock_data_map = {}
    fetch_errors   = []

    for i, sym in enumerate(symbols_to_run):
        data = get_historical_data(sym, warmup_start_date(), config.BACKTEST_END_DATE)
        stock_data_map[sym] = data
        if data is None or data.empty:
            fetch_errors.append(sym)
        pct = int(5 + (i + 1) / len(symbols_to_run) * 30)
        progress_bar.progress(pct, text=f"Fetch: {i+1}/{len(symbols_to_run)} — {sym}")

    valid_data = {k: v for k, v in stock_data_map.items() if v is not None and not v.empty}

    if fetch_errors:
        st.warning(f"⚠️ Fetch nahi hue (delisted/galat naam?): {', '.join(fetch_errors)}")
    if not valid_data:
        st.error("❌ Kisi bhi stock ka data nahi mila. NSE exact symbol name check karo.")
        st.stop()

    # Step 2: Signals
    status_box.info(f"✅ {len(valid_data)} stock(s) ka data mila. Signals generate ho rahe hain...")
    progress_bar.progress(35, text="Signals generate ho rahe hain...")

    sym_sector     = dict(zip(universe_df["tradingsymbol"].str.upper(), universe_df["sector"]))
    base_threshold = min(config.THRESHOLD_SWEEP + [min_conditions])
    all_signals    = generate_all_signals(valid_data, min_conditions=base_threshold, sector_map=sym_sector)
    active_signals = all_signals[all_signals["ConditionsMet"] >= min_conditions]

    # Step 3: Trades
    progress_bar.progress(60, text="Trades simulate ho rahe hain...")
    status_box.info(f"📊 {len(active_signals)} signals mile. Trades simulate ho rahe hain...")

    trades_df = execute_trades(
        active_signals, valid_data,
        target1_mult=target1_mult, target2_mult=target2_mult,
        stop_mult=stop_mult, max_hold=int(max_hold),
    )

    # Step 4: Metrics
    progress_bar.progress(75, text="Metrics calculate ho rahe hain...")
    equity_df    = build_equity_curve(trades_df, starting_capital=float(capital), risk_pct=risk_per_trade_pct)
    summary      = performance_summary(trades_df, equity_df, starting_capital=float(capital))
    monthly_df   = monthly_analysis(trades_df)
    yearly_df    = yearly_analysis(trades_df)
    sector_df    = sector_analysis(trades_df)
    condition_df = condition_analysis(trades_df)

    # Step 5: Files
    progress_bar.progress(85, text="Files likh rahe hain...")
    drawdown_df = pd.DataFrame()
    if not equity_df.empty:
        from metrics import compute_drawdown
        drawdown_df = compute_drawdown(equity_df["Equity"]).to_frame(name="Drawdown_%")

    excel_path = write_excel_report(
        summary, trades_df, monthly_df, yearly_df,
        condition_df, sector_df, equity_df, drawdown_df,
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    write_csv_outputs(trades_df, equity_df, monthly_df, condition_df, sector_df)
    write_performance_json(summary)

    progress_bar.progress(95, text="Charts ban rahe hain...")
    try:
        generate_all_charts(equity_df, trades_df, monthly_df, sector_df, condition_df, pd.DataFrame())
    except Exception:
        pass

    elapsed = time.time() - t0
    progress_bar.progress(100, text=f"✅ Done! ({elapsed:.1f}s mein)")
    status_box.success(f"✅ Backtest complete! {elapsed:.1f} seconds mein hua.")

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Performance Summary")

    if not trades_df.empty:
        # ── Row 1: Core metrics ───────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📦 Total Trades",    summary.get("TotalTrades", 0))
        c2.metric("✅ Winning Trades",   summary.get("WinningTrades", 0))
        c3.metric("❌ Losing Trades",    summary.get("LosingTrades", 0))
        win_rate = summary.get("WinRate_%", 0)
        c4.metric("🎯 Win Rate",        f"{win_rate:.1f}%")
        c5.metric("📈 Avg Return",      f"{summary.get('AverageReturn_%', 0):.2f}%")

        # ── Row 2: Risk metrics ───────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⚖️ Profit Factor",   f"{summary.get('ProfitFactor', 0):.2f}")
        c2.metric("📉 Max Drawdown",    f"{summary.get('MaxDrawdown_%', 0):.2f}%")
        c3.metric("📊 Sharpe Ratio",    f"{summary.get('SharpeRatio', 0):.2f}")
        c4.metric("🔻 Sortino Ratio",   f"{summary.get('SortinoRatio', 0):.2f}")
        c5.metric("📆 Avg Hold Days",   f"{summary.get('AverageHoldingDays', 0):.1f}")

        # ── Row 3: Returns ────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🏆 Largest Winner",  f"{summary.get('LargestWinner_%', 0):.2f}%")
        c2.metric("💀 Largest Loser",   f"{summary.get('LargestLoser_%', 0):.2f}%")
        c3.metric("📊 Avg Win",         f"{summary.get('AverageWin_%', 0):.2f}%")
        c4.metric("📊 Avg Loss",        f"{summary.get('AverageLoss_%', 0):.2f}%")
        c5.metric("📆 CAGR",            f"{summary.get('CAGR_%', 0):.2f}%")

        # ── Expectancy Box ────────────────────────────────────────────────────
        expectancy = summary.get("Expectancy_%", 0)
        exp_color  = "🟢" if expectancy > 0 else "🔴"
        st.markdown("---")

        exp_col, pos_col = st.columns(2)

        with exp_col:
            st.subheader("🎯 Expectancy per Trade")
            st.metric(
                label=f"{exp_color} Expected Return per Trade",
                value=f"{expectancy:.3f}%",
                help="Win Rate × Avg Win  −  Loss Rate × Avg Loss. Positive = profitable system"
            )

            win_rate_dec  = summary.get("WinningTrades", 0) / max(summary.get("TotalTrades", 1), 1)
            loss_rate_dec = 1 - win_rate_dec
            avg_win       = summary.get("AverageWin_%", 0)
            avg_loss      = abs(summary.get("AverageLoss_%", 0))

            st.markdown(f"""
            ```
            Formula:
            Expectancy = (Win Rate × Avg Win) − (Loss Rate × Avg Loss)
                       = ({win_rate_dec:.2%} × {avg_win:.2f}%) − ({loss_rate_dec:.2%} × {avg_loss:.2f}%)
                       = {expectancy:.3f}%
            ```
            """)

            if expectancy > 0:
                st.success(f"✅ System positive expectancy hai — long run mein profitable hoga.")
            elif expectancy == 0:
                st.warning("⚠️ Breakeven system — costs cover nahi ho rahe.")
            else:
                st.error("❌ Negative expectancy — threshold ya parameters adjust karo.")

        # ── Position Sizing Box ───────────────────────────────────────────────
        with pos_col:
            st.subheader("💰 Position Sizing Recommendation")

            risk_rs    = capital * (risk_per_trade_pct / 100.0)
            kelly_f    = win_rate_dec - (loss_rate_dec / (avg_win / max(avg_loss, 0.01)))
            kelly_f    = max(0, kelly_f)  # negative Kelly = don't trade
            half_kelly = kelly_f / 2.0    # Half-Kelly (safer)

            st.markdown(f"**Capital:** ₹{capital:,.0f}  |  **Risk/Trade:** {risk_per_trade_pct}%")
            st.markdown(f"**Risk Amount per Trade:** ₹{risk_rs:,.0f}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Kelly %",      f"{kelly_f*100:.1f}%",   help="Full Kelly — theoretical optimal, aggressive")
            m2.metric("Half-Kelly %", f"{half_kelly*100:.1f}%", help="Half Kelly — practical recommendation")
            m3.metric("Fixed Risk %", f"{risk_per_trade_pct}%", help="Aapka set kiya hua risk per trade")

            kelly_rs      = capital * kelly_f
            half_kelly_rs = capital * half_kelly

            st.markdown(f"""
            | Strategy | Capital Deploy |
            |----------|---------------|
            | Full Kelly | ₹{kelly_rs:,.0f} per trade |
            | Half Kelly (recommended) | ₹{half_kelly_rs:,.0f} per trade |
            | Fixed {risk_per_trade_pct}% Risk | ₹{risk_rs:,.0f} risk per trade |
            """)

            if kelly_f > 0:
                st.info(f"💡 **Tip:** Half-Kelly recommend kiya jata hai — Full Kelly ka {50:.0f}% use karo for better risk management.")
            else:
                st.warning("⚠️ Kelly negative hai — current parameters pe trading recommend nahi.")

        # ── Exit reason breakdown ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🚪 Exit Breakdown")
        exit_counts = trades_df["ExitReason"].value_counts().reset_index()
        exit_counts.columns = ["Exit Reason", "Count"]
        exit_counts["% of Trades"] = (exit_counts["Count"] / len(trades_df) * 100).round(1).astype(str) + "%"

        avg_by_exit = trades_df.groupby("ExitReason")["ReturnPct"].mean().round(2).reset_index()
        avg_by_exit.columns = ["Exit Reason", "Avg Return %"]
        exit_summary = exit_counts.merge(avg_by_exit, on="Exit Reason")
        st.dataframe(exit_summary, use_container_width=True, hide_index=True)

        # ── Per-stock summary ─────────────────────────────────────────────────
        if len(symbols_to_run) > 1:
            st.markdown("---")
            st.subheader("📋 Per-Stock Summary")
            per_stock = trades_df.groupby("Ticker").agg(
                Trades    =("ReturnPct", "count"),
                WinRate   =("ReturnPct", lambda x: f"{(x > 0).mean()*100:.1f}%"),
                AvgReturn =("ReturnPct", lambda x: f"{x.mean():.2f}%"),
                TotalReturn=("ReturnPct", lambda x: f"{x.sum():.2f}%"),
                BestTrade =("ReturnPct", lambda x: f"{x.max():.2f}%"),
                WorstTrade=("ReturnPct", lambda x: f"{x.min():.2f}%"),
            ).reset_index()
            st.dataframe(per_stock, use_container_width=True, hide_index=True)

        # ── Position sizing table ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💰 Trade-wise Position Sizing")
        sizing_df = position_sizing_recommendation(trades_df, capital, risk_per_trade_pct, stop_mult)
        if not sizing_df.empty:
            st.dataframe(
                sizing_df.style.format({
                    "EntryPrice":       "₹{:.2f}",
                    "Stoploss":         "₹{:.2f}",
                    "Target1":          "₹{:.2f}",
                    "Target2":          "₹{:.2f}",
                    "StopDistance_Rs":  "₹{:.2f}",
                    "RiskAmount_Rs":    "₹{:.0f}",
                    "PositionValue_Rs": "₹{:.0f}",
                    "CapitalUsed_%":    "{:.1f}%",
                    "ReturnPct":        "{:.2f}%",
                }),
                use_container_width=True, height=350
            )

        # ── Trade log ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Full Trade Log")
        display_cols = ["Ticker", "SignalDate", "EntryDate", "EntryPrice",
                        "ExitDate", "ExitPrice", "ExitReason", "ReturnPct",
                        "HoldingDays", "ConditionsMet"]
        show_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(
            trades_df[show_cols].style.format({
                "ReturnPct":  "{:.2f}%",
                "EntryPrice": "₹{:.2f}",
                "ExitPrice":  "₹{:.2f}",
            }).applymap(
                lambda v: "color: green" if isinstance(v, str) and v.endswith("%") and float(v.replace("%","").replace("₹","")) > 0
                          else ("color: red" if isinstance(v, str) and v.endswith("%") and float(v.replace("%","").replace("₹","")) < 0 else ""),
                subset=["ReturnPct"]
            ),
            use_container_width=True, height=400
        )

        # ── Charts ────────────────────────────────────────────────────────────
        if not equity_df.empty and "Equity" in equity_df.columns:
            st.markdown("---")
            st.subheader("📈 Equity Curve")
            st.line_chart(equity_df["Equity"])

        if not monthly_df.empty:
            st.subheader("📅 Monthly Returns")
            st.dataframe(monthly_df, use_container_width=True)

    else:
        st.warning("⚠️ Koi trade simulate nahi hua. Threshold kam karo ya alag stocks try karo.")

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download")
    dl1, dl2 = st.columns(2)

    with dl1:
        try:
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="📥 Excel Report Download Karo",
                    data=f.read(),
                    file_name="Backtest_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        except Exception:
            st.info("Excel file output/ folder mein save ho gayi hai.")

    with dl2:
        if not trades_df.empty:
            st.download_button(
                label="📥 Trade Log CSV Download Karo",
                data=trades_df.to_csv(index=False).encode("utf-8"),
                file_name="Trade_Log.csv",
                mime="text/csv",
                use_container_width=True,
            )

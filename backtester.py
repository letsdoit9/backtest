"""
backtester.py  —  Streamlit UI version
---------------------------------------
Streamlit par run hota hai. User pehle stocks select karta hai,
tab backtest chalta hai — poora universe fetch nahi hota.

Deploy: streamlit run backtester.py
"""

import time
import streamlit as st
import pandas as pd

import config
from utils import load_universe, get_historical_data, warmup_start_date
from engine import generate_all_signals
from trade_executor import execute_trades
from metrics import build_equity_curve, performance_summary, monthly_analysis, yearly_analysis
from analytics import sector_analysis, condition_analysis
from excel_export import write_excel_report, write_csv_outputs, write_performance_json
from charts import generate_all_charts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Scanner Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Upstox Elite Swing Scanner — Backtester")
st.markdown("---")

# ── Load universe (sirf symbol list chahiye, data nahi) ───────────────────────
@st.cache_data(show_spinner="Universe load ho raha hai...")
def get_universe():
    return load_universe()

universe_df = get_universe()
all_symbols = sorted(universe_df["tradingsymbol"].str.upper().tolist())

# ── Sidebar — settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Backtest mode:",
        ["🎯 Specific stocks chunno", "🌐 Poora universe (slow)"],
        index=0,
    )

    selected_symbols = None

    if mode == "🎯 Specific stocks chunno":
        manual_input = st.text_input(
            "Stock symbols (NSE) — comma se alag karo:",
            placeholder="e.g. RELIANCE,TCS,INFY,HDFCBANK",
            help="Exact NSE symbol likho. Multiple stocks: RELIANCE,TCS,INFY"
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

    min_conditions = st.slider(
        "Min Conditions (threshold):",
        min_value=8, max_value=16,
        value=config.DEFAULT_MIN_CONDITIONS,
        help="Kitne conditions meet hone chahiye signal ke liye"
    )

    col1, col2 = st.columns(2)
    with col1:
        target1_mult = st.number_input("Target 1 (ATR×)", value=float(config.TARGET1_ATR_MULT), step=0.25, min_value=0.5)
        target2_mult = st.number_input("Target 2 (ATR×)", value=float(config.TARGET2_ATR_MULT), step=0.25, min_value=0.5)
    with col2:
        stop_mult    = st.number_input("Stoploss (ATR×)", value=float(config.STOPLOSS_ATR_MULT), step=0.25, min_value=0.25)
        max_hold     = st.number_input("Max Holding Days", value=int(config.MAX_HOLDING_DAYS), step=1, min_value=1)

    st.markdown("---")
    run_btn = st.button("🚀 Backtest Chalao!", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────

if not run_btn:
    st.info("👈 Left sidebar mein stocks select karo, phir **'Backtest Chalao!'** button dabaao.")

    st.markdown("### Kaise use karein:")
    st.markdown("""
    1. **Sidebar** mein **'Specific stocks chunno'** mode select karo
    2. Stock ka naam type karo (e.g. `RELIANCE`) ya dropdown se select karo
    3. Parameters set karo (ya default rakhno)
    4. **🚀 Backtest Chalao!** button dabaao
    5. Results neeche aayenge — Excel/CSV download kar sakte ho
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks Available", len(all_symbols))
    with col2:
        st.metric("Default Threshold", f"{config.DEFAULT_MIN_CONDITIONS}/16")
    with col3:
        st.metric("Backtest Start", config.BACKTEST_START_DATE)

else:
    if mode == "🎯 Specific stocks chunno" and not selected_symbols:
        st.error("❌ Koi stock select nahi kiya! Pehle sidebar mein stock chunno.")
        st.stop()

    if mode == "🎯 Specific stocks chunno":
        symbols_to_run = selected_symbols
        st.subheader(f"🎯 Backtest: {', '.join(symbols_to_run)}")
    else:
        symbols_to_run = all_symbols
        st.subheader(f"🌐 Full Universe Backtest ({len(symbols_to_run)} stocks)")

    progress_bar = st.progress(0, text="Shuru ho raha hai...")
    status_box   = st.empty()
    t0           = time.time()

    status_box.info(f"📥 {len(symbols_to_run)} stock(s) ka data download ho raha hai...")
    progress_bar.progress(5, text="Data fetch ho raha hai...")

    start = warmup_start_date()
    end   = config.BACKTEST_END_DATE

    stock_data_map = {}
    fetch_errors   = []

    for i, sym in enumerate(symbols_to_run):
        data = get_historical_data(sym, start, end)
        stock_data_map[sym] = data
        if data is None or data.empty:
            fetch_errors.append(sym)
        pct = int(5 + (i + 1) / len(symbols_to_run) * 30)
        progress_bar.progress(pct, text=f"Fetch: {i+1}/{len(symbols_to_run)} — {sym}")

    valid_data = {k: v for k, v in stock_data_map.items() if v is not None and not v.empty}

    if fetch_errors:
        st.warning(f"⚠️ Yeh symbols fetch nahi hue (delisted/wrong name?): {', '.join(fetch_errors)}")

    if not valid_data:
        st.error("❌ Kisi bhi stock ka data nahi mila. Symbol names check karo (NSE exact name chahiye).")
        st.stop()

    status_box.info(f"✅ {len(valid_data)} stock(s) ka data mila. Signal generation shuru...")
    progress_bar.progress(35, text="Signals generate ho rahe hain...")

    sym_sector     = dict(zip(universe_df["tradingsymbol"].str.upper(), universe_df["sector"]))
    base_threshold = min(config.THRESHOLD_SWEEP + [min_conditions])
    all_signals    = generate_all_signals(valid_data, min_conditions=base_threshold, sector_map=sym_sector)
    active_signals = all_signals[all_signals["ConditionsMet"] >= min_conditions]

    progress_bar.progress(60, text="Trades simulate ho rahe hain...")
    status_box.info(f"📊 {len(active_signals)} signals mile. Trades simulate ho rahe hain...")

    trades_df = execute_trades(
        active_signals, valid_data,
        target1_mult=target1_mult,
        target2_mult=target2_mult,
        stop_mult=stop_mult,
        max_hold=int(max_hold),
    )

    progress_bar.progress(75, text="Metrics calculate ho rahe hain...")

    equity_df    = build_equity_curve(trades_df)
    summary      = performance_summary(trades_df, equity_df)
    monthly_df   = monthly_analysis(trades_df)
    yearly_df    = yearly_analysis(trades_df)
    sector_df    = sector_analysis(trades_df)
    condition_df = condition_analysis(trades_df)

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

    st.markdown("---")
    st.subheader("📊 Results")

    if not trades_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Trades",  len(trades_df))
        col2.metric("Win Rate",      f"{summary.get('WinRate_Pct', 0):.1f}%")
        col3.metric("Avg Return",    f"{summary.get('AvgReturn_Pct', 0):.2f}%")
        col4.metric("Profit Factor", f"{summary.get('ProfitFactor', 0):.2f}")
        col5.metric("Sharpe Ratio",  f"{summary.get('SharpeRatio', 0):.2f}")

        st.markdown("---")
        st.subheader("📋 Trade Log")
        display_cols = ["Ticker", "SignalDate", "EntryDate", "EntryPrice",
                        "ExitDate", "ExitPrice", "ExitReason", "ReturnPct",
                        "HoldingDays", "ConditionsMet"]
        show_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(
            trades_df[show_cols].style.format({
                "ReturnPct":  "{:.2f}%",
                "EntryPrice": "{:.2f}",
                "ExitPrice":  "{:.2f}",
            }),
            use_container_width=True, height=400
        )

        if not equity_df.empty and "Equity" in equity_df.columns:
            st.subheader("📈 Equity Curve")
            st.line_chart(equity_df["Equity"])

        if not monthly_df.empty:
            st.subheader("📅 Monthly Returns")
            st.dataframe(monthly_df, use_container_width=True)

    else:
        st.warning("⚠️ Koi trade simulate nahi hua. Threshold kam karo ya alag stocks try karo.")

    st.markdown("---")
    st.subheader("⬇️ Download")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
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

    with dl_col2:
        if not trades_df.empty:
            csv_data = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Trade Log CSV Download Karo",
                data=csv_data,
                file_name="Trade_Log.csv",
                mime="text/csv",
                use_container_width=True,
            )

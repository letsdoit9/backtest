"""
backtester.py
-------------
Main orchestrator for the Upstox Elite Swing Scanner Backtesting Engine.

Usage:
  # Poore universe par backtest (pehle jaisa)
  python backtester.py

  # Sirf ek stock par backtest
  python backtester.py --symbols RELIANCE

  # Multiple stocks par backtest (comma se alag karo, koi space nahi)
  python backtester.py --symbols RELIANCE,TCS,INFY,HDFCBANK

  # Symbols ke saath threshold bhi change kar sakte ho
  python backtester.py --symbols TATAMOTORS,BAJFINANCE --min-conditions 12

  # Custom workers
  python backtester.py --min-conditions 12 --max-workers 8

  # Full optimization suite
  python backtester.py --full-optimization
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
from utils import load_universe, get_historical_data, warmup_start_date
from engine import generate_all_signals
from trade_executor import execute_trades
from metrics import build_equity_curve, performance_summary, monthly_analysis, yearly_analysis
from analytics import (sector_analysis, condition_analysis, threshold_analysis,
                       atr_optimization, holding_period_optimization, sensitivity_analysis)
from excel_export import write_excel_report, write_csv_outputs, write_performance_json
from charts import generate_all_charts


def fetch_all_data(universe_df, max_workers=None, symbols_filter=None):
    """
    Fetches (and caches) OHLCV history for every symbol in the universe.

    symbols_filter: list of stock symbols — agar diya toh sirf unhi ko fetch karta hai.
                    None hone par poora universe fetch hota hai (pehle jaisa behavior).
    """
    max_workers = max_workers or config.MAX_WORKERS
    start = warmup_start_date()
    end   = config.BACKTEST_END_DATE

    # ── Universe filter ──────────────────────────────────────────────────────
    if symbols_filter:
        # Case-insensitive match, strip whitespace
        filter_upper = [s.strip().upper() for s in symbols_filter]
        filtered_df  = universe_df[
            universe_df["tradingsymbol"].str.upper().isin(filter_upper)
        ].copy()

        if filtered_df.empty:
            print(f"\n[backtester] ⚠️  Koi bhi symbol universe mein nahi mila: {filter_upper}")
            print(f"[backtester]    Universe mein available symbols check karo (CUSTOM_UNIVERSE_CSV ya live scanner).")
            # Fir bhi koshish karo — seedha yfinance se fetch hoga
            # Fake rows bana do taaki fetch ho sake
            filtered_df = pd.DataFrame({
                "tradingsymbol": filter_upper,
                "instrument_key": filter_upper,
                "sector": ["Unknown"] * len(filter_upper),
            })
        else:
            not_found = [s for s in filter_upper
                         if s not in filtered_df["tradingsymbol"].str.upper().tolist()]
            if not_found:
                print(f"[backtester] ⚠️  Yeh symbols universe mein nahi mile (phir bhi try karenge): {not_found}")
                extra = pd.DataFrame({
                    "tradingsymbol": not_found,
                    "instrument_key": not_found,
                    "sector": ["Unknown"] * len(not_found),
                })
                filtered_df = pd.concat([filtered_df, extra], ignore_index=True)

        print(f"\n[backtester] 🎯 Filter mode: Sirf {len(filtered_df)} symbol(s) par backtest hoga:")
        for sym in filtered_df["tradingsymbol"].tolist():
            print(f"             • {sym}")
        print()
        universe_df = filtered_df
    # ────────────────────────────────────────────────────────────────────────

    stock_data_map = {}
    sector_map     = dict(zip(universe_df["tradingsymbol"], universe_df["sector"]))
    symbols        = universe_df["tradingsymbol"].tolist()

    print(f"[backtester] Fetching historical data for {len(symbols)} symbol(s) "
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
                print(f"[backtester] {sym}: fetch error — {e}")
                data = None
            stock_data_map[sym] = data
            done += 1
            if done % 25 == 0 or done == len(symbols):
                print(f"[backtester] ...{done}/{len(symbols)} fetched")

    return stock_data_map, sector_map


def run_backtest(min_conditions=None, max_workers=None, run_optimizations=False,
                 target1_mult=None, target2_mult=None, stop_mult=None, max_hold=None,
                 symbols_filter=None):
    """
    symbols_filter: list of symbol strings, ya None for full universe.
    Baaki parameters pehle jaisi.
    """
    t0 = time.time()

    min_conditions = min_conditions or config.DEFAULT_MIN_CONDITIONS

    universe_df = load_universe()
    stock_data_map, sector_map = fetch_all_data(
        universe_df, max_workers=max_workers, symbols_filter=symbols_filter
    )

    valid_data = {k: v for k, v in stock_data_map.items()
                  if v is not None and not v.empty}

    print(f"[backtester] {len(valid_data)}/{len(stock_data_map)} symbols have usable data.")

    if not valid_data:
        print("[backtester] ❌ Koi bhi valid data nahi mila. Check karo ki symbol name sahi hai.")
        print("             NSE symbols hone chahiye (e.g. RELIANCE, TCS, INFY)")
        return {}

    print(f"[backtester] Generating signals at threshold>={min(config.THRESHOLD_SWEEP + [min_conditions])}...")

    base_threshold = min(config.THRESHOLD_SWEEP + [min_conditions])
    all_signals = generate_all_signals(
        valid_data, min_conditions=base_threshold, sector_map=sector_map
    )

    print(f"[backtester] {len(all_signals)} raw qualifying signal-days found at threshold>={base_threshold}.")

    active_signals = all_signals[all_signals["ConditionsMet"] >= min_conditions]
    print(f"[backtester] {len(active_signals)} signals at the active threshold ({min_conditions}/16).")

    trades_df = execute_trades(
        active_signals, valid_data,
        target1_mult=target1_mult, target2_mult=target2_mult,
        stop_mult=stop_mult, max_hold=max_hold
    )
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
    sensitivity_df = pd.DataFrame()

    if run_optimizations:
        print("[backtester] Running threshold analysis...")
        threshold_df, best_threshold = threshold_analysis(all_signals, valid_data)
        print("[backtester] Running ATR multiplier optimization...")
        atr_df, best_atr = atr_optimization(active_signals, valid_data)
        print("[backtester] Running holding period optimization...")
        holding_df, best_holding = holding_period_optimization(active_signals, valid_data)
        print("[backtester] Running sensitivity analysis (this can take a while)...")
        sensitivity_df = sensitivity_analysis(all_signals, valid_data)

        summary["RecommendedThreshold"]    = f"{best_threshold}/16" if best_threshold else "N/A"
        summary["RecommendedATRMultiplier"] = best_atr
        summary["RecommendedHoldingDays"]  = best_holding

    drawdown_df = pd.DataFrame()
    if not equity_df.empty:
        from metrics import compute_drawdown
        drawdown_df = compute_drawdown(equity_df["Equity"]).to_frame(name="Drawdown_%")

    print("[backtester] Writing output files...")
    excel_path  = write_excel_report(summary, trades_df, monthly_df, yearly_df,
                                     condition_df, sector_df, equity_df, drawdown_df,
                                     threshold_df, atr_df, holding_df)
    csv_paths   = write_csv_outputs(trades_df, equity_df, monthly_df, condition_df, sector_df)
    json_path   = write_performance_json(summary)
    chart_paths = generate_all_charts(equity_df, trades_df, monthly_df,
                                      sector_df, condition_df, threshold_df)

    elapsed = time.time() - t0
    print(f"\n[backtester] ✅ Done in {elapsed:.1f}s.")
    print(f"[backtester] Excel report : {excel_path}")
    print(f"[backtester] CSVs         : {csv_paths}")
    print(f"[backtester] JSON summary : {json_path}")
    print(f"[backtester] Charts       : {list(chart_paths.values())}")

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
        "files": {
            "excel":  excel_path,
            "csv":    csv_paths,
            "json":   json_path,
            "charts": chart_paths,
        },
    }


def _parse_args():
    p = argparse.ArgumentParser(
        description="Upstox Elite Swing Scanner Backtester",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python backtester.py
      → Poore universe par backtest

  python backtester.py --symbols RELIANCE
      → Sirf RELIANCE par backtest

  python backtester.py --symbols RELIANCE,TCS,INFY
      → Sirf in 3 stocks par backtest

  python backtester.py --symbols TATAMOTORS --min-conditions 12
      → TATAMOTORS, threshold 12/16
"""
    )
    p.add_argument(
        "--symbols", type=str, default=None,
        help=(
            "Comma-separated NSE symbols (e.g. RELIANCE,TCS,INFY).\n"
            "Default: poora universe (pehle jaisa behavior)."
        )
    )
    p.add_argument("--min-conditions", type=int, default=config.DEFAULT_MIN_CONDITIONS,
                   help="Minimum conditions required (10-16). Default: %(default)s")
    p.add_argument("--max-workers", type=int, default=config.MAX_WORKERS,
                   help="Parallel fetch workers. Default: %(default)s")
    p.add_argument("--full-optimization", action="store_true",
                   help="Threshold/ATR/holding-period/sensitivity sweeps bhi chalao (slow)")
    p.add_argument("--target1-mult",  type=float, default=None)
    p.add_argument("--target2-mult",  type=float, default=None)
    p.add_argument("--stop-mult",     type=float, default=None)
    p.add_argument("--max-hold",      type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # --symbols "RELIANCE,TCS,INFY" → ['RELIANCE', 'TCS', 'INFY']
    symbols_list = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else None
    )

    run_backtest(
        min_conditions=args.min_conditions,
        max_workers=args.max_workers,
        run_optimizations=args.full_optimization,
        target1_mult=args.target1_mult,
        target2_mult=args.target2_mult,
        stop_mult=args.stop_mult,
        max_hold=args.max_hold,
        symbols_filter=symbols_list,
    )

# Upstox Elite Swing Scanner — Backtesting Engine

A standalone backtesting package that **imports and reuses** your existing
scanner's indicator and condition functions without modifying a single line
of the live scanner logic.

## How it stays decoupled

- `scanner_bridge.py` is the **only** file that touches the live scanner. It
  loads `upstox_scanner_hardcoded__6_.py` from disk via `importlib` and
  re-exports, untouched:
  `calculate_indicators_ultra_fast`, `check_conditions_vectorized`,
  `fast_ema_calculation`, `fast_rsi_calculation`, `fast_stochrsi_calculation`,
  `fast_atr_calculation`, `fast_sma_calculation`, `load_hardcoded_stocks`.
- Every other module (`engine.py`, `trade_executor.py`, `metrics.py`,
  `analytics.py`, `charts.py`, `excel_export.py`) only calls into
  `scanner_bridge`. No indicator formula or condition threshold is
  redefined anywhere in the backtester.
- If you ever change the live scanner file, the backtester picks up the
  change automatically on next run — no edits needed here.

## Files

| File | Purpose |
|---|---|
| `scanner_bridge.py` | Loads the live scanner by file path, re-exports required functions |
| `config.py` | All tunable parameters (dates, thresholds, ATR multipliers, holding days, paths) |
| `utils.py` | Historical data fetch/cache (yfinance), universe & sector loading |
| `engine.py` | No-look-ahead signal generation per stock per day |
| `trade_executor.py` | Next-day-open entry, target/stop/time-exit simulation |
| `metrics.py` | Win rate, profit factor, Sharpe/Sortino/Calmar, CAGR, drawdown, etc. |
| `analytics.py` | Sector, condition, threshold, ATR, holding-period, sensitivity analysis |
| `charts.py` | All required PNG charts (matplotlib, Agg backend) |
| `excel_export.py` | `Backtest_Report.xlsx` + CSV/JSON outputs |
| `backtester.py` | CLI entry point that wires everything together |

## Setup

```bash
pip install pandas numpy yfinance openpyxl matplotlib streamlit aiohttp
```

Place `upstox_scanner_hardcoded__6_.py` (your live scanner) in the same
folder as this package — already done — or point to it elsewhere:

```bash
export SCANNER_PATH=/path/to/upstox_scanner_hardcoded__6_.py
```

## Run a backtest

```bash
# Basic run at the default threshold (10/16), Jan 1 2020 -> today
python backtester.py

# Custom threshold and more parallel workers
python backtester.py --min-conditions 13 --max-workers 16

# Full optimization suite: threshold sweep, ATR sweep, holding-period sweep,
# and a full sensitivity grid (slower — backtests every combination)
python backtester.py --full-optimization
```

Outputs land in `./output/`:

- `Backtest_Report.xlsx` (Summary, Trade Log, Monthly, Yearly, Condition
  Analysis, Sector Analysis, Equity Curve, Drawdown, Threshold Comparison,
  ATR Optimization, Holding Period Optimization sheets)
- `Trade_Log.csv`, `Equity_Curve.csv`, `Monthly_Returns.csv`,
  `Condition_Analysis.csv`, `Sector_Analysis.csv`
- `Performance_Summary.json`
- `output/charts/*.png` — equity curve, drawdown, monthly heatmap, rolling
  CAGR/Sharpe, return distribution, sector performance, condition
  contribution, threshold comparison

## Programmatic use

```python
from backtester import run_backtest

results = run_backtest(min_conditions=12, run_optimizations=True)
results["trades"]      # DataFrame of every simulated trade
results["summary"]     # dict of performance metrics
results["sensitivity"] # full parameter-sweep DataFrame
```

## Design notes / assumptions (documented, not hidden)

- **No look-ahead**: indicators are computed once per stock over the full
  causal history (every formula in the live scanner is already
  backward-looking — EMA/SMA/rolling RSI/ATR — so row *i* only reflects data
  through day *i*). The condition check for day *i* reads only row *i* and
  row *i-1*. Entries always execute at the **next day's Open**.
- **Condition gating vs. reporting**: the actual trade-qualifying decision
  (`ConditionsMet >= threshold`) comes 1:1 from `check_conditions_vectorized`
  exactly as the live scanner returns it. The 16 `ConditionN` boolean columns
  stored per trade are derived separately (same formulas, for reporting only)
  so the Condition Analysis sheet can show per-condition win rates —
  they never influence which signals qualify.
- **"Price" for daily-bar conditions** (e.g. "97% of today's high",
  "today's open") uses the daily Close/Open/High/Low of the signal day,
  the closest faithful analogue to the live scanner's real-time-quote logic
  when run on EOD historical bars.
- **Target/Stop hit detection** uses each day's intrabar High/Low range, with
  exit priority Stoploss → Target2 → Target1 → time exit, exactly as specified.
- **Equity curve** assumes a configurable fixed risk-per-trade compounding
  model (`metrics.build_equity_curve`, default 1% of equity per trade) — swap
  in your own position-sizing rule if needed via that one function.
- **Sector data** defaults to `"Unknown"` unless you supply a
  `SECTOR_MAP_CSV` (`symbol,sector`) or a universe CSV that already has a
  `sector` column.

"""
config.py
---------
Central configuration for the Upstox Elite Swing Scanner Backtesting Engine.

Nothing in this file touches the live scanner. It only defines parameters
used by the backtesting package itself.
"""

import os
from datetime import datetime

# --------------------------------------------------------------------------
# PATH TO THE LIVE SCANNER FILE
# --------------------------------------------------------------------------
# The backtester imports calculate_indicators_ultra_fast, check_conditions_vectorized,
# and the fast_* indicator functions directly from this file. It NEVER copies,
# rewrites, or modifies any logic inside it.
#
# Override with environment variable SCANNER_PATH if the file lives elsewhere.
SCANNER_PATH = os.environ.get(
    "SCANNER_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "upstox_scanner_hardcoded__6_.py")
)

# --------------------------------------------------------------------------
# BACKTEST PERIOD
# --------------------------------------------------------------------------
BACKTEST_START_DATE = "2020-01-01"
BACKTEST_END_DATE = datetime.today().strftime("%Y-%m-%d")  # "until today"

# Extra lookback so that indicators (SMA200/High_52w etc.) are warmed up
# BEFORE the official start date. This data is fetched but never used to
# generate signals before BACKTEST_START_DATE.
WARMUP_CALENDAR_DAYS = 400

# --------------------------------------------------------------------------
# UNIVERSE
# --------------------------------------------------------------------------
# By default re-uses load_hardcoded_stocks() from the live scanner.
# Can be overridden with a custom CSV having instrument_key,tradingsymbol[,sector]
CUSTOM_UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", None)

# Optional sector mapping CSV: symbol,sector  (used only for sector analysis)
SECTOR_MAP_CSV = os.environ.get("SECTOR_MAP_CSV", None)

# --------------------------------------------------------------------------
# SIGNAL THRESHOLD
# --------------------------------------------------------------------------
DEFAULT_MIN_CONDITIONS = 10
THRESHOLD_SWEEP = [10, 11, 12, 13, 14, 15, 16]

# Weighted scoring (mirrors the live scanner's optional weighted mode)
USE_WEIGHTED_SCORING = False
CONDITION_WEIGHTS = [1.2, 1.2, 1.0, 1.0, 1.1, 1.0, 0.8, 1.3,
                     1.4, 1.1, 0.9, 1.3, 0.8, 1.2, 1.0, 1.0]

CONDITION_NAMES = [
    "C1_Price>EMA5>EMA13>EMA26",
    "C2_Price>SMA50>SMA100>SMA200",
    "C3_RSI>55",
    "C4_StochRSI>50",
    "C5_MACD>Signal",
    "C6_Volume>100k_and_>VolSMA50",
    "C7_Price>Open",
    "C8_Price>=UpperBB",
    "C9_Price>High200x1.05",
    "C10_GapUp_LowGtPrevHigh",
    "C11_Price>=97pct_TodayHigh",
    "C12_Price>=95pct_52wHigh",
    "C13_ATR/Price<6pct",
    "C14_Vol>2xVolSMA50_and_Ret>2pct",
    "C15_Price>PrevClosex1.01",
    "C16_Volume>1.5xVolSMA50",
]

# --------------------------------------------------------------------------
# EXIT RULES
# --------------------------------------------------------------------------
TARGET1_ATR_MULT = 1.5
TARGET2_ATR_MULT = 2.0
STOPLOSS_ATR_MULT = 1.0
MAX_HOLDING_DAYS = 10

# Optimization sweeps
ATR_MULT_SWEEP = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
HOLDING_DAYS_SWEEP = [5, 7, 10, 12, 15, 20]

# --------------------------------------------------------------------------
# EXECUTION / PERFORMANCE
# --------------------------------------------------------------------------
MAX_WORKERS = int(os.environ.get("BT_MAX_WORKERS", 8))
USE_MULTIPROCESSING = os.environ.get("BT_USE_MP", "0") == "1"
RISK_FREE_RATE_ANNUAL = 0.065  # used for Sharpe/Sortino (India ~ T-bill rate); override as needed
TRADING_DAYS_PER_YEAR = 252

# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("BT_OUTPUT_DIR",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

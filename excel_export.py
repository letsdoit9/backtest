"""
excel_export.py  (UPDATED)
--------------------------
Writes Backtest_Report.xlsx (multi-sheet) plus the individual CSV/JSON
output files required by the spec.

CHANGES:
- write_excel_report() now accepts optional `condition_combos_df` parameter
  and writes it as a new "Best Condition Combos" sheet.
- Equity Curve sheet: index reset cleanly so duplicate-date rows from the
  fixed build_equity_curve land correctly.
- All sheet names kept ≤31 chars (Excel limit).
"""

import os
import json
import pandas as pd
import config


def _safe_write_sheet(writer, df, sheet_name):
    name = sheet_name[:31]
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        pd.DataFrame({"Info": ["No data"]}).to_excel(writer, sheet_name=name, index=False)
        return
    # If DatetimeIndex, reset so dates appear as a column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    df.to_excel(writer, sheet_name=name, index=False)


def write_excel_report(summary_dict, trades_df, monthly_df, yearly_df,
                       condition_df, sector_df, equity_df, drawdown_df,
                       threshold_df, atr_df, holding_df,
                       condition_combos_df=None,   # NEW
                       out_path=None):

    out_path = out_path or os.path.join(config.OUTPUT_DIR, "Backtest_Report.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary_rows = pd.DataFrame(list(summary_dict.items()), columns=["Metric", "Value"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _safe_write_sheet(writer, summary_rows,    "Summary")
        _safe_write_sheet(writer, trades_df,        "Trade Log")
        _safe_write_sheet(writer, monthly_df,       "Monthly")
        _safe_write_sheet(writer, yearly_df,        "Yearly")
        _safe_write_sheet(writer, condition_df,     "Condition Analysis")

        # NEW: Best condition combos sheet
        if condition_combos_df is not None and not condition_combos_df.empty:
            _safe_write_sheet(writer, condition_combos_df, "Best Condition Combos")

        _safe_write_sheet(writer, sector_df,        "Sector Analysis")
        _safe_write_sheet(writer, equity_df,        "Equity Curve")
        _safe_write_sheet(writer, drawdown_df,      "Drawdown")
        _safe_write_sheet(writer, threshold_df,     "Threshold Comparison")
        _safe_write_sheet(writer, atr_df,           "ATR Optimization")
        _safe_write_sheet(writer, holding_df,       "Holding Period Optim")

    return out_path


def write_csv_outputs(trades_df, equity_df, monthly_df, condition_df,
                      sector_df, condition_combos_df=None, out_dir=None):

    out_dir = out_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    def _csv(df, filename):
        p = os.path.join(out_dir, filename)
        if df is not None and not df.empty:
            df.to_csv(p, index=isinstance(df.index, pd.DatetimeIndex))
        return p

    paths["trade_log"]           = _csv(trades_df,           "Trade_Log.csv")
    paths["equity_curve"]        = _csv(equity_df,           "Equity_Curve.csv")
    paths["monthly_returns"]     = _csv(monthly_df,          "Monthly_Returns.csv")
    paths["condition_analysis"]  = _csv(condition_df,        "Condition_Analysis.csv")
    paths["sector_analysis"]     = _csv(sector_df,           "Sector_Analysis.csv")

    if condition_combos_df is not None and not condition_combos_df.empty:
        paths["condition_combos"] = _csv(condition_combos_df, "Condition_Combos.csv")

    return paths


def write_performance_json(summary_dict, out_path=None):
    out_path = out_path or os.path.join(config.OUTPUT_DIR, "Performance_Summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _clean(v):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return str(v)
        return v

    clean_dict = {k: _clean(v) for k, v in summary_dict.items()}
    with open(out_path, "w") as f:
        json.dump(clean_dict, f, indent=2, default=str)
    return out_path

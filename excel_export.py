"""
excel_export.py
----------------
Writes Backtest_Report.xlsx (multi-sheet) plus the individual CSV/JSON
output files required by the spec.
"""

import os
import json
import pandas as pd

import config


def _safe_write_sheet(writer, df, sheet_name):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        pd.DataFrame({"Info": ["No data"]}).to_excel(writer, sheet_name=sheet_name[:31], index=False)
        return
    df.to_excel(writer, sheet_name=sheet_name[:31], index=isinstance(df.index, pd.DatetimeIndex))


def write_excel_report(summary_dict, trades_df, monthly_df, yearly_df, condition_df,
                        sector_df, equity_df, drawdown_df, threshold_df,
                        atr_df, holding_df, out_path=None):
    out_path = out_path or os.path.join(config.OUTPUT_DIR, "Backtest_Report.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary_df = pd.DataFrame(list(summary_dict.items()), columns=["Metric", "Value"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _safe_write_sheet(writer, summary_df, "Summary")
        _safe_write_sheet(writer, trades_df, "Trade Log")
        _safe_write_sheet(writer, monthly_df, "Monthly")
        _safe_write_sheet(writer, yearly_df, "Yearly")
        _safe_write_sheet(writer, condition_df, "Condition Analysis")
        _safe_write_sheet(writer, sector_df, "Sector Analysis")
        _safe_write_sheet(writer, equity_df.reset_index() if not equity_df.empty else equity_df, "Equity Curve")
        _safe_write_sheet(writer, drawdown_df.reset_index() if not drawdown_df.empty else drawdown_df, "Drawdown")
        _safe_write_sheet(writer, threshold_df, "Threshold Comparison")
        _safe_write_sheet(writer, atr_df, "ATR Optimization")
        _safe_write_sheet(writer, holding_df, "Holding Period Optimization")

    return out_path


def write_csv_outputs(trades_df, equity_df, monthly_df, condition_df, sector_df, out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    paths["trade_log"] = os.path.join(out_dir, "Trade_Log.csv")
    trades_df.to_csv(paths["trade_log"], index=False)

    paths["equity_curve"] = os.path.join(out_dir, "Equity_Curve.csv")
    equity_df.to_csv(paths["equity_curve"])

    paths["monthly_returns"] = os.path.join(out_dir, "Monthly_Returns.csv")
    monthly_df.to_csv(paths["monthly_returns"], index=False)

    paths["condition_analysis"] = os.path.join(out_dir, "Condition_Analysis.csv")
    condition_df.to_csv(paths["condition_analysis"], index=False)

    paths["sector_analysis"] = os.path.join(out_dir, "Sector_Analysis.csv")
    sector_df.to_csv(paths["sector_analysis"], index=False)

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

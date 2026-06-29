"""
scanner_bridge.py
------------------
The ONLY file that touches the live scanner module. It loads the live
scanner file from disk via importlib (by path, not by package name, so the
scanner's filename doesn't need to be a valid import target) and exposes the
exact functions the backtester is allowed to reuse:

    calculate_indicators_ultra_fast
    check_conditions_vectorized
    fast_ema_calculation
    fast_rsi_calculation
    fast_stochrsi_calculation
    fast_atr_calculation
    fast_sma_calculation
    load_hardcoded_stocks   (universe loader, reused for convenience)

No line of the live scanner is copied, edited, or reimplemented here.
"""

import importlib.util
import os
import sys
import types

from config import SCANNER_PATH

_REQUIRED_NAMES = [
    "calculate_indicators_ultra_fast",
    "check_conditions_vectorized",
    "fast_ema_calculation",
    "fast_rsi_calculation",
    "fast_stochrsi_calculation",
    "fast_atr_calculation",
    "fast_sma_calculation",
]


def _load_scanner_module() -> types.ModuleType:
    if not os.path.exists(SCANNER_PATH):
        cwd_listing = []
        try:
            cwd_listing = sorted(os.listdir(os.path.dirname(SCANNER_PATH) or "."))
        except Exception:
            pass
        raise FileNotFoundError(
            f"\n\nLive scanner file not found at:\n  {SCANNER_PATH}\n\n"
            f"Files found in that directory instead:\n  {cwd_listing}\n\n"
            f"Fix this by either:\n"
            f"  1. Renaming/placing your scanner file at the path above, or\n"
            f"  2. Setting the SCANNER_PATH environment variable (or Streamlit secret) "
            f"to the correct absolute path of your scanner .py file.\n"
        )
    spec = importlib.util.spec_from_file_location("live_upstox_scanner", SCANNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load live scanner from '{SCANNER_PATH}'. "
            f"Set the SCANNER_PATH environment variable to the correct file path."
        )
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the scanner's own module-level code (e.g.
    # @st.cache_data decorators) resolves correctly.
    sys.modules["live_upstox_scanner"] = module
    spec.loader.exec_module(module)
    return module


_scanner_module = _load_scanner_module()

missing = [name for name in _REQUIRED_NAMES if not hasattr(_scanner_module, name)]
if missing:
    raise ImportError(
        f"Live scanner at '{SCANNER_PATH}' is missing expected function(s): {missing}. "
        f"The backtester requires these exact names to remain unchanged."
    )

# Re-export, untouched, exactly as implemented in the live scanner.
calculate_indicators_ultra_fast = _scanner_module.calculate_indicators_ultra_fast
check_conditions_vectorized = _scanner_module.check_conditions_vectorized
fast_ema_calculation = _scanner_module.fast_ema_calculation
fast_rsi_calculation = _scanner_module.fast_rsi_calculation
fast_stochrsi_calculation = _scanner_module.fast_stochrsi_calculation
fast_atr_calculation = _scanner_module.fast_atr_calculation
fast_sma_calculation = _scanner_module.fast_sma_calculation

# Optional convenience re-export (universe loader). Not part of the required
# indicator/condition set, but useful so the backtester can default to the
# same stock list as the live scanner without re-typing it.
load_hardcoded_stocks = getattr(_scanner_module, "load_hardcoded_stocks", None)

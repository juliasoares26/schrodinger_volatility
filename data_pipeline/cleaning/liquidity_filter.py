import numpy as np
import pandas as pd
from typing import Dict, Tuple


def apply_liquidity_filter(
    df: pd.DataFrame,
    min_volume: float = 10,
    min_open_interest: float = 100,
    min_dte: int   = 7,
    max_dte: int   = 400,
    min_moneyness: float = 0.75,
    max_moneyness: float = 1.30,
    min_iv: float = 0.03,
    max_iv: float = 2.00,
    require_all_liq: bool  = False,
) -> Tuple[pd.DataFrame, Dict]:
    n_start = len(df)
    df = df.copy()
    report: Dict = {}

    # DTE filter
    n_before = len(df)
    mask = (df["dte"] >= min_dte) & (df["dte"] <= max_dte)
    df = df[mask]
    report["dte_filter"] = int(n_before - len(df))

    # Moneyness filter
    n_before = len(df)
    mask = (df["moneyness"] >= min_moneyness) & (df["moneyness"] <= max_moneyness)
    df = df[mask]
    report["moneyness_filter"] = int(n_before - len(df))

    # IV sanity filter
    n_before = len(df)
    mask = (df["iv"] >= min_iv) & (df["iv"] <= max_iv) & df["iv"].notna()
    df = df[mask]
    report["iv_filter"] = int(n_before - len(df))

    # Volume / OI filter
    if min_volume > 0 or min_open_interest > 0:
        n_before = len(df)
        vol_ok = df["volume"] >= min_volume
        oi_ok = df["open_interest"] >= min_open_interest
        mask = (vol_ok & oi_ok) if require_all_liq else (vol_ok | oi_ok)
        df = df[mask]
        report["liquidity_filter"] = int(n_before - len(df))

    report["n_start"] = n_start
    report["n_end"] = len(df)
    report["n_dropped"] = n_start - len(df)

    print(f"\n  Liquidity filter: {n_start} → {len(df)} records")
    for k in ("dte_filter", "moneyness_filter", "iv_filter", "liquidity_filter"):
        if k in report:
            print(f"{k:<22}: {report[k]} dropped")

    return df.reset_index(drop=True), report
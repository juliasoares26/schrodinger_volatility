import numpy as np
import pandas as pd
from typing import Dict, Tuple


def apply_bid_ask_filter(
    df: pd.DataFrame,
    max_spread_pct: float = 0.15,
    min_bid: float = 0.05,
    require_positive_bid: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove options with excessively wide or zero bid-ask spreads

    Parameters
    df: raw option DataFrame (must have bid, ask, mid columns)
    max_spread_pct: max (ask - bid) / mid ratio allowed
    min_bid: minimum bid price accepted
    require_positive_bid: drop rows with bid <= 0

    Returns
    df_clean: pd.DataFrame
    report: dict with filter counts
    """
    n_start = len(df)
    df = df.copy()
    report: Dict = {}

    # Non-positive mid price — must come BEFORE spread_pct calculation
    n_before = len(df)
    mask = df["mid"] > 0
    df = df[mask]
    report["non_positive_mid"] = int(n_before - len(df))

    # Zero / negative bid
    if require_positive_bid:
        n_before = len(df)
        mask = df["bid"] > min_bid
        df = df[mask]
        report["zero_bid"] = int(n_before - len(df))

    # Crossed market (ask < bid)
    n_before = len(df)
    mask = df["ask"] >= df["bid"]
    df = df[mask]
    report["crossed_market"] = int(n_before - len(df))

    # Compute spread_pct now that mid > 0 is guaranteed
    if "spread_pct" not in df.columns:
        df = df.copy()
        df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]

    # Spread too wide
    n_before = len(df)
    mask = df["spread_pct"] <= max_spread_pct
    df = df[mask]
    report["wide_spread"] = int(n_before - len(df))

    report["n_start"] = n_start
    report["n_end"] = len(df)
    report["n_dropped"] = n_start - len(df)

    print(f"\n  Bid-ask filter: {n_start} → {len(df)} records")
    for k, v in report.items():
        if k.startswith("n_") or k in ("crossed_market", "wide_spread", "zero_bid", "non_positive_mid"):
            print(f"{k:<22}: {v}")

    return df.reset_index(drop=True), report
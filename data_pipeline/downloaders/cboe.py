import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
import warnings

warnings.filterwarnings("ignore")

from yfinance_loader import (
    get_risk_free_rate,
    get_dividend_yield,
    implied_vol,
    compute_forward,
)


def load_spx_cboe_csv(
    filepath,
    r: float = None,
    q: float = None,
) -> tuple:
    """
    Load SPX options from a CBOE delayed-quote CSV export

    Parameters
    filepath: str or Path
    r: float or None — fetched from Treasury if None
    q: float or None — fetched from SPY info if None

    Returns
    -------
    df: pd.DataFrame
    S0: float
    r: float
    q: float
    fetch_date: date
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CBOE CSV not found: {filepath}")

    print(f"\n Loading CBOE CSV: {filepath.name}")

    if r is None:
        r = get_risk_free_rate()
    if q is None:
        q = get_dividend_yield()

    # Detect spot price from header rows.
    # CBOE CSVs typically embed the spot in the first 1-2 lines as
    # "SPX,<spot>,..." or similar.  We scan only the first column of
    # each header row first (most reliable), then fall back to all cells
    raw = pd.read_csv(filepath, nrows=5, header=None)
    S0  = None
    SPX_LO, SPX_HI = 2000.0, 10000.0   # wider guard-band; still filters non-spot

    def _try_spot(val) -> float | None:
        try:
            v = float(str(val).replace(",", ""))
            return v if SPX_LO < v < SPX_HI else None
        except (ValueError, TypeError):
            return None

    # Pass 1: first cell of each row (column 0 in CBOE exports)
    for i in range(min(5, len(raw))):
        candidate = _try_spot(raw.iloc[i, 0])
        if candidate:
            S0 = candidate
            break

    # Pass 2: all cells, prefer values that look like an index level
    # (avoid large OI / volume integers by requiring a decimal component
    # OR by being in a tighter band typical of SPX)
    if S0 is None:
        for i in range(min(5, len(raw))):
            for val in raw.iloc[i]:
                raw_str = str(val)
                candidate = _try_spot(val)
                if candidate and ("." in raw_str or 3000 < candidate < 8000):
                    S0 = candidate
                    break
            if S0:
                break

    if S0 is None:
        raise ValueError(
            "Could not detect SPX spot from CBOE CSV header"
            "Check that the file starts with the spot price in the first column"
        )

    df_raw = pd.read_csv(filepath, skiprows=3, header=0)
    df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]

    print(f"  SPX spot (from CSV): {S0:.2f}")
    print(f"  Columns: {list(df_raw.columns)}")

    records    = []
    fetch_date = date.today()

    for _, row in df_raw.iterrows():
        for opt_type in ["call", "put"]:
            prefix = "calls_" if opt_type == "call" else "puts_"
            try:
                K = float(row.get("strike", row.get(prefix + "strike", np.nan)))
                exp_str = str(row.get("expiration_date", row.get("exp_date", "")))
                bid = float(row.get(prefix + "bid", row.get("bid", np.nan)))
                ask = float(row.get(prefix + "ask", row.get("ask", np.nan)))
                volume = float(row.get(prefix + "vol", row.get("volume", 0)) or 0)
                oi = float(row.get(prefix + "oi",  row.get("open_interest", 0)) or 0)
            except (TypeError, ValueError):
                continue

            if np.isnan(K) or np.isnan(bid) or np.isnan(ask):
                continue
            if bid <= 0 or ask <= bid:
                continue

            try:
                exp_date = pd.to_datetime(exp_str).date()
                dte = (exp_date - fetch_date).days
                T = dte / 365.0
            except (ValueError, TypeError):
                continue

            if T <= 0:
                continue

            mid = (bid + ask) / 2.0
            iv  = implied_vol(S0, K, T, mid, r, q, opt_type)
            if iv is None or np.isnan(iv):
                continue

            records.append({
                "fetch_date": fetch_date,
                "expiration": exp_date,
                "dte": dte,
                "T": T,
                "strike": K,
                "moneyness": K / S0,
                "log_moneyness": np.log(K / S0),
                "option_type": opt_type,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": ask - bid,
                "volume": volume,
                "open_interest": oi,
                "iv": iv,
                "S0": S0,
                "r": r,
                "q": q,
                "forward": compute_forward(S0, r, q, T),
            })

    df_out = pd.DataFrame(records)
    print(f"  Total raw records: {len(df_out)}")
    return df_out, S0, r, q, fetch_date
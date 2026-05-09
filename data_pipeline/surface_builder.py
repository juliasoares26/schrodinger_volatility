import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

# Fixed grid — must match generate_synthetic_data.py and feature_engineer.py

MONEYNESS_GRID = np.linspace(0.80, 1.20, 21)   # K/S0
MATURITY_GRID  = np.array([0.25, 0.50, 0.75, 1.00])


# Step 1: standard filters

def apply_standard_filters(
    df: pd.DataFrame,
    S0: float,
    max_spread_pct: float = 0.15,
    min_moneyness: float = 0.75,
    max_moneyness: float = 1.30,
    min_dte: int = 7,
    max_dte: int = 400,
    min_iv: float = 0.03,
    max_iv: float = 2.00,
    min_volume: float = 0,
    min_open_interest: float = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply standard liquidity and sanity filters

    Returns (filtered_df, counts_dict)
    """
    n_start = len(df)
    counts: Dict = {}
    df = df.copy()

    # Drop mid <= 0 first so spread_pct is well-defined
    df = df[df["mid"] > 0]

    if "spread_pct" not in df.columns:
        df["spread_pct"] = df["spread"] / df["mid"]

    # Spread
    mask = df["spread_pct"] <= max_spread_pct
    df = df[mask]; counts["spread"] = n_start - len(df)

    # Moneyness
    n_before = len(df)
    mask = (df["moneyness"] >= min_moneyness) & (df["moneyness"] <= max_moneyness)
    df   = df[mask]; counts["moneyness"] = n_before - len(df)

    # DTE
    n_before = len(df)
    mask = (df["dte"] >= min_dte) & (df["dte"] <= max_dte)
    df   = df[mask]; counts["dte"] = n_before - len(df)

    # IV sanity
    n_before = len(df)
    mask = (df["iv"] >= min_iv) & (df["iv"] <= max_iv) & df["iv"].notna()
    df   = df[mask]; counts["iv"] = n_before - len(df)

    # Volume / OI
    if min_volume > 0 or min_open_interest > 0:
        n_before = len(df)
        mask = (df["volume"] >= min_volume) | (df["open_interest"] >= min_open_interest)
        df   = df[mask]; counts["liquidity"] = n_before - len(df)

    print(f"\n  Standard filters: {n_start} → {len(df)} records")
    for name, dropped in counts.items():
        print(f"    {name:<12}: {dropped} dropped")

    return df.reset_index(drop=True), counts


# Step 2: select grid points

def select_surface_grid(
    df:              pd.DataFrame,
    moneyness_grid:  np.ndarray = None,
    maturity_grid:   np.ndarray = None,
    moneyness_tol:   float      = 0.05,
    prefer:          str        = "otm",
) -> pd.DataFrame:
    if moneyness_grid is None:
        moneyness_grid = MONEYNESS_GRID
    if maturity_grid is None:
        maturity_grid = MATURITY_GRID

    rows = []
    for T_target in maturity_grid:
        unique_T  = df["T"].unique()
        if len(unique_T) == 0:
            continue
        closest_T = unique_T[np.argmin(np.abs(unique_T - T_target))]
        if abs(closest_T - T_target) > 20 / 365:
            print(f"  Warning: no expiry near T={T_target:.2f}Y (closest {closest_T:.2f}Y)")

        df_T = df[np.abs(df["T"] - closest_T) < 1e-4].copy()

        for m_target in moneyness_grid:
            use_type = ("call" if m_target >= 1.0 else "put") if prefer == "otm" else prefer
            df_cell  = df_T[df_T["option_type"] == use_type].copy()
            if df_cell.empty:
                df_cell = df_T.copy()

            if df_cell.empty:
                rows.append({
                    "T_target": T_target, "T_actual": np.nan,
                    "moneyness_target": m_target, "moneyness_actual": np.nan,
                    "strike": np.nan, "iv": np.nan, "mid": np.nan,
                    "spread_pct": np.nan, "option_type": use_type,
                })
                continue

            dist     = np.abs(df_cell["moneyness"] - m_target)
            best_idx = dist.idxmin()
            best     = df_cell.loc[best_idx]
            iv_val   = best["iv"] if dist[best_idx] <= moneyness_tol else np.nan

            rows.append({
                "T_target": T_target,
                "T_actual": best["T"],
                "moneyness_target": m_target,
                "moneyness_actual": best["moneyness"],
                "strike": best["strike"],
                "iv": iv_val,
                "mid": best["mid"],
                "spread_pct": best["spread_pct"],
                "option_type": use_type,
            })

    return pd.DataFrame(rows)


# Step 3: build IV surface

def build_iv_surface(
    df_grid:  pd.DataFrame,
    moneyness_grid: np.ndarray = None,
    maturity_grid: np.ndarray = None,
    interpolate_missing: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Reshape grid DataFrame into a (n_maturities, n_strikes) IV matrix

    Returns
    vol_surface: ndarray shape (n_maturities, n_strikes)
    coverage: float  fraction of cells with valid IV
    """
    if moneyness_grid is None:
        moneyness_grid = MONEYNESS_GRID
    if maturity_grid is None:
        maturity_grid = MATURITY_GRID

    n_T, n_K = len(maturity_grid), len(moneyness_grid)
    vol_surface = np.full((n_T, n_K), np.nan)

    for i, T in enumerate(maturity_grid):
        for j, m in enumerate(moneyness_grid):
            mask = (
                (np.abs(df_grid["T_target"] - T) < 1e-9) &
                (np.abs(df_grid["moneyness_target"] - m) < 1e-9)
            )
            rows = df_grid[mask]
            if not rows.empty and not np.isnan(rows["iv"].iloc[0]):
                vol_surface[i, j] = rows["iv"].iloc[0]

    n_valid  = int(np.sum(~np.isnan(vol_surface)))
    coverage = n_valid / vol_surface.size
    print(f"\n  IV surface coverage: {n_valid}/{vol_surface.size} ({coverage*100:.1f}%)")

    if interpolate_missing and coverage > 0.5:
        vol_surface = _interpolate_surface(vol_surface, moneyness_grid, maturity_grid)
        n_after = int(np.sum(~np.isnan(vol_surface)))
        print(f"  After interpolation: {n_after}/{vol_surface.size} cells")

    return vol_surface, coverage


def _interpolate_surface(
    vol_surface:    np.ndarray,
    moneyness_grid: np.ndarray,
    maturity_grid:  np.ndarray,
) -> np.ndarray:
    """Fill NaN cells via 2D scattered linear interpolation in (√T, log(K/S0)) space"""
    log_m  = np.log(moneyness_grid)
    sqrt_T = np.sqrt(maturity_grid)

    valid_rows, valid_cols = np.where(~np.isnan(vol_surface))
    if len(valid_rows) < 4:
        return vol_surface

    x_fit = sqrt_T[valid_rows]
    y_fit = log_m[valid_cols]
    z_fit = vol_surface[valid_rows, valid_cols]

    filled = vol_surface.copy()
    n_T_valid = len(np.unique(valid_rows))
    n_K_valid = len(np.unique(valid_cols))

    if n_T_valid >= 2 and n_K_valid >= 2:
        try:
            points   = np.column_stack([x_fit, y_fit])
            lin_interp = LinearNDInterpolator(points, z_fit)
            nn_interp  = NearestNDInterpolator(points, z_fit)

            for i, T_val in enumerate(sqrt_T):
                for j, m_val in enumerate(log_m):
                    if np.isnan(filled[i, j]):
                        pt  = np.array([[T_val, m_val]])
                        val = lin_interp(pt)[0]
                        if np.isnan(val):
                            val = nn_interp(pt)[0]
                        filled[i, j] = float(val)
        except Exception:
            pass

    return filled
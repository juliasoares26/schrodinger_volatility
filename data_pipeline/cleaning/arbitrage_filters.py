import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


# 1. Static arbitrage (per-maturity)

def check_call_spread(prices: np.ndarray, strikes: np.ndarray) -> Dict:
    """
    Verify C(K1) >= C(K2) for K1 < K2 (call spreads non-negative)

    Returns dict with violation count and indices
    """
    violations = []
    for i in range(len(prices) - 1):
        spread = prices[i] - prices[i + 1]
        if spread < -1e-4:
            violations.append({
                "i": i,
                "K_low": float(strikes[i]),
                "K_high": float(strikes[i + 1]),
                "spread": float(spread),
            })
    return {
        "n_violations": len(violations),
        "is_arbitrage_free": len(violations) == 0,
        "violations": violations,
    }


def check_butterfly(prices: np.ndarray, strikes: np.ndarray) -> Dict:
    """
    Verify d²C/dK² >= 0 (butterfly spreads non-negative)
    This is equivalent to the risk-neutral density being non-negative
    """
    violations = []
    if len(prices) < 3:
        return {"n_violations": 0, "is_arbitrage_free": True, "violations": []}

    d2C = np.diff(prices, n=2)
    dK = np.diff(strikes)

    for i in range(len(d2C)):
        # Approximate second derivative: Δ²C / ΔK²
        dk2 = dK[i] * dK[i + 1] if i + 1 < len(dK) else dK[i] ** 2
        curvature = d2C[i] / (dk2 + 1e-12)
        if curvature < -1e-4:
            violations.append({
                "i": i + 1,
                "K_mid": float(strikes[i + 1]),
                "curvature": float(curvature),
            })
    return {
        "n_violations": len(violations),
        "is_arbitrage_free": len(violations) == 0,
        "violations": violations,
    }


def remove_static_arbitrage_iv(
    ivs: np.ndarray,
    moneyness: np.ndarray,
    T: float,
    S0: float,
    r: float = 0.0,
    q: float = 0.0,
    max_iterations: int = 10,
) -> np.ndarray:
    """
    Iteratively smooth out static arbitrage from an IV smile

    Strategy: if a butterfly violation is detected at strike K_mid, replace the IV with the average of its two neighbours. Repeat until no violations or max_iterations reached

    Returns cleaned IV array (same shape as input)
    """
    from scipy.stats import norm

    ivs = ivs.copy()

    def _bs_call(S, K, T_, sigma, r_, q_):
        if T_ <= 0 or sigma <= 0:
            return max(S * np.exp(-q_ * T_) - K * np.exp(-r_ * T_), 0.0)
        F = S * np.exp((r_ - q_) * T_)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T_) / (sigma * np.sqrt(T_))
        d2 = d1 - sigma * np.sqrt(T_)
        return np.exp(-r_ * T_) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    strikes = moneyness * S0

    for _ in range(max_iterations):
        prices = np.array([_bs_call(S0, K, T, iv, r, q)
                           for K, iv in zip(strikes, ivs)])
        result = check_butterfly(prices, strikes)
        if result["is_arbitrage_free"]:
            break
        for v in result["violations"]:
            idx = v["i"]
            if 0 < idx < len(ivs) - 1:
                ivs[idx] = 0.5 * (ivs[idx - 1] + ivs[idx + 1])

    return ivs


# 2. Calendar arbitrage (inter-maturity)

def check_calendar_arbitrage(
    vol_surface: np.ndarray,
    maturity_grid: np.ndarray,
) -> Dict:
    """
    Total variance w(K,T) = σ²(K,T)·T must be non-decreasing in T

    Parameters
    vol_surface  : (n_T, n_K) implied volatility matrix
    maturity_grid: (n_T,) maturities in years

    Returns dict with violation count and positions
    """
    n_T, n_K = vol_surface.shape
    total_var = vol_surface ** 2 * maturity_grid[:, None]  # (n_T, n_K)

    violations = []
    for i in range(n_T - 1):
        diff = total_var[i + 1, :] - total_var[i, :]
        bad = np.where(diff < -1e-6)[0]
        for j in bad:
            violations.append({
                "T_lower": float(maturity_grid[i]),
                "T_upper": float(maturity_grid[i + 1]),
                "strike_idx": int(j),
                "diff_total_var": float(diff[j]),
            })

    return {
        "n_violations": len(violations),
        "is_arbitrage_free": len(violations) == 0,
        "violations": violations,
        "total_variance": total_var,
    }


def fix_calendar_arbitrage(
    vol_surface: np.ndarray,
    maturity_grid: np.ndarray,
) -> np.ndarray:
    """
    Patch calendar arbitrage violations by linearly interpolating total variance between the nearest clean neighbours (before and after the violation block), then converting back to IV

    This avoids the kink produced by simple forward-filling (w(T_viol) = w(T_prev) + ε) and keeps the term structure smooth across the patched region

    Returns cleaned vol_surface (same shape)
    """
    surf = vol_surface.copy()
    n_T, n_K = surf.shape
    total_var = surf ** 2 * maturity_grid[:, None]   # (n_T, n_K)

    for j in range(n_K):
        tv = total_var[:, j].copy()

        # Iterate until no violations remain for this strike column
        changed = True
        while changed:
            changed = False
            for i in range(1, n_T):
                if tv[i] < tv[i - 1]:
                    # Find the next index k > i where tv[k] >= tv[i-1]
                    # (i.e. the first "clean" point after the violation block)
                    k = i + 1
                    while k < n_T and tv[k] < tv[i - 1]:
                        k += 1

                    if k < n_T:
                        # Linear interpolation in (T, w) space between
                        # the anchor before (i-1) and the anchor after (k)
                        T_lo, w_lo = maturity_grid[i - 1], tv[i - 1]
                        T_hi, w_hi = maturity_grid[k],     tv[k]
                        for m in range(i, k):
                            alpha = (maturity_grid[m] - T_lo) / (T_hi - T_lo)
                            tv[m] = w_lo + alpha * (w_hi - w_lo)
                    else:
                        # No clean point after — forward-fill with small epsilon
                        for m in range(i, n_T):
                            tv[m] = tv[m - 1] + 1e-6

                    changed = True
                    break   # restart scan after mutation

        # Write back IV from corrected total variance
        for i in range(n_T):
            if maturity_grid[i] > 0:
                surf[i, j] = np.sqrt(max(tv[i], 0.0) / maturity_grid[i])

    return surf


# 3. Put-Call Parity

def check_put_call_parity(
    df: pd.DataFrame,
    S0: float,
    r: float = 0.0,
    q: float = 0.0,
    iv_gap_threshold: float = 0.02,
    moneyness_grid_max: float = 1.20,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Flag options where the call IV and put IV at the same (K, T) differ by more than an adaptive threshold (PCP violation in IV terms)

    Parameters
    iv_gap_threshold: base threshold (used for ATM bucket, default 0.05)
    moneyness_grid_max: options above this moneyness are skipped (default 1.20)

    Returns filtered DataFrame (violations removed) and a report dict
    """
    if "option_type" not in df.columns or "iv" not in df.columns:
        return df, {"n_violations": 0, "is_arbitrage_free": True}

    # Moneyness column — compute if not present
    df = df.copy()
    if "moneyness" not in df.columns:
        df["moneyness"] = df["strike"] / S0

    # Adaptive threshold: pairs outside the surface grid are ignored;
    # within the grid, threshold scales with distance from ATM where
    # bid-ask spreads and skew both widen.
    # iv_gap_threshold is honoured as a ceiling on the adaptive values,
    # so a caller that passes a tight threshold (e.g. 0.02) gets strict
    # enforcement everywhere
    def _threshold(moneyness: float) -> float:
        m = abs(moneyness - 1.0)
        if m < 0.05:
            adaptive = 0.05
        elif m < 0.10:
            adaptive = 0.08
        elif m < 0.15:
            adaptive = 0.12
        else:
            adaptive = 0.20
        return min(adaptive, iv_gap_threshold)

    calls = df[df["option_type"] == "call"][["T", "strike", "iv", "moneyness"]].rename(
        columns={"iv": "iv_call", "moneyness": "moneyness_c"}
    )
    puts = df[df["option_type"] == "put"][["T", "strike", "iv"]].rename(
        columns={"iv": "iv_put"}
    )

    merged = pd.merge(calls, puts, on=["T", "strike"], how="inner")
    merged["iv_gap"] = np.abs(merged["iv_call"] - merged["iv_put"])

    # Skip pairs outside the surface grid — no impact on surface, inflates count
    merged_in_grid = merged[merged["moneyness_c"] <= moneyness_grid_max].copy()

    # Apply per-pair adaptive threshold
    merged_in_grid["threshold"] = merged_in_grid["moneyness_c"].apply(_threshold)
    violations = merged_in_grid[merged_in_grid["iv_gap"] > merged_in_grid["threshold"]]

    n_viol = len(violations)

    if n_viol > 0:
        viol_keys = set(zip(violations["T"], violations["strike"]))
        mask_keep = ~df.apply(
            lambda row: (row["T"], row["strike"]) in viol_keys, axis=1
        )
        df_clean = df[mask_keep].reset_index(drop=True)
    else:
        df_clean = df

    max_gap = float(merged_in_grid["iv_gap"].max()) if len(merged_in_grid) > 0 else 0.0
    report = {
        "n_violations": n_viol,
        "is_arbitrage_free": n_viol == 0,
        "max_iv_gap": max_gap,
        "mean_iv_gap": float(merged_in_grid["iv_gap"].mean()) if len(merged_in_grid) > 0 else 0.0,
        "n_skipped_outside_grid": len(merged) - len(merged_in_grid),
    }

    if n_viol > 0:
        print(f"\n  PCP violations: {n_viol} (K,T) pairs removed "
              f"(max gap={max_gap*100:.2f}%)")

    return df_clean, report


# 4. Combined entry point

def filter_surface(
    df: pd.DataFrame,
    vol_surface: np.ndarray,
    maturity_grid: np.ndarray,
    moneyness_grid: np.ndarray,
    S0: float = 100.0,
    r: float = 0.0,
    q: float = 0.0,
    run_static:   bool = True,
    run_calendar: bool = True,
    run_pcp:      bool = True,
    iv_gap_threshold: float = 0.02,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Apply all arbitrage filters to a vol surface and option DataFrame

    Parameters
    df: raw option DataFrame (from data_loader / preprocessor)
    vol_surfae: (n_T, n_K) IV matrix on fixed grid
    maturity_grid: (n_T,) maturities in years
    moneyness_grid: (n_K,) K/S0 ratios

    Returns
    df_clean: filtered option DataFrame
    vol_surface: arbitrage-free IV surface (patched in place)
    report: dict summarising all violations found/fixed
    """
    report: Dict = {}
    df_clean = df.copy() if df is not None else pd.DataFrame()

    print("\nArbitrage Filters")

    # Static (per-maturity)
    if run_static:
        static_violations = 0
        surf = vol_surface.copy()

        for i, T in enumerate(maturity_grid):
            ivs_row = surf[i, :].copy()
            if np.all(np.isnan(ivs_row)):
                continue
            # Interpolate NaNs before check
            valid = ~np.isnan(ivs_row)
            if valid.sum() < 2:
                continue
            ivs_row[~valid] = np.interp(
                np.where(~valid)[0],
                np.where(valid)[0],
                ivs_row[valid],
            )
            ivs_clean = remove_static_arbitrage_iv(
                ivs_row, moneyness_grid, T, S0, r, q
            )
            n_changed = int(np.sum(np.abs(ivs_clean - ivs_row) > 1e-6))
            static_violations += n_changed
            surf[i, :] = ivs_clean

        vol_surface = surf
        report["static_violations_fixed"] = static_violations
        print(f"  Static  : {static_violations} cells patched")

    # Calendar
    if run_calendar:
        cal_report = check_calendar_arbitrage(vol_surface, maturity_grid)
        if not cal_report["is_arbitrage_free"]:
            vol_surface = fix_calendar_arbitrage(vol_surface, maturity_grid)
        report["calendar_violations"] = cal_report["n_violations"]
        print(f"  Calendar: {cal_report['n_violations']} violations fixed")

    # Put-Call Parity
    if run_pcp and len(df_clean) > 0:
        df_clean, pcp_report = check_put_call_parity(
            df_clean, S0=S0, r=r, q=q,
            iv_gap_threshold=iv_gap_threshold,
        )
        report["pcp_violations"] = pcp_report["n_violations"]
        report["pcp_max_iv_gap"] = pcp_report["max_iv_gap"]
        print(f"PCP: {pcp_report['n_violations']} pairs removed")

    total = (
        report.get("static_violations_fixed", 0)
        + report.get("calendar_violations", 0)
        + report.get("pcp_violations", 0)
    )
    report["total_fixes"] = total
    report["is_clean"] = total == 0
    print(f"Total: {total} fixes applied")

    return df_clean, vol_surface, report


# 5. Diagnostics

def surface_diagnostics(
    vol_surface: np.ndarray,
    maturity_grid: np.ndarray,
    moneyness_grid: np.ndarray,
) -> Dict:
    """
    Print and return summary statistics for a vol surface.
    """
    n_nan = int(np.sum(np.isnan(vol_surface)))
    n_total = vol_surface.size
    atm_idx = len(moneyness_grid) // 2

    diag = {
        "n_nan": n_nan,
        "coverage": float(1 - n_nan / n_total),
        "min_iv": float(np.nanmin(vol_surface)),
        "max_iv": float(np.nanmax(vol_surface)),
        "mean_iv": float(np.nanmean(vol_surface)),
        "atm_ivs": {
            float(T): float(vol_surface[i, atm_idx])
            for i, T in enumerate(maturity_grid)
            if not np.isnan(vol_surface[i, atm_idx])
        },
    }

    # Check if total variance is non-decreasing at ATM
    total_var_atm = vol_surface[:, atm_idx] ** 2 * maturity_grid
    is_cal_free_atm = bool(np.all(np.diff(total_var_atm) >= -1e-6))
    diag["is_calendar_arb_free_atm"] = is_cal_free_atm

    print("\nSurface Diagnostics")
    print(f"Coverage: {diag['coverage']*100:.1f}%  ({n_nan} NaN cells)")
    print(f"IV range: [{diag['min_iv']*100:.2f}%, {diag['max_iv']*100:.2f}%]")
    print(f"ATM IVs:")
    for T, iv in diag["atm_ivs"].items():
        print(f"T={T:.2f}Y  →  {iv*100:.2f}%")
    print(f"Calendar-arb free (ATM): {is_cal_free_atm}")

    return diag


# Smoke test

if __name__ == "__main__":
    print("arbitrage_filter.py — smoke test\n")

    np.random.seed(42)
    n_T, n_K = 4, 21
    maturity_grid  = np.array([0.25, 0.50, 0.75, 1.00])
    moneyness_grid = np.linspace(0.80, 1.20, n_K)

    # Synthetic surface: ATM IV grows monotonically with T (term structure convex)
    vol_surface = np.zeros((n_T, n_K))
    for i, T in enumerate(maturity_grid):
        atm = 0.18 + 0.03 * i          # 18%, 21%, 24%, 27% — clearly increasing
        skew = -0.10 * np.log(moneyness_grid)
        vol_surface[i, :] = atm + skew

    # Introduce a strong calendar violation: T=0.50 slice has LOWER total var
    # than T=0.25 at the ATM strike (index 10, moneyness=1.0)
    # total_var[0, 10] = 0.18² * 0.25 = 0.0081
    # Setting vol_surface[1, 10] = 0.15 → total_var = 0.15² * 0.50 = 0.01125 > 0.0081 (ok)
    # So we need tv[1] < tv[0]: σ[1] < sqrt(tv[0] / T[1]) = sqrt(0.0081/0.50) = 0.1273
    vol_surface[1, 10] = 0.10  # total_var = 0.01 * 0.50 = 0.005 < 0.0081  violation

    cal = check_calendar_arbitrage(vol_surface, maturity_grid)
    print(f"Calendar violations before fix: {cal['n_violations']}")

    vol_fixed = fix_calendar_arbitrage(vol_surface, maturity_grid)
    cal2 = check_calendar_arbitrage(vol_fixed, maturity_grid)
    print(f"Calendar violations after fix: {cal2['n_violations']}")

    surface_diagnostics(vol_fixed, maturity_grid, moneyness_grid)

    # filter_surface
    df_empty = pd.DataFrame()
    _, surf_out, report = filter_surface(
        df_empty, vol_surface.copy(), maturity_grid, moneyness_grid,
        run_pcp=False,
    )
    print(f"\nReport: {report}")
    assert report["calendar_violations"] > 0, "Expected calendar violations to be detected"
    assert report["is_clean"] == (report["total_fixes"] == 0), "is_clean inconsistency"
    print("\nSmoke test passed")
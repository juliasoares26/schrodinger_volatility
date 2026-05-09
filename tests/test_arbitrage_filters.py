import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'data_pipeline' / 'cleaning'))

from arbitrage_filter import (
    check_call_spread,
    check_butterfly,
    check_calendar_arbitrage,
    fix_calendar_arbitrage,
    check_put_call_parity,
    filter_surface,
    surface_diagnostics,
    remove_static_arbitrage_iv,
)


# Helpers

MATURITY_GRID  = np.array([0.25, 0.50, 0.75, 1.00])
MONEYNESS_GRID = np.linspace(0.80, 1.20, 21)


def _flat_surface(iv: float = 0.20) -> np.ndarray:
    """Arbitrage-free flat surface."""
    n_T = len(MATURITY_GRID)
    n_K = len(MONEYNESS_GRID)
    surf = np.zeros((n_T, n_K))
    for i, T in enumerate(MATURITY_GRID):
        atm = iv + 0.005 * i   # slight term-structure to ensure total var non-decreasing
        skew = -0.05 * np.log(MONEYNESS_GRID)
        surf[i] = atm + skew
    return np.clip(surf, 0.05, 2.0)


def _bs_call_prices(S0, strikes, T, ivs, r=0.0):
    from scipy.stats import norm
    prices = []
    for K, iv in zip(strikes, ivs):
        if T <= 0 or iv <= 0:
            prices.append(max(S0 - K, 0.0))
            continue
        d1 = (np.log(S0/K) + 0.5*iv**2*T) / (iv*np.sqrt(T))
        d2 = d1 - iv*np.sqrt(T)
        prices.append(S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
    return np.array(prices)

# Call spread

class TestCallSpread:

    def test_no_violation_on_clean_data(self):
        strikes = np.array([90., 95., 100., 105., 110.])
        prices  = np.array([15., 10.,  6.,   3.,   1.])  # decreasing — OK
        result  = check_call_spread(prices, strikes)
        assert result["is_arbitrage_free"]
        assert result["n_violations"] == 0

    def test_detects_violation(self):
        strikes = np.array([90., 95., 100., 105.])
        prices  = np.array([10., 12.,  6.,   3.])  # prices[1] > prices[0] — violation
        result  = check_call_spread(prices, strikes)
        assert not result["is_arbitrage_free"]
        assert result["n_violations"] > 0



# Butterfly


class TestButterfly:

    def test_convex_prices_ok(self):
        """Linearly decreasing prices are convex, no violations"""
        strikes = np.linspace(80, 120, 10)
        prices  = np.linspace(20, 1, 10)
        result  = check_butterfly(prices, strikes)
        assert result["is_arbitrage_free"]

    def test_detects_concavity(self):
        """Introduce a concave kink."""
        strikes = np.array([90., 95., 100., 105., 110.])
        prices  = np.array([15.,  9.,  10.,   4.,   1.])  # kink at 100
        result  = check_butterfly(prices, strikes)
        assert result["n_violations"] > 0


# Calendar arbitrage

class TestCalendarArbitrage:

    def test_clean_surface_ok(self):
        surf   = _flat_surface()
        result = check_calendar_arbitrage(surf, MATURITY_GRID)
        assert result["is_arbitrage_free"], f"Unexpected violations: {result['violations']}"

    def test_detects_violation(self):
        surf = _flat_surface()
        # Introduce violation: total variance at T1 > T2 for one strike
        surf[1, 10] = surf[0, 10] * 0.5  # reduce IV so total var decreases
        result = check_calendar_arbitrage(surf, MATURITY_GRID)
        assert result["n_violations"] > 0

    def test_fix_removes_violations(self):
        surf = _flat_surface()
        surf[1, 10] = surf[0, 10] * 0.5
        surf_fixed = fix_calendar_arbitrage(surf, MATURITY_GRID)
        result = check_calendar_arbitrage(surf_fixed, MATURITY_GRID)
        assert result["is_arbitrage_free"], (
            f"Violations remain after fix: {result['violations']}"
        )

    def test_total_variance_non_decreasing_after_fix(self):
        surf = _flat_surface()
        surf[2, :5] *= 0.3   # multiple violations
        surf_fixed  = fix_calendar_arbitrage(surf, MATURITY_GRID)
        total_var   = surf_fixed ** 2 * MATURITY_GRID[:, None]
        diffs = np.diff(total_var, axis=0)
        assert np.all(diffs >= -1e-6), "Total variance still non-monotone after fix"

# Put-Call Parity

class TestPCP:

    def _make_pcp_df(self, gap: float = 0.0) -> pd.DataFrame:
        rows = []
        for K in [90., 100., 110.]:
            rows.append({"T": 1.0, "strike": K, "option_type": "call", "iv": 0.20})
            rows.append({"T": 1.0, "strike": K, "option_type": "put",  "iv": 0.20 + gap})
        return pd.DataFrame(rows)

    def test_no_violation_when_matching(self):
        df = self._make_pcp_df(gap=0.0)
        df_clean, report = check_put_call_parity(df, S0=100., r=0.0, q=0.0)
        assert report["n_violations"] == 0

    def test_detects_large_gap(self):
        df = self._make_pcp_df(gap=0.05)  # 5% IV gap — above default 2% threshold
        df_clean, report = check_put_call_parity(
            df, S0=100., r=0.0, q=0.0, iv_gap_threshold=0.02
        )
        assert report["n_violations"] > 0

    def test_removes_violating_rows(self):
        df = self._make_pcp_df(gap=0.05)
        n_before = len(df)
        df_clean, report = check_put_call_parity(
            df, S0=100., r=0.0, q=0.0, iv_gap_threshold=0.02
        )
        assert len(df_clean) < n_before

# filter_surface combined

class TestFilterSurface:

    def test_clean_surface_unchanged(self):
        surf = _flat_surface()
        df_empty = pd.DataFrame()
        _, surf_out, report = filter_surface(
            df_empty, surf.copy(), MATURITY_GRID, MONEYNESS_GRID,
            run_pcp=False,
        )
        assert report["is_clean"] or report["static_violations_fixed"] == 0
        assert surf_out.shape == surf.shape

    def test_calendar_violation_fixed(self):
        surf = _flat_surface()
        surf[1, 10] *= 0.3
        df_empty = pd.DataFrame()
        _, surf_out, report = filter_surface(
            df_empty, surf.copy(), MATURITY_GRID, MONEYNESS_GRID,
            run_static=False, run_pcp=False,
        )
        cal_after = check_calendar_arbitrage(surf_out, MATURITY_GRID)
        assert cal_after["is_arbitrage_free"]

    def test_returns_three_values(self):
        surf = _flat_surface()
        result = filter_surface(pd.DataFrame(), surf, MATURITY_GRID, MONEYNESS_GRID)
        assert len(result) == 3

# surface_diagnostics

class TestDiagnostics:

    def test_diagnostics_returns_dict(self):
        surf = _flat_surface()
        diag = surface_diagnostics(surf, MATURITY_GRID, MONEYNESS_GRID)
        assert isinstance(diag, dict)
        for key in ["coverage", "min_iv", "max_iv", "mean_iv", "atm_ivs"]:
            assert key in diag, f"Missing key: {key}"

    def test_full_surface_coverage(self):
        surf = _flat_surface()
        diag = surface_diagnostics(surf, MATURITY_GRID, MONEYNESS_GRID)
        assert diag["coverage"] == 1.0

    def test_partial_nan_coverage(self):
        surf = _flat_surface()
        surf[0, :5] = np.nan
        diag = surface_diagnostics(surf, MATURITY_GRID, MONEYNESS_GRID)
        assert diag["coverage"] < 1.0

# remove_static_arbitrage_iv

class TestStaticIVRepair:

    def test_repair_does_not_increase_violations(self):
        """After repair, butterfly violations should not increase"""
        S0      = 100.0
        T       = 1.0
        ivs_raw = np.array([0.30, 0.25, 0.15, 0.22, 0.28])  # concave kink at idx 3
        ivs_fixed = remove_static_arbitrage_iv(ivs_raw, MONEYNESS_GRID[:5], T, S0)

        prices_raw   = _bs_call_prices(S0, MONEYNESS_GRID[:5] * S0, T, ivs_raw)
        prices_fixed = _bs_call_prices(S0, MONEYNESS_GRID[:5] * S0, T, ivs_fixed)

        n_raw   = check_butterfly(prices_raw,   MONEYNESS_GRID[:5] * S0)["n_violations"]
        n_fixed = check_butterfly(prices_fixed, MONEYNESS_GRID[:5] * S0)["n_violations"]
        assert n_fixed <= n_raw, (
            f"Violations increased after repair: {n_raw} → {n_fixed}"
        )

# Run

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
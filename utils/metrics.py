"""
Utility functions for model evaluation and arbitrage checking
"""

import numpy as np

def compute_metrics(vol_predicted, vol_true):
    """
    Compute prediction error metrics for implied volatility surfaces.
    
    Parameters
    ----------
    vol_predicted : ndarray, shape (n_maturities, n_strikes)
        Predicted implied volatilities
    vol_true : ndarray, shape (n_maturities, n_strikes)
        True (reference) implied volatilities
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - max_error: Maximum absolute error
        - mape: Mean Absolute Percentage Error
    """
    diff = vol_predicted - vol_true
    
    metrics = {
        'mse': np.mean(diff ** 2),
        'mae': np.mean(np.abs(diff)),
        'rmse': np.sqrt(np.mean(diff ** 2)),
        'max_error': np.max(np.abs(diff)),
        'mape': np.mean(np.abs(diff / (vol_true + 1e-10))) * 100
    }
    
    return metrics


def check_arbitrage(vol_surface, strikes_norm, maturities, S0=100):
    """
    Check for potential arbitrage violations in volatility surface.
    
    Performs simplified heuristic checks for:
    1. Butterfly arbitrage: Excessive concavity in strike direction
    2. Calendar arbitrage: Decreasing call prices with maturity
    
    Parameters
    ----------
    vol_surface : ndarray, shape (n_maturities, n_strikes)
        Implied volatility surface
    strikes_norm : ndarray, shape (n_strikes,)
        Normalized strikes (K/S0)
    maturities : ndarray, shape (n_maturities,)
        Option maturities in years
    S0 : float, default=100
        Spot price
        
    Returns
    -------
    violations : dict
        Dictionary containing counts of detected violations:
        - butterfly: Violations in strike direction
        - calendar: Violations in maturity direction
        - total: Total violations
        
    Notes
    -----
    These are approximate heuristic checks and may not capture all arbitrage
    opportunities. For rigorous no-arbitrage verification, consult:
    - Gatheral & Jacquier (2014) for SVI
    - Fengler (2009) for general arbitrage-free conditions
    """
    violations = {
        'butterfly': 0,
        'calendar': 0,
        'total': 0
    }
    
    # Butterfly spread: check convexity in strike direction
    # Requires that call prices are convex in strike
    # Simplified: check that d²σ/dK² is not too negative
    for i, T in enumerate(maturities):
        vols = vol_surface[i, :]
        if len(vols) >= 3:
            # Second derivative via finite differences
            d2_vol = np.diff(vols, n=2)
            # Flag strong negative curvature
            violations['butterfly'] += np.sum(d2_vol < -0.05)
    
    # Calendar spread: check monotonicity in maturity
    # Call prices should increase with maturity (for same strike)
    # Simplified: volatilities should not decrease too much
    for j in range(len(strikes_norm)):
        vols_T = vol_surface[:, j]
        # First derivative in maturity
        d_vol = np.diff(vols_T)
        # Flag strong decreases
        violations['calendar'] += np.sum(d_vol < -0.05)
    
    violations['total'] = violations['butterfly'] + violations['calendar']
    
    return violations


def compute_smile_metrics(vol_surface, strikes_norm, maturities):
    """
    Compute metrics characterizing the volatility smile shape.
    
    Parameters
    ----------
    vol_surface : ndarray, shape (n_maturities, n_strikes)
        Implied volatility surface
    strikes_norm : ndarray, shape (n_strikes,)
        Normalized strikes
    maturities : ndarray, shape (n_maturities,)
        Maturities in years
        
    Returns
    -------
    smile_metrics : dict
        Dictionary containing per-maturity metrics:
        - atm_vol: At-the-money volatility
        - skew: Left-right asymmetry
        - convexity: Smile curvature
    """
    smile_metrics = {}
    
    for i, T in enumerate(maturities):
        vols = vol_surface[i, :]
        
        # ATM volatility (center strike)
        atm_idx = len(strikes_norm) // 2
        atm_vol = vols[atm_idx]
        
        # Skew: difference between OTM put and call vols
        left_vol = vols[atm_idx // 2]
        right_vol = vols[-(atm_idx // 2)]
        skew = left_vol - right_vol
        
        # Convexity: second derivative at ATM
        if len(vols) >= 3:
            convexity = vols[atm_idx - 1] + vols[atm_idx + 1] - 2 * vols[atm_idx]
        else:
            convexity = 0.0
        
        smile_metrics[T] = {
            'atm_vol': atm_vol,
            'skew': skew,
            'convexity': convexity
        }
    
    return smile_metrics


def relative_error_by_moneyness(vol_predicted, vol_true, strikes_norm):
    """
    Compute relative errors grouped by moneyness (ITM/ATM/OTM).
    
    Parameters
    ----------
    vol_predicted : ndarray, shape (n_maturities, n_strikes)
        Predicted volatilities
    vol_true : ndarray, shape (n_maturities, n_strikes)
        True volatilities
    strikes_norm : ndarray, shape (n_strikes,)
        Normalized strikes
        
    Returns
    -------
    errors_by_region : dict
        Relative errors for ITM, ATM, OTM regions
    """
    n_strikes = len(strikes_norm)
    atm_idx = n_strikes // 2
    
    # Define regions
    itm_slice = slice(0, atm_idx - 1)
    atm_slice = slice(atm_idx - 1, atm_idx + 2)
    otm_slice = slice(atm_idx + 2, None)
    
    diff = np.abs(vol_predicted - vol_true)
    
    errors_by_region = {
        'itm': np.mean(diff[:, itm_slice]),
        'atm': np.mean(diff[:, atm_slice]),
        'otm': np.mean(diff[:, otm_slice])
    }
    
    return errors_by_region
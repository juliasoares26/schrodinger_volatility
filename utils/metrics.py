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


def moment_matching_error(estimated_moments, true_moments):
    """
    Compute errors in moment matching for distribution comparison.
    
    Parameters
    ----------
    estimated_moments : dict
        Dictionary with keys: 'mean', 'std', 'skew', 'kurt'
    true_moments : dict
        Dictionary with same keys as estimated_moments
        
    Returns
    -------
    errors : dict
        Dictionary with absolute errors for each moment
    """
    errors = {}
    for key in ['mean', 'std', 'skew', 'kurt']:
        if key in estimated_moments and key in true_moments:
            errors[f'{key}_error'] = abs(estimated_moments[key] - true_moments[key])
        else:
            errors[f'{key}_error'] = np.nan
    
    return errors


def distribution_distance(samples1, weights1, samples2, weights2, metric='wasserstein'):
    """
    Compute distance between two weighted empirical distributions.
    
    Parameters
    ----------
    samples1 : ndarray, shape (n1,)
        First set of samples
    weights1 : ndarray, shape (n1,)
        Weights for first samples
    samples2 : ndarray, shape (n2,)
        Second set of samples
    weights2 : ndarray, shape (n2,)
        Weights for second samples
    metric : str, default='wasserstein'
        Distance metric: 'wasserstein', 'kl', 'total_variation'
        
    Returns
    -------
    distance : float
        Distance between distributions
    """
    if metric == 'wasserstein':
        # Wasserstein-1 distance (Earth Mover's Distance)
        return wasserstein_distance(samples1, samples2, weights1, weights2)
    
    elif metric == 'kl':
        # KL divergence (requires discretization)
        return kl_divergence(samples1, weights1, samples2, weights2)
    
    elif metric == 'total_variation':
        # Total variation distance
        return total_variation_distance(samples1, weights1, samples2, weights2)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def wasserstein_distance(samples1, samples2, weights1=None, weights2=None):
    """
    Compute Wasserstein-1 (Earth Mover's) distance.
    
    Parameters
    ----------
    samples1, samples2 : ndarray
        Sample arrays
    weights1, weights2 : ndarray, optional
        Sample weights (default: uniform)
        
    Returns
    -------
    distance : float
        Wasserstein distance
    """
    if weights1 is None:
        weights1 = np.ones(len(samples1)) / len(samples1)
    if weights2 is None:
        weights2 = np.ones(len(samples2)) / len(samples2)
    
    # Sort samples
    idx1 = np.argsort(samples1)
    idx2 = np.argsort(samples2)
    
    sorted_samples1 = samples1[idx1]
    sorted_samples2 = samples2[idx2]
    sorted_weights1 = weights1[idx1]
    sorted_weights2 = weights2[idx2]
    
    # Compute cumulative distributions
    cum_weights1 = np.cumsum(sorted_weights1)
    cum_weights2 = np.cumsum(sorted_weights2)
    
    # Compute Wasserstein distance
    all_samples = np.sort(np.concatenate([sorted_samples1, sorted_samples2]))
    distance = 0.0
    
    for i in range(len(all_samples) - 1):
        x1, x2 = all_samples[i], all_samples[i + 1]
        
        # Find cumulative weights at this point
        F1 = np.searchsorted(sorted_samples1, (x1 + x2) / 2, side='right')
        F2 = np.searchsorted(sorted_samples2, (x1 + x2) / 2, side='right')
        
        w1 = cum_weights1[F1 - 1] if F1 > 0 else 0
        w2 = cum_weights2[F2 - 1] if F2 > 0 else 0
        
        distance += abs(w1 - w2) * (x2 - x1)
    
    return distance


def kl_divergence(samples1, weights1, samples2, weights2, n_bins=50):
    """
    Compute KL divergence between two distributions using histograms.
    
    Parameters
    ----------
    samples1, samples2 : ndarray
        Sample arrays
    weights1, weights2 : ndarray
        Sample weights
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    kl : float
        KL divergence (in nats)
    """
    # Create common bins
    all_samples = np.concatenate([samples1, samples2])
    bins = np.linspace(all_samples.min(), all_samples.max(), n_bins + 1)
    
    # Compute histograms
    hist1, _ = np.histogram(samples1, bins=bins, weights=weights1, density=True)
    hist2, _ = np.histogram(samples2, bins=bins, weights=weights2, density=True)
    
    # Normalize
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    
    # Add small epsilon to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Compute KL divergence
    kl = np.sum(hist1 * np.log(hist1 / hist2))
    
    return kl


def total_variation_distance(samples1, weights1, samples2, weights2, n_bins=50):
    """
    Compute total variation distance between distributions.
    
    Parameters
    ----------
    samples1, samples2 : ndarray
        Sample arrays
    weights1, weights2 : ndarray
        Sample weights
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    tv : float
        Total variation distance
    """
    # Create common bins
    all_samples = np.concatenate([samples1, samples2])
    bins = np.linspace(all_samples.min(), all_samples.max(), n_bins + 1)
    
    # Compute histograms
    hist1, _ = np.histogram(samples1, bins=bins, weights=weights1, density=True)
    hist2, _ = np.histogram(samples2, bins=bins, weights=weights2, density=True)
    
    # Normalize
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    
    # Total variation distance
    tv = 0.5 * np.sum(np.abs(hist1 - hist2))
    
    return tv
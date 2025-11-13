"""
Baseline Method 1: Gaussian Process Interpolation
Classical approach for volatility surface interpolation
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import matplotlib.pyplot as plt
from scipy.stats import norm

class GaussianProcessVolatility:
    """
    Volatility surface interpolation using Gaussian Processes.
    
    This baseline method provides fast, probabilistic interpolation with uncertainty
    quantification. However, it does not enforce arbitrage-free constraints.
    
    Advantages:
        - Fast to implement and train
        - Provides uncertainty quantification via posterior variance
        - Reasonable baseline for comparison
    
    Disadvantages:
        - No arbitrage-free guarantees
        - Computational complexity scales as O(n³) with number of observations
        - May produce arbitrageable surfaces in regions with sparse data
    
    Parameters
    ----------
    kernel_type : str, default='rbf'
        Type of kernel function: 'rbf' for Radial Basis Function or 'matern' for Matérn
    length_scale : float, default=1.0
        Characteristic length scale of the kernel
    """
    
    def __init__(self, kernel_type='rbf', length_scale=1.0):
        if kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=1.5)
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=10
        )
        
        self.is_fitted = False
        
    def fit(self, strikes_norm, maturities, impl_vols):
        """
        Train Gaussian Process on observed volatility data.
        
        Parameters
        ----------
        strikes_norm : ndarray, shape (n_strikes,)
            Normalized strike prices (K/S0)
        maturities : ndarray, shape (n_maturities,)
            Option maturities in years
        impl_vols : ndarray, shape (n_maturities, n_strikes)
            Observed implied volatilities
        """
        X_obs = []
        y_obs = []
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes_norm):
                X_obs.append([T, K])
                y_obs.append(impl_vols[i, j])
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        print(f"Fitting GP on {len(X_obs)} observations...")
        self.gp.fit(X_obs, y_obs)
        
        self.is_fitted = True
        print(f"GP fitted successfully. Kernel: {self.gp.kernel_}")
        
    def predict(self, strikes_norm, maturities, return_std=False):
        """
        Predict implied volatilities at new points.
        
        Parameters
        ----------
        strikes_norm : ndarray, shape (n_strikes,)
            Normalized strikes for prediction
        maturities : ndarray, shape (n_maturities,)
            Maturities for prediction
        return_std : bool, default=False
            If True, return posterior standard deviation
            
        Returns
        -------
        vol_surface : ndarray, shape (n_maturities, n_strikes)
            Predicted implied volatilities
        std_surface : ndarray, shape (n_maturities, n_strikes), optional
            Posterior standard deviation (uncertainty)
        """
        if not self.is_fitted:
            raise ValueError("Must fit GP first")
        
        X_pred = []
        for T in maturities:
            for K in strikes_norm:
                X_pred.append([T, K])
        X_pred = np.array(X_pred)
        
        if return_std:
            y_pred, y_std = self.gp.predict(X_pred, return_std=True)
            
            vol_surface = y_pred.reshape(len(maturities), len(strikes_norm))
            std_surface = y_std.reshape(len(maturities), len(strikes_norm))
            
            return vol_surface, std_surface
        else:
            y_pred = self.gp.predict(X_pred)
            vol_surface = y_pred.reshape(len(maturities), len(strikes_norm))
            return vol_surface
    
    def check_arbitrage_violations(self, strikes_norm, maturities, S0=100):
        """
        Check for potential arbitrage violations in the interpolated surface.
        
        This is a simplified heuristic check that looks for:
        1. Butterfly arbitrage: excessive concavity in strike direction
        2. Calendar arbitrage: decreasing call prices with maturity
        
        Note: These are approximate checks and may not capture all arbitrage opportunities.
        
        Parameters
        ----------
        strikes_norm : ndarray
            Normalized strikes
        maturities : ndarray
            Maturities in years
        S0 : float, default=100
            Spot price
            
        Returns
        -------
        violations : dict
            Dictionary containing counts of detected violations by type
        """
        vol_surface = self.predict(strikes_norm, maturities)
        
        violations = {
            'butterfly': 0,
            'calendar': 0,
            'total': 0
        }
        
        # Butterfly spread: check convexity in strike direction
        # Requires C''(K) > 0, approximately checking d²σ/dK² is not too negative
        for i, T in enumerate(maturities):
            vols = vol_surface[i, :]
            if len(vols) >= 3:
                d2_vol = np.diff(vols, n=2)
                violations['butterfly'] += np.sum(d2_vol < -0.05)
        
        # Calendar spread: check monotonicity in maturity
        # Call prices should increase with maturity (simplified check on vols)
        for j in range(len(strikes_norm)):
            vols_T = vol_surface[:, j]
            d_vol = np.diff(vols_T)
            violations['calendar'] += np.sum(d_vol < -0.05)
        
        violations['total'] = violations['butterfly'] + violations['calendar']
        
        return violations


if __name__ == "__main__":
    data = np.load('../data/synthetic/heston_data.npz', allow_pickle=True)
    
    strikes_norm = data['strikes']
    maturities = data['maturities']
    vol_surface_true = data['vol_surface']
    
    print("Data loaded successfully:")
    print(f"  Strikes: {len(strikes_norm)}")
    print(f"  Maturities: {len(maturities)}")
    print(f"  Vol range: {vol_surface_true.min()*100:.1f}% - {vol_surface_true.max()*100:.1f}%")
    
    # Test scenario: sparse observations
    # Observe only 50% of strikes at all maturities
    obs_indices = np.arange(0, len(strikes_norm), 2)
    obs_strikes = strikes_norm[obs_indices]
    obs_vols = vol_surface_true[:, obs_indices]
    
    print(f"\nTest scenario:")
    print(f"  Observations: {len(obs_strikes)} strikes × {len(maturities)} maturities")
    print(f"  Total observed points: {len(obs_strikes) * len(maturities)}")
    print(f"  Interpolation grid: {len(strikes_norm)} strikes (dense)")
    
    # Fit Gaussian Process
    gp_model = GaussianProcessVolatility(kernel_type='rbf', length_scale=0.5)
    gp_model.fit(obs_strikes, maturities, obs_vols)
    
    # Predict on dense grid
    vol_surface_pred, vol_std = gp_model.predict(
        strikes_norm, maturities, return_std=True
    )
    
    print(f"\nPrediction completed successfully")
    
    # Compute interpolation errors
    mse = np.mean((vol_surface_pred - vol_surface_true) ** 2)
    mae = np.mean(np.abs(vol_surface_pred - vol_surface_true))
    max_error = np.max(np.abs(vol_surface_pred - vol_surface_true))
    
    print(f"\nInterpolation errors:")
    print(f"  MSE: {mse*10000:.2f} (bps²)")
    print(f"  MAE: {mae*100:.2f}%")
    print(f"  Max error: {max_error*100:.2f}%")
    
    # Check for arbitrage violations
    violations = gp_model.check_arbitrage_violations(strikes_norm, maturities)
    
    print(f"\nArbitrage violations (simplified heuristic):")
    print(f"  Butterfly: {violations['butterfly']}")
    print(f"  Calendar: {violations['calendar']}")
    print(f"  Total: {violations['total']}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Volatility smiles (true vs predicted)
    for i, T in enumerate(maturities):
        ax = axes[0, i]
        
        ax.plot(strikes_norm, vol_surface_true[i, :] * 100, 
                'o-', label='True', linewidth=2, markersize=6)
        
        ax.plot(strikes_norm, vol_surface_pred[i, :] * 100,
                's--', label='GP', linewidth=2, markersize=4, alpha=0.7)
        
        ax.scatter(obs_strikes, obs_vols[i, :] * 100,
                  marker='x', s=100, color='red', label='Observed', zorder=5)
        
        ax.fill_between(strikes_norm,
                       (vol_surface_pred[i, :] - 2*vol_std[i, :]) * 100,
                       (vol_surface_pred[i, :] + 2*vol_std[i, :]) * 100,
                       alpha=0.2, color='orange', label='±2σ')
        
        ax.set_xlabel('Strike / Spot')
        ax.set_ylabel('Implied Vol (%)')
        ax.set_title(f'T = {T}Y')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Row 2: Error analysis
    errors = np.abs(vol_surface_pred - vol_surface_true) * 100
    
    ax = axes[1, 0]
    for i, T in enumerate(maturities):
        ax.plot(strikes_norm, errors[i, :], 'o-', label=f'T={T}Y')
    ax.set_xlabel('Strike / Spot')
    ax.set_ylabel('Absolute Error (%)')
    ax.set_title('Interpolation Error by Strike')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    im = ax.imshow(errors, aspect='auto', cmap='Reds',
                   extent=[strikes_norm[0], strikes_norm[-1],
                          maturities[0], maturities[-1]],
                   origin='lower')
    ax.set_xlabel('Strike / Spot')
    ax.set_ylabel('Maturity (years)')
    ax.set_title('Error Heatmap')
    plt.colorbar(im, ax=ax, label='Abs Error (%)')
    
    ax = axes[1, 2]
    im = ax.imshow(vol_std * 100, aspect='auto', cmap='Blues',
                   extent=[strikes_norm[0], strikes_norm[-1],
                          maturities[0], maturities[-1]],
                   origin='lower')
    ax.set_xlabel('Strike / Spot')
    ax.set_ylabel('Maturity (years)')
    ax.set_title('GP Uncertainty (std %)')
    plt.colorbar(im, ax=ax, label='Std (%)')
    
    plt.tight_layout()
    plt.savefig('../data/synthetic/gp_baseline_results.png', dpi=150)
    plt.show()
    
    print("\nResults saved to: data/synthetic/gp_baseline_results.png")
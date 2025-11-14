"""
Baseline Method 2: SVI (Stochastic Volatility Inspired) Parametrization
Industry-standard approach for volatility smile fitting
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

class SVIModel:
    """
    Stochastic Volatility Inspired (SVI) parametrization for volatility smiles.
    
    The SVI model represents the total implied variance as a function of log-moneyness:
    
        w(k) = a + b(ρ(k - m) + √((k - m)² + σ²))
    
    where k = log(K/F) is the log-moneyness (log of strike relative to forward).
    
    Parameters
    ----------
    a : float
        Vertical shift parameter controlling overall variance level
    b : float
        Slope parameter controlling the angle between left and right asymptotes
    ρ : float, range [-1, 1]
        Skew parameter controlling asymmetry of the smile
    m : float
        Horizontal shift parameter locating the smile's center
    σ : float
        Smoothness parameter controlling smile curvature/convexity
        
    Notes
    -----
    The SVI parametrization is widely used in practice due to its:
    - Analytical tractability (closed-form formula)
    - Flexibility in capturing various smile shapes
    - Natural economic interpretation of parameters
    - Ability to satisfy no-arbitrage constraints with appropriate bounds
    
    Key no-arbitrage conditions (simplified):
    1. b|ρ| < 2 (prevents calendar spread arbitrage)
    2. a + bσ√(1-ρ²) ≥ 0 (ensures non-negative variance)
    
    References
    ----------
    Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility 
        parameterization with application to the valuation of volatility derivatives."
    Gatheral, J., & Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces."
    """
    
    def __init__(self):
        self.params_per_maturity = {}
        
    @staticmethod
    def svi_variance(k, a, b, rho, m, sigma):
        """
        Compute total implied variance using SVI formula.
        
        Parameters
        ----------
        k : array_like
            Log-moneyness: k = log(K/F)
        a, b, rho, m, sigma : float
            SVI parameters
            
        Returns
        -------
        variance : ndarray
            Total implied variance w(k)
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    @staticmethod
    def svi_volatility(k, a, b, rho, m, sigma):
        """
        Compute implied volatility from SVI variance.
        
        Parameters
        ----------
        k : array_like
            Log-moneyness
        a, b, rho, m, sigma : float
            SVI parameters
            
        Returns
        -------
        volatility : ndarray
            Implied volatility σ_impl = √(w(k)/T)
        """
        var = SVIModel.svi_variance(k, a, b, rho, m, sigma)
        return np.sqrt(np.maximum(var, 1e-8))
    
    def fit_maturity(self, strikes_norm, impl_vols, T, S0=100, initial_guess=None):
        """
        Calibrate SVI parameters for a single maturity slice.
        
        Uses nonlinear least squares optimization with no-arbitrage constraints.
        Falls back to differential evolution (global optimization) if local 
        optimization fails.
        
        Parameters
        ----------
        strikes_norm : ndarray, shape (n_strikes,)
            Normalized strikes (K/S0)
        impl_vols : ndarray, shape (n_strikes,)
            Observed implied volatilities
        T : float
            Time to maturity in years
        S0 : float, default=100
            Spot price
        initial_guess : array_like, optional
            Initial parameter guess [a, b, rho, m, sigma]
            
        Returns
        -------
        params : ndarray, shape (5,)
            Optimized SVI parameters [a, b, rho, m, sigma]
        success : bool
            True if optimization converged successfully
        """
        k = np.log(strikes_norm)
        
        # Total variance (market)
        w_market = impl_vols**2 * T
        
        if initial_guess is None:
            atm_vol = impl_vols[len(impl_vols)//2]
            initial_guess = [
                atm_vol**2 * T * 0.5,
                0.1,
                -0.3,
                0.0,
                0.2
            ]
        
        # Bounds ensuring parameter validity
        bounds = [
            (1e-4, None),
            (1e-4, 1.0),
            (-0.999, 0.999),
            (-1.0, 1.0),
            (1e-4, 2.0)
        ]
        
        def objective(params):
            a, b, rho, m, sigma = params
            w_svi = self.svi_variance(k, a, b, rho, m, sigma)
            return np.mean((w_svi - w_market)**2)
        
        # No-arbitrage constraints
        def constraint1(params):
            """Ensure b|ρ| < 2"""
            a, b, rho, m, sigma = params
            return 2 - b * np.abs(rho)
        
        def constraint2(params):
            """Ensure minimum variance is non-negative"""
            a, b, rho, m, sigma = params
            return a + b * sigma * np.sqrt(1 - rho**2)
        
        constraints = [
            {'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2}
        ]
        
        result = minimize(
            objective,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Warning: Local optimization failed for T={T}: {result.message}")
            print("Attempting global optimization...")
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=500,
                seed=42
            )
        
        params = result.x
        final_error = result.fun
        
        print(f"  T={T:.2f}Y: MSE={final_error*1e6:.2f} (×10⁻⁶), "
              f"a={params[0]:.4f}, b={params[1]:.4f}, ρ={params[2]:.3f}")
        
        return params, result.success
    
    def fit(self, strikes_norm, maturities, vol_surface, S0=100):
        """
        Calibrate SVI parameters across all maturity slices.
        
        Fits independent SVI curves to each maturity, treating each time slice
        as a separate optimization problem.
        
        Parameters
        ----------
        strikes_norm : ndarray, shape (n_strikes,)
            Normalized strikes (K/S0)
        maturities : ndarray, shape (n_maturities,)
            Option maturities in years
        vol_surface : ndarray, shape (n_maturities, n_strikes)
            Observed implied volatilities
        S0 : float, default=100
            Spot price
        """
        print("Fitting SVI for each maturity slice...")
        
        self.params_per_maturity = {}
        
        for i, T in enumerate(maturities):
            impl_vols = vol_surface[i, :]
            params, success = self.fit_maturity(strikes_norm, impl_vols, T, S0)
            self.params_per_maturity[T] = {
                'params': params,
                'success': success
            }
        
        print("SVI fitting completed successfully")
        
    def predict(self, strikes_norm, maturities):
        """
        Predict implied volatilities using calibrated SVI parameters.
        
        Parameters
        ----------
        strikes_norm : ndarray, shape (n_strikes,)
            Normalized strikes for prediction
        maturities : ndarray, shape (n_maturities,)
            Maturities for prediction
            
        Returns
        -------
        vol_surface : ndarray, shape (n_maturities, n_strikes)
            Predicted implied volatilities
        """
        k = np.log(strikes_norm)
        vol_surface = np.zeros((len(maturities), len(strikes_norm)))
        
        for i, T in enumerate(maturities):
            if T not in self.params_per_maturity:
                raise ValueError(f"Maturity T={T} not fitted")
            
            params = self.params_per_maturity[T]['params']
            a, b, rho, m, sigma = params
            
            vol_surface[i, :] = self.svi_volatility(k, a, b, rho, m, sigma)
        
        return vol_surface
    
    def interpolate_maturity(self, strikes_norm, T_new, T_lower, T_upper):
        """
        Interpolate volatility smile for intermediate maturity.
        
        Uses linear interpolation of SVI parameters. More sophisticated 
        approaches (e.g., SSVI) may preserve calendar arbitrage-free conditions.
        
        Parameters
        ----------
        strikes_norm : ndarray
            Normalized strikes
        T_new : float
            Target maturity (must satisfy T_lower < T_new < T_upper)
        T_lower, T_upper : float
            Bracketing maturities with fitted parameters
            
        Returns
        -------
        impl_vols : ndarray
            Interpolated implied volatilities
        """
        if T_lower not in self.params_per_maturity or T_upper not in self.params_per_maturity:
            raise ValueError("Bracketing maturities must be fitted first")
        
        w = (T_new - T_lower) / (T_upper - T_lower)
        
        params_lower = self.params_per_maturity[T_lower]['params']
        params_upper = self.params_per_maturity[T_upper]['params']
        
        params_new = (1 - w) * params_lower + w * params_upper
        
        k = np.log(strikes_norm)
        a, b, rho, m, sigma = params_new
        
        return self.svi_volatility(k, a, b, rho, m, sigma)


if __name__ == "__main__":
    data = np.load('../data/synthetic/heston_data.npz', allow_pickle=True)
    
    strikes_norm = data['strikes']
    maturities = data['maturities']
    vol_surface_true = data['vol_surface']
    
    print("Testing SVI Parametrization")
    print(f"  Data: {len(strikes_norm)} strikes × {len(maturities)} maturities")
    
    # Calibrate SVI
    svi = SVIModel()
    svi.fit(strikes_norm, maturities, vol_surface_true)
    
    # Predict using fitted parameters
    vol_surface_svi = svi.predict(strikes_norm, maturities)
    
    # Compute fitting errors
    mse = np.mean((vol_surface_svi - vol_surface_true) ** 2)
    mae = np.mean(np.abs(vol_surface_svi - vol_surface_true))
    
    print(f"\nFitting errors:")
    print(f"  MSE: {mse*10000:.2f} (bps²)")
    print(f"  MAE: {mae*100:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, T in enumerate(maturities):
        ax = axes[i]
        
        ax.plot(strikes_norm, vol_surface_true[i, :] * 100,
                'o-', label='True (Heston)', linewidth=2, markersize=6)
        ax.plot(strikes_norm, vol_surface_svi[i, :] * 100,
                's--', label='SVI Fit', linewidth=2, markersize=4, alpha=0.7)
        
        params = svi.params_per_maturity[T]['params']
        param_str = f"a={params[0]:.3f}, b={params[1]:.3f}\nρ={params[2]:.2f}, m={params[3]:.2f}, σ={params[4]:.2f}"
        ax.text(0.05, 0.95, param_str, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        ax.set_xlabel('Strike / Spot')
        ax.set_ylabel('Implied Vol (%)')
        ax.set_title(f'T = {T}Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/synthetic/svi_baseline_results.png', dpi=150)
    plt.show()
    
    print("\nResults saved to: data/synthetic/svi_baseline_results.png")

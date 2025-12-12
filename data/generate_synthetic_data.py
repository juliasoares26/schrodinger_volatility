"""
Generate synthetic volatility surface data using Heston model.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append('..')

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.heston import HestonModel
import matplotlib.pyplot as plt


def black_scholes_iv(S, K, T, r, price, option_type='call'):
    """
    Compute implied volatility using Newton-Raphson.
    
    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    price : float
        Option price
    option_type : str
        'call' or 'put'
        
    Returns
    -------
    iv : float
        Implied volatility
    """
    from scipy.stats import norm
    
    # Initial guess
    sigma = 0.3
    
    for _ in range(100):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price_est = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            vega = S*norm.pdf(d1)*np.sqrt(T)
        else:
            price_est = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            vega = S*norm.pdf(d1)*np.sqrt(T)
        
        diff = price - price_est
        
        if abs(diff) < 1e-6:
            break
        
        if vega < 1e-10:
            break
        
        sigma += diff / vega
        sigma = max(0.01, min(2.0, sigma))  # Bounds
    
    return sigma


def generate_heston_surface(heston_params, strikes_norm, maturities, n_paths=10000, r=0.0):
    """
    Generate implied volatility surface using Monte Carlo.
    
    Parameters
    ----------
    heston_params : dict
        Heston model parameters
    strikes_norm : ndarray
        Normalized strikes (K/S0)
    maturities : ndarray
        Maturities in years
    n_paths : int
        Number of Monte Carlo paths
    r : float
        Risk-free rate
        
    Returns
    -------
    vol_surface : ndarray, shape (n_maturities, n_strikes)
        Implied volatility surface
    """
    S0 = heston_params['S0']
    
    vol_surface = np.zeros((len(maturities), len(strikes_norm)))
    
    print(f"\nGenerating volatility surface...")
    print(f"  Maturities: {maturities}")
    print(f"  Strikes: {len(strikes_norm)} points")
    print(f"  Paths: {n_paths:,}")
    
    heston = HestonModel(**heston_params)
    
    for i, T in enumerate(maturities):
        print(f"\n  Processing T={T:.2f}Y...")
        
        n_steps = max(50, int(T * 252))
        
        # Generate all paths at once for this maturity
        print(f"    Simulating {n_paths:,} paths with {n_steps} steps...")
        
        try:
            # Generate all paths at once
            # IMPORTANT: Heston.simulate_paths returns (S, v, t) NOT (t, S, v)!
            S_paths, v_paths, times = heston.simulate_paths(T, n_steps, n_paths=n_paths)
            
            # Debug first path
            print(f"    S_paths shape: {S_paths.shape}")
            print(f"    First path first 5 values: {S_paths[0, :5]}")
            print(f"    First path last value: {S_paths[0, -1]:.2f}")
            
            # Extract final prices
            # S_paths has shape (n_paths, n_steps+1)
            final_prices = S_paths[:, -1]
            
        except MemoryError as e:
            print(f"    Memory error: {e}")
            print(f"    Falling back to one-by-one simulation...")
            
            # Fall back to one-by-one simulation
            final_prices = []
            
            for path_idx in range(n_paths):
                if (path_idx + 1) % 1000 == 0:
                    print(f"      Progress: {path_idx + 1}/{n_paths}")
                
                # IMPORTANT: Returns (S, v, t) NOT (t, S, v)
                S_path, v_path, times = heston.simulate_paths(T, n_steps, n_paths=1)
                final_prices.append(S_path[0, -1])
            
            final_prices = np.array(final_prices)
        
        print(f"    ✓ Simulation complete")
        print(f"    Final prices: min={final_prices.min():.2f}, max={final_prices.max():.2f}, mean={final_prices.mean():.2f}")
        print(f"    Expected around S0={S0:.2f}")
        
        # Sanity check
        if np.abs(final_prices.mean() - S0) > S0 * 0.5:
            print(f"    ⚠ WARNING: Prices seem wrong!")
            print(f"    Mean price={final_prices.mean():.2f} vs S0={S0:.2f}")
        
        # Compute option prices for each strike
        print(f"    Computing implied volatilities...")
        for j, K_norm in enumerate(strikes_norm):
            K = K_norm * S0
            
            # Call payoff
            payoffs = np.maximum(final_prices - K, 0)
            price = np.exp(-r*T) * np.mean(payoffs)
            
            # Compute implied vol
            try:
                # Check if price is reasonable
                if price < 1e-10:
                    # Deep OTM or pricing issue
                    if j > 0:
                        vol_surface[i, j] = vol_surface[i, j-1]
                    else:
                        vol_surface[i, j] = 0.2
                else:
                    iv = black_scholes_iv(S0, K, T, r, price, 'call')
                    vol_surface[i, j] = iv
                    
                    if (j+1) % 5 == 0:
                        print(f"      Strike {j+1}/{len(strikes_norm)}: K/S0={K_norm:.3f}, IV={vol_surface[i,j]*100:.2f}%")
            except Exception as e:
                # If IV computation fails, use neighbor or default
                if j > 0:
                    vol_surface[i, j] = vol_surface[i, j-1]
                else:
                    vol_surface[i, j] = 0.2
    
    return vol_surface


def main():
    """Generate and save synthetic data."""
    print("="*60)
    print("SYNTHETIC DATA GENERATION")
    print("="*60)
    
    # Create directories
    data_dir = Path('synthetic')
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Heston parameters (realistic)
    heston_params = {
        'kappa': 2.0,      # Mean reversion speed
        'theta': 0.04,     # Long-term variance (20% vol)
        'sigma': 0.3,      # Vol of vol
        'rho': -0.7,       # Negative correlation (leverage effect)
        'v0': 0.04,        # Initial variance
        'S0': 100.0        # Spot price
    }
    
    print("\nHeston Parameters:")
    for key, value in heston_params.items():
        print(f"  {key}: {value}")
    
    # Grid specification
    strikes_norm = np.linspace(0.8, 1.2, 21)  # 80% to 120% of spot
    maturities = np.array([0.25, 0.5, 0.75, 1.0])    # 3M, 6M, 9M, 1Y
    
    print(f"\nGrid Specification:")
    print(f"  Strikes: {len(strikes_norm)} points from {strikes_norm[0]} to {strikes_norm[-1]}")
    print(f"  Maturities: {maturities}")
    
    # Generate surface
    vol_surface = generate_heston_surface(
        heston_params,
        strikes_norm,
        maturities,
        n_paths=5000  # Adjust based on your memory
    )
    
    print("\n" + "="*60)
    print("Surface Statistics:")
    print("="*60)
    print(f"  Min vol: {vol_surface.min()*100:.2f}%")
    print(f"  Max vol: {vol_surface.max()*100:.2f}%")
    print(f"  Mean vol: {vol_surface.mean()*100:.2f}%")
    print(f"  ATM vol (1Y): {vol_surface[-1, 10]*100:.2f}%")
    
    # Save data
    output_file = data_dir / 'heston_data.npz'
    np.savez(
        output_file,
        strikes=strikes_norm,
        maturities=maturities,
        vol_surface=vol_surface,
        heston_params=heston_params
    )
    
    print(f"\n✓ Data saved to: {output_file}")
    
    # Visualization
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Volatility smiles
    ax = axes[0]
    for i, T in enumerate(maturities):
        ax.plot(strikes_norm, vol_surface[i, :] * 100, 
                'o-', label=f'T={T}Y', linewidth=2, markersize=6)
    
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax.set_xlabel('Strike / Spot', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title('Volatility Smiles', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 3D surface using pcolormesh
    ax = axes[1]
    
    # Create meshgrid
    K_mesh, T_mesh = np.meshgrid(strikes_norm, maturities)
    
    if vol_surface.min() == vol_surface.max():
        # All values are the same
        print("  Warning: All volatilities are identical, skipping surface plot")
        ax.text(0.5, 0.5, 'All volatilities identical\n(Check Heston model output)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        # Use pcolormesh for better visualization
        pcm = ax.pcolormesh(K_mesh, T_mesh, vol_surface * 100, 
                           cmap='viridis', shading='auto')
        
        # Add contour lines on top
        levels = np.linspace(vol_surface.min() * 100, vol_surface.max() * 100, 10)
        ax.contour(K_mesh, T_mesh, vol_surface * 100, 
                  levels=levels, colors='white', alpha=0.5, linewidths=1)
        
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Implied Vol (%)', fontsize=10)
    
    ax.set_xlabel('Strike / Spot', fontsize=12)
    ax.set_ylabel('Maturity (years)', fontsize=12)
    ax.set_title('Volatility Surface', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plot_file = data_dir / 'heston_surface.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_file}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {output_file}")
    print(f"  - {plot_file}")
    print(f"\nYou can now run:")
    print(f"  python ../baselines/gaussian_process.py")
    print(f"  python ../experiments/quick_test.py")


if __name__ == "__main__":
    main()
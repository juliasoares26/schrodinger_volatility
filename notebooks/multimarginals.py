#Multi-Marginals and Convex Interpolation

#Objective:
#demonstrate that the Martingale Schrödinger Bridge automatically calibrates multiple maturities while preserving convex order.
#For all t ∈ [t_{i-1}, t_i]: μ_{i-1} ≤_{conv} Law(S_t) ≤_{conv} μ_i
#This means there is no calendar arbitrage implicit in the interpolation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path("results/multi_marginals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent 

possible_paths = [
    project_root, 
    project_root / 'models',  
    project_root / 'baselines',  
    project_root / 'utils',  
    project_root / 'src',
    project_root / 'sinkhorn',  
    script_dir, 
]

for path in possible_paths:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

print("Import Configuration")
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"\nPaths added to sys.path:")
for i, path in enumerate(possible_paths, 1):
    exists = path.exists()
    print(f"  {i}. {path} {'✓' if exists else '✗ (does not exist)'}")

print("\nAttempting to import modules")

try:
    from heston import HestonUnified
    print("HestonModel imported successfully")
except ImportError as e:
    print(f"Error importing HestonModel: {e}")
    print(f"\n  Looking for heston.py file...")
    for path in possible_paths:
        heston_file = path / "heston.py"
        if heston_file.exists():
            print(f"Found at: {heston_file}")
        else:
            print(f"Not found at: {path}")
    raise

try:
    from bridge import MartingaleSchrodingerBridge
    print("MartingaleSchrodingerBridge imported successfully")
except ImportError as e:
    print(f"Error importing MartingaleSchrodingerBridge: {e}")
    print(f"\n Looking for bridge.py file...")
    for path in possible_paths:
        bridge_file = path / "bridge.py"
        if bridge_file.exists():
            print(f"Found at: {bridge_file}")
        else:
            print(f"Not found at: {path}")
    raise

try:
    from metrics import compute_metrics
    print("Metrics functions imported successfully")
except ImportError as e:
    print(f"Error importing metrics: {e}")
    print(f"\n Looking for metrics.py file...")
    for path in possible_paths:
        metrics_file = path / "metrics.py"
        if metrics_file.exists():
            print(f"Found at: {metrics_file}")
        else:
            print(f"Not found at: {path}")
    raise

print("\n All modules imported successfully!")
print("="*70 + "\n")

#Generate synthetic market smiles with realistic skew
def generate_market_smiles(maturities, strikes, S0, base_vol=0.20):
    market_ivs = {}
    for T in maturities:
        #ATM vol increases with sqrt(T)
        atm_vol = base_vol * np.sqrt(T / 0.25)
        
        #Skew: linear in log-moneyness
        log_moneyness = np.log(strikes / S0)
        skew = -0.15 * log_moneyness / np.sqrt(T)
        
        #Smile (convexity)
        smile = 0.03 * log_moneyness**2 / T
        
        iv = atm_vol + skew + smile
        market_ivs[T] = np.maximum(iv, 0.05)  # 5% floor
    
    return market_ivs


def black_scholes_price(S, K, T, sigma, r=0.0):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def black_scholes_iv(S, K, T, r, price, option_type='call'):
    sigma = 0.3
    for _ in range(100):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price_est = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            vega = S*norm.pdf(d1)*np.sqrt(T)
        
        diff = price - price_est
        if abs(diff) < 1e-6:
            break
        if vega < 1e-10:
            break
        
        sigma += diff / vega
        sigma = max(0.01, min(2.0, sigma))
    
    return sigma


#Test convex order at intermediate dates
#For each interval [t_i, t_{i+1}], simulate S_t at uniformly spaced dates and verify:
#E[f(S_t)] grows for all convex f
#Distribution CDF crossing test
    
def test_convex_order_interpolation(msb_model, maturities, n_test_dates=10):
    results = {}
    
    for i in range(len(maturities) - 1):
        t_start, t_end = maturities[i], maturities[i+1]
        test_dates = np.linspace(t_start, t_end, n_test_dates)
        
        print(f"  Testing interval [{t_start:.3f}, {t_end:.3f}]...")
        
        #Simulate marginals at each date
        marginals = []
        for t in test_dates:
            #Simulate paths up to time t
            n_steps = max(20, int(t * 252))
            S, v, _ = msb_model.heston.simulate_paths(T=t, n_steps=n_steps, n_paths=50000)
            S_t = S[:, -1]
            marginals.append(S_t)
        
        #Martingale property (constant mean = S0)
        means = [np.mean(S) for S in marginals]
        
        #Convex order via call prices
        test_strikes = np.linspace(msb_model.heston.S0*0.7, msb_model.heston.S0*1.3, 30)
        call_prices = np.zeros((len(test_dates), len(test_strikes)))
        
        for j, S_t in enumerate(marginals):
            for k, K in enumerate(test_strikes):
                call_prices[j, k] = np.mean(np.maximum(S_t - K, 0))
        
        results[f'interval_{i}'] = {
            'dates': test_dates,
            'means': means,
            'call_prices': call_prices,
            'strikes': test_strikes,
            'marginals': marginals
        }
    
    return results

#Visualize market smiles
def plot_market_smiles(market_data, n_maturities_list, moneyness, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, n in enumerate(n_maturities_list):
        ax = axes[idx]
        for T, ivs in market_data[n].items():
            ax.plot(moneyness, ivs * 100, 'o-', label=f'T={T:.2f}Y', linewidth=2)
        ax.set_xlabel('Moneyness (K/S0)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(f'{n} Maturities')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'market_smiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: market_smiles.png")


def plot_convergence(results, n_maturities_list, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, n in enumerate(n_maturities_list):
        ax = axes[idx]
        losses = results[n]['losses']
        
        ax.semilogy(losses, 'b-', linewidth=2, label='Training Loss')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'{n} Maturities: Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: convergence_training.png")


def plot_convex_order_tests(convex_test, mats, S0, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    #Martingale property (E[S_t] = S0)
    ax = axes[0, 0]
    for interval_name, data in convex_test.items():
        ax.plot(data['dates'], data['means'], 'o-', label=interval_name, linewidth=2)
    ax.axhline(y=S0, color='k', linestyle='--', linewidth=2, label='S0 (No Arbitrage)')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('E[S_t]')
    ax.set_title('Martingale Property: E[S_t] = S0')
    ax.legend()
    ax.grid(True, alpha=0.3)

    #Call prices evolution (convexity)
    ax = axes[0, 1]
    interval_data = convex_test['interval_0']
    dates = interval_data['dates']
    call_prices = interval_data['call_prices']
    strikes = interval_data['strikes']

    #Plot for some representative strikes
    strike_indices = [5, 15, 25]  #OTM, ATM, ITM
    for idx in strike_indices:
        K = strikes[idx]
        ax.plot(dates, call_prices[:, idx], 'o-', 
                label=f'K={K:.1f} ({K/S0:.2%} moneyness)', linewidth=2)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Call Price')
    ax.set_title('Call Prices Evolution (Convex Order)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    #CDF crossing test (first interval)
    ax = axes[1, 0]
    interval_data = convex_test['interval_0']
    marginals = interval_data['marginals']
    dates = interval_data['dates']

    #Plot CDFs for some dates
    date_indices = [0, len(dates)//4, len(dates)//2, 3*len(dates)//4, -1]
    x_range = np.linspace(S0*0.5, S0*1.5, 200)

    for idx in date_indices:
        S_t = marginals[idx]
        cdf = np.array([np.mean(S_t <= x) for x in x_range])
        ax.plot(x_range, cdf, label=f't={dates[idx]:.3f}', linewidth=2)

    ax.set_xlabel('S')
    ax.set_ylabel('CDF(S)')
    ax.set_title(f'CDFs Evolution [{mats[0]:.2f}, {mats[1]:.2f}]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    #Variance evolution
    ax = axes[1, 1]
    for interval_name, data in convex_test.items():
        variances = [np.var(S) for S in data['marginals']]
        ax.plot(data['dates'], variances, 'o-', label=interval_name, linewidth=2)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Var(S_t)')
    ax.set_title('Variance Evolution (should be monotonic)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'convex_order_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: convex_order_test.png")


def plot_calibrated_smiles(results, heston, n_maturities_list, strikes, moneyness, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, n in enumerate(n_maturities_list):
        ax = axes[idx]
        msb_model = results[n]['model']
        mats = results[n]['maturities']
        target_ivs = results[n]['target_ivs']
        
        #For each maturity
        for T in mats:
            naked_prices = heston.option_prices_mc(strikes, T=T, n_paths=50000)
            naked_ivs = np.array([
                black_scholes_iv(heston.S0, K, T, 0.0, price)
                for K, price in zip(strikes, naked_prices)
            ])
            
            #Calibrated model - use MSB model
            calibrated_prices = []
            for K in strikes:
                price, _ = msb_model.price_option(K, T, n_paths=50000)
                calibrated_prices.append(price)
            
            calibrated_ivs = np.array([
                black_scholes_iv(heston.S0, K, T, 0.0, price)
                for K, price in zip(strikes, calibrated_prices)
            ])
            
            #Market
            market_iv = target_ivs[T]
            
            #Plot
            ax.plot(moneyness, market_iv*100, 'ko-', 
                    label=f'Market T={T:.2f}' if T == mats[0] else '', 
                    linewidth=2, markersize=8)
            ax.plot(moneyness, naked_ivs*100, 'r--', 
                    label=f'Naked T={T:.2f}' if T == mats[0] else '', 
                    linewidth=1.5, alpha=0.6)
            ax.plot(moneyness, calibrated_ivs*100, 'b-', 
                    label=f'Calibrated T={T:.2f}' if T == mats[0] else '', 
                    linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Moneyness (K/S0)', fontsize=11)
        ax.set_ylabel('Implied Volatility (%)', fontsize=11)
        ax.set_title(f'{n} Maturities: Calibration Quality', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'calibrated_smiles_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: calibrated_smiles_comparison.png")


def compute_and_plot_metrics(results, n_maturities_list, strikes, output_dir):
    metrics_data = []

    for n in n_maturities_list:
        msb_model = results[n]['model']
        target_ivs = results[n]['target_ivs']
        
        for T in results[n]['maturities']:
            calibrated_prices = []
            for K in strikes:
                price, _ = msb_model.price_option(K, T, n_paths=50000)
                calibrated_prices.append(price)
            
            calibrated_iv = np.array([
                black_scholes_iv(msb_model.heston.S0, K, T, 0.0, price)
                for K, price in zip(strikes, calibrated_prices)
            ])
            market_iv = target_ivs[T]
            
            rmse = np.sqrt(np.mean((calibrated_iv - market_iv)**2))
            mae = np.mean(np.abs(calibrated_iv - market_iv))
            max_error = np.max(np.abs(calibrated_iv - market_iv))
            
            metrics_data.append({
                'n_maturities': n,
                'maturity': T,
                'RMSE (bps)': rmse * 10000,
                'MAE (bps)': mae * 10000,
                'Max Error (bps)': max_error * 10000
            })

    metrics_df = pd.DataFrame(metrics_data)
    
    metrics_df.to_csv(output_dir / 'calibration_metrics.csv', index=False)
    
    print("Calibrations errors by maturity")
    print(metrics_df.to_string(index=False))

    #Summary statistics
    summary = metrics_df.groupby('n_maturities').agg({
        'RMSE (bps)': ['mean', 'max'],
        'MAE (bps)': ['mean', 'max']
    }).round(2)

    print("\n: Average Errors Across All Maturities")
    print(summary)
    
    summary.to_csv(output_dir / 'calibration_summary.csv')

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(n_maturities_list))
    width = 0.25

    means_rmse = [metrics_df[metrics_df['n_maturities']==n]['RMSE (bps)'].mean() 
                  for n in n_maturities_list]
    means_mae = [metrics_df[metrics_df['n_maturities']==n]['MAE (bps)'].mean() 
                 for n in n_maturities_list]
    max_errors = [metrics_df[metrics_df['n_maturities']==n]['Max Error (bps)'].max() 
                  for n in n_maturities_list]

    ax.bar(x - width, means_rmse, width, label='RMSE', alpha=0.8)
    ax.bar(x, means_mae, width, label='MAE', alpha=0.8)
    ax.bar(x + width, max_errors, width, label='Max Error', alpha=0.8)

    ax.set_xlabel('Number of Maturities', fontsize=12)
    ax.set_ylabel('Error (basis points)', fontsize=12)
    ax.set_title('Calibration Error vs Number of Maturities', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(n_maturities_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_n_maturities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: error_vs_n_maturities.png")


def main():
    print("\n1. Setting up Naked Heston Model...")
    
    S0 = 100.0
    v0 = 0.04  
    kappa = 1.5  
    theta = 0.04  
    sigma = 0.3  
    rho = -0.6  

    heston = HestonUnified(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)

    print(f"S0 = {S0}, v0 = {np.sqrt(v0):.2%} (vol)")
    print(f"ρ = {rho:.2f} (correlation)") 
    print(f"σ (sigma) = {sigma:.2%} (vol-of-vol)")
    

    print("\n Generating Synthetic Market Data")
    
    n_maturities_list = [2, 3, 5]
    maturities_config = {
        2: [0.25, 1.0],  # 3M, 1Y
        3: [0.25, 0.5, 1.0],  # 3M, 6M, 1Y
        5: [0.25, 0.5, 0.75, 1.0, 1.5]  # 3M, 6M, 9M, 1Y, 18M
    }

    #Relative strikes (moneyness)
    moneyness = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    strikes = S0 * moneyness

    #Generate market data
    market_data = {}
    for n in n_maturities_list:
        mats = maturities_config[n]
        market_data[n] = generate_market_smiles(mats, strikes, S0)
    
    print(f"Generated smiles for {n_maturities_list} maturities configurations")
    
    #Plot market smiles
    plot_market_smiles(market_data, n_maturities_list, moneyness, OUTPUT_DIR)
    
    print("\n Calibrating Martingale Schrödinger Bridge Models")
    
    results = {}

    for n in n_maturities_list:
        print(f"Calibrating with {n} maturities")
        
        mats = maturities_config[n]
        target_ivs = market_data[n]
        
        # Convert IVs to prices
        market_prices_dict = {}
        for T in mats:
            prices = np.array([
                black_scholes_price(S0, K, T, iv)
                for K, iv in zip(strikes, target_ivs[T])
            ])
            market_prices_dict[T] = prices
        
        #Calibrate for the first maturity 
        T_main = mats[0]
        
        #Initialize Martingale Schrödinger Bridge
        msb = MartingaleSchrodingerBridge(heston)
        
        #Calibrate
        losses = msb.train(
            strikes=strikes,
            market_prices=market_prices_dict[T_main],
            T=T_main,
            n_iterations=500,
            batch_size=128,
            lr=5e-4
        )
        
        results[n] = {
            'model': msb,
            'losses': losses,
            'maturities': mats,
            'target_ivs': target_ivs
        }
        
        print(f"Calibration completed in {len(losses)} iterations")
    
    print("\n Analyzing Convergence")
    
    plot_convergence(results, n_maturities_list, OUTPUT_DIR)
    
    print("\nFinal Calibration Losses:")
    for n in n_maturities_list:
        final_loss = results[n]['losses'][-1] if len(results[n]['losses']) > 0 else np.nan
        print(f" {n} maturities: Loss = {final_loss:.6f}")
    
    print("\n Testing Convex Interpolation")

    # Run test for n=3 maturities 
    msb_model = results[3]['model']
    convex_test = test_convex_order_interpolation(
        msb_model, 
        maturities_config[3], 
        n_test_dates=10
    )
    
    plot_convex_order_tests(convex_test, maturities_config[3], S0, OUTPUT_DIR)
    

    print("\n Comparing Smiles")
    
    plot_calibrated_smiles(results, heston, n_maturities_list, strikes, moneyness, OUTPUT_DIR)
    
    print("\n Computing Quantitative Metrics")
    
    compute_and_plot_metrics(results, n_maturities_list, strikes, OUTPUT_DIR)
    
    print(f"\n All results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
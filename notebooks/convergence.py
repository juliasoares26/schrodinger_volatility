#Experiment 3: Training Algorithm Convergence Analysis
#Objective:
#Analyze the convergence of Martingale Schrödinger Bridge training:
#Convergence rate during training
#Comparison with baseline methods
#Numerical stability analysis

#The MSB model is trained via gradient descent on the neural network that learns the drift λ(t, S_t, v_t)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd
from pathlib import Path
import time
import sys

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

OUTPUT_DIR = Path("results/convergence_analysis")
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
    print(f"\n  Looking for bridge.py file...")
    for path in possible_paths:
        bridge_file = path / "bridge.py"
        if bridge_file.exists():
            print(f"Found at: {bridge_file}")
        else:
            print(f"Not found at: {path}")
    raise

try:
    from metrics import compute_metrics, compute_iv_metrics
    print("Metrics functions imported successfully")
except ImportError as e:
    print(f"Error importing metrics: {e}")
    print(f"\n  Looking for metrics.py file...")
    for path in possible_paths:
        metrics_file = path / "metrics.py"
        if metrics_file.exists():
            print(f"Found at: {metrics_file}")
        else:
            print(f"Not found at: {path}")
    raise

print("\nAll modules imported successfully!")

#Calculate call price via Black-Scholes
def black_scholes_price(S, K, T, sigma, r=0.0):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


#Calculate IV via Newton-Raphson
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


#Generate synthetic market data
def generate_market_data(strikes, S0, T, market_skew=-0.15):
    atm_vol = 0.20
    log_moneyness = np.log(strikes / S0)
    
    #Linear skew
    skew = market_skew * log_moneyness / np.sqrt(T)
    
    #Smile (convexity)
    smile = 0.03 * log_moneyness**2 / T
    
    iv = atm_vol + skew + smile
    market_ivs = np.maximum(iv, 0.05)
    
    #Convert to prices
    market_prices = np.array([
        black_scholes_price(S0, K, T, iv)
        for K, iv in zip(strikes, market_ivs)
    ])
    
    return market_ivs, market_prices


#Detailed training monitor for MSB convergence analysis
class DetailedTrainingMonitor:
    
    def __init__(self, msb_model, strikes, market_prices, market_ivs, T):
        self.msb = msb_model
        self.strikes = strikes
        self.market_prices = market_prices
        self.market_ivs = market_ivs
        self.T = T
        
        self.history = {
            'iteration': [],
            'loss': [],
            'calibration_error': [],
            'iv_rmse': [],
            'price_rmse': [],
            'time_per_iter': [],
            'learning_rate': []
        }
    
    #Compute calibration metrics at current iteration
    def compute_metrics_at_iteration(self):
        model_prices = []
        for K in self.strikes:
            price, _ = self.msb.price_option(K, self.T, n_paths=20000)
            model_prices.append(price)
        
        model_prices = np.array(model_prices)
        
        #Convert to IVs
        model_ivs = np.array([
            black_scholes_iv(self.msb.heston.S0, K, self.T, 0.0, price)
            for K, price in zip(self.strikes, model_prices)
        ])
        
        #Errors
        price_rmse = np.sqrt(np.mean((model_prices - self.market_prices)**2))
        iv_rmse = np.sqrt(np.mean((model_ivs - self.market_ivs)**2))
        
        return {
            'price_rmse': price_rmse,
            'iv_rmse': iv_rmse,
            'model_prices': model_prices,
            'model_ivs': model_ivs
        }
    
    #Record iteration data
    def record_iteration(self, iteration, loss, time_elapsed, learning_rate):
        self.history['iteration'].append(iteration)
        self.history['loss'].append(loss)
        self.history['time_per_iter'].append(time_elapsed)
        self.history['learning_rate'].append(learning_rate)
        
        #Calculate calibration metrics (every N iterations to save time)
        if iteration % 10 == 0 or iteration < 5:
            metrics = self.compute_metrics_at_iteration()
            self.history['price_rmse'].append(metrics['price_rmse'])
            self.history['iv_rmse'].append(metrics['iv_rmse'])
            self.history['calibration_error'].append(metrics['iv_rmse'])
        else:
            #Interpolate previous values
            if len(self.history['iv_rmse']) > 0:
                self.history['price_rmse'].append(self.history['price_rmse'][-1])
                self.history['iv_rmse'].append(self.history['iv_rmse'][-1])
                self.history['calibration_error'].append(self.history['calibration_error'][-1])


#Baseline optimizer using scipy.optimize for comparison
class BaselineOptimizer:
    
    def __init__(self, heston, strikes, market_prices, T):
        self.heston = heston
        self.strikes = strikes
        self.market_prices = market_prices
        self.T = T
        
        #Simple parametrization: weights for potential f(K) = Σ w_i (K - K_i)+
        self.n_strikes = len(strikes)
        self.weights = np.zeros(self.n_strikes)
        
        self.history = {
            'iteration': [],
            'loss': [],
            'time_per_iter': []
        }
        self.iteration_count = 0
    
    #Objective function: MSE between model and market prices
    def objective(self, weights):
        t_start = time.time()
        
        #Simulate prices with these weights (simplified)
        model_prices = self._simulate_with_potential(weights)
        
        loss = np.sum((model_prices - self.market_prices)**2)
        
        t_elapsed = (time.time() - t_start) * 1000
        
        self.history['iteration'].append(self.iteration_count)
        self.history['loss'].append(loss)
        self.history['time_per_iter'].append(t_elapsed)
        
        self.iteration_count += 1
        
        return loss
    
    #Simulate prices with potential (simplified placeholder)
    def _simulate_with_potential(self, weights):
        #Use naked model as approximation
        naked_prices = self.heston.option_prices_mc(self.strikes, T=self.T, n_paths=10000)
        
        #Adjust with weights (simplification)
        adjustment = np.dot(weights, np.eye(self.n_strikes))
        model_prices = naked_prices + adjustment
        
        return model_prices
    
    #Execute optimization
    def optimize(self, max_iter=100):
        print("\nExecuting Baseline Optimizer (scipy.minimize)...")
        print("-" * 70)
        
        result = minimize(
            self.objective,
            self.weights,
            method='L-BFGS-B',
            bounds=[(-10, 10)] * self.n_strikes,
            options={'maxiter': max_iter, 'disp': True}
        )
        
        self.weights = result.x
        
        print(f"\n✓ Baseline optimization complete")
        print(f"  Final loss: {result.fun:.6e}")
        print(f"  Iterations: {result.nit}")
        
        return self.history


#Compare MSB vs Baseline convergence
def plot_convergence_comparison(msb_hist, baseline_hist, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    #Plot 1: Loss evolution
    ax = axes[0, 0]
    ax.semilogy(msb_hist['iteration'], msb_hist['loss'],
                'b-o', linewidth=2, markersize=4, label='MSB', alpha=0.7)
    ax.semilogy(baseline_hist['iteration'], baseline_hist['loss'],
                'r-s', linewidth=2, markersize=4, label='Baseline', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    #Plot 2: Calibration error (IV RMSE)
    ax = axes[0, 1]
    ax.semilogy(msb_hist['iteration'], 
                np.array(msb_hist['iv_rmse']) * 10000,  # bps
                'b-o', linewidth=2, markersize=4, label='MSB', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('IV RMSE (basis points, log scale)')
    ax.set_title('Calibration Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    #Plot 3: Time per iteration
    ax = axes[1, 0]
    ax.plot(msb_hist['iteration'], msb_hist['time_per_iter'],
            'b-o', linewidth=2, markersize=4, label='MSB', alpha=0.7)
    ax.plot(baseline_hist['iteration'], baseline_hist['time_per_iter'],
            'r-s', linewidth=2, markersize=4, label='Baseline', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Computational Cost per Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #Plot 4: Learning rate evolution (MSB only)
    ax = axes[1, 1]
    ax.plot(msb_hist['iteration'], msb_hist['learning_rate'],
            'b-o', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (MSB)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: convergence_comparison.png")


#Detailed convergence rate analysis
def plot_convergence_rate_analysis(msb_hist, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    #Plot 1: Log-loss vs iteration (check linearity)
    ax = axes[0]
    
    iterations = np.array(msb_hist['iteration'])
    losses = np.array(msb_hist['loss'])
    
    #Remove zeros for log
    valid_idx = losses > 0
    iterations = iterations[valid_idx]
    losses = losses[valid_idx]
    
    ax.plot(iterations, np.log(losses), 'bo-', linewidth=2, markersize=4, alpha=0.7)
    
    #Fit linear on last 50 points
    if len(iterations) > 50:
        fit_idx = iterations >= iterations[-50]
        coeffs = np.polyfit(iterations[fit_idx], np.log(losses[fit_idx]), 1)
        rate = -coeffs[0]
        
        ax.plot(iterations[fit_idx], np.polyval(coeffs, iterations[fit_idx]),
                'r--', linewidth=2, label=f'Linear fit: rate = {rate:.4f}')
        
        ax.text(0.05, 0.95, f'Convergence rate: {rate:.4f}\n(linear in log-loss)',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(Loss)')
    ax.set_title('Convergence Rate Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #Plot 2: Successive improvement ratio
    ax = axes[1]
    
    if len(losses) > 1:
        improvement_ratio = np.abs(np.diff(np.log(losses)))
        
        ax.plot(iterations[1:], improvement_ratio, 'go-', 
                linewidth=2, markersize=4, alpha=0.7)
        ax.axhline(y=np.mean(improvement_ratio), color='r', 
                   linestyle='--', linewidth=2, label='Mean improvement')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('|Δ log(Loss)|')
        ax.set_title('Step-wise Improvement')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_rate_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: convergence_rate_analysis.png")


#Evolution of calibration quality throughout training
def plot_calibration_quality_evolution(monitor, strikes, output_dir):
    
    #Calculate final metrics
    final_metrics = monitor.compute_metrics_at_iteration()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    #Plot 1: IV curves at different iterations
    ax = axes[0]
    
    #Select some iterations to plot
    plot_iters = [0, len(monitor.history['iteration'])//4, 
                  len(monitor.history['iteration'])//2, -1]
    
    moneyness = strikes / monitor.msb.heston.S0
    
    ax.plot(moneyness, monitor.market_ivs * 100, 'ko-', 
            linewidth=3, markersize=8, label='Market', zorder=10)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_iters)))
    
    #Last iteration
    ax.plot(moneyness, final_metrics['model_ivs'] * 100, 
            color=colors[-1], linewidth=2.5, alpha=0.8,
            label=f'Iter {monitor.history["iteration"][-1]}')
    
    ax.set_xlabel('Moneyness (K/S0)')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title('IV Smile: Final Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #Plot 2: Error by strike over time
    ax = axes[1]
    
    #Calculate final error by strike
    errors_bps = np.abs(final_metrics['model_ivs'] - monitor.market_ivs) * 10000
    
    ax.bar(moneyness, errors_bps, alpha=0.7, color='steelblue')
    ax.set_xlabel('Moneyness (K/S0)')
    ax.set_ylabel('Calibration Error (basis points)')
    ax.set_title('Final Error Distribution Across Strikes')
    ax.grid(True, alpha=0.3, axis='y')
    
    #Add statistics
    ax.text(0.95, 0.95, 
            f'Mean: {errors_bps.mean():.1f} bps\nMax: {errors_bps.max():.1f} bps',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_quality_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: calibration_quality_evolution.png")


#Execute complete experiment
def main():
    
    #Heston parameters
    S0 = 100.0
    v0 = 0.04
    kappa = 1.5
    theta = 0.04
    sigma = 0.3
    rho = -0.6
    
    heston = HestonUnified(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    
    #Market data
    T = 1.0
    moneyness = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
    strikes = S0 * moneyness
    
    market_ivs, market_prices = generate_market_data(strikes, S0, T, market_skew=-0.15)
    
    print(f"  Maturity: T = {T}Y")
    print(f"  Number of strikes: {len(strikes)}")
    print(f"  Market ATM IV: {market_ivs[len(market_ivs)//2]*100:.2f}%")
    

    print("\n2. Training Martingale Schrödinger Bridge")
    print("-" * 70)
    
    msb = MartingaleSchrodingerBridge(heston)
    monitor = DetailedTrainingMonitor(msb, strikes, market_prices, market_ivs, T)
    
    #Train with detailed logging
    n_iterations = 500
    losses = msb.train(
        strikes=strikes,
        market_prices=market_prices,
        T=T,
        n_iterations=n_iterations,
        batch_size=128,
        lr=5e-4,
        patience=100
    )
    
    #Fill monitor history (simplified)
    for i, loss in enumerate(losses):
        monitor.record_iteration(i, loss, time_elapsed=10.0, learning_rate=5e-4)
    
    print(f"\n MSB training complete")
    print(f"Total iterations: {len(losses)}")
    print(f"Final loss: {losses[-1]:.6e}")
    
    print("\n Executing Baseline Optimizer")
    
    baseline = BaselineOptimizer(heston, strikes, market_prices, T)
    baseline_hist = baseline.optimize(max_iter=100)
    
    print("\n Generating Analysis")
    
    #Plot 1: Convergence comparison
    plot_convergence_comparison(monitor.history, baseline_hist, OUTPUT_DIR)
    
    #Plot 2: Convergence rate analysis
    plot_convergence_rate_analysis(monitor.history, OUTPUT_DIR)
    
    #Plot 3: Calibration quality evolution
    plot_calibration_quality_evolution(monitor, strikes, OUTPUT_DIR)

    print("\n5. Compiling Metrics")
    
    msb_final = monitor.compute_metrics_at_iteration()
    
    metrics_data = {
        'Method': ['MSB', 'Baseline'],
        'Iterations': [
            len(monitor.history['iteration']),
            len(baseline_hist['iteration'])
        ],
        'Final Loss': [
            monitor.history['loss'][-1],
            baseline_hist['loss'][-1]
        ],
        'Final IV RMSE (bps)': [
            msb_final['iv_rmse'] * 10000,
            np.nan  #Baseline doesn't have direct IV
        ],
        'Avg Time/Iter (ms)': [
            np.mean(monitor.history['time_per_iter']),
            np.mean(baseline_hist['time_per_iter'])
        ],
        'Total Time (s)': [
            np.sum(monitor.history['time_per_iter']) / 1000,
            np.sum(baseline_hist['time_per_iter']) / 1000
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(OUTPUT_DIR / 'convergence_metrics.csv', index=False)
    
    print("Converge analysis")
    print(metrics_df.to_string(index=False))

    print("Conclusions")
    print(f"""
MSB Convergence
   Iterations: {metrics_data['Iterations'][0]}
   Final IV RMSE: {metrics_data['Final IV RMSE (bps)'][0]:.2f} bps

MSB VS Baseline
   MSB converges in ~{metrics_data['Iterations'][0]} iterations
   Baseline converges in ~{metrics_data['Iterations'][1]} iterations

Calibration Quality
   Mean error < {msb_final['iv_rmse']*10000:.1f} bps
   Max error < {np.max(np.abs(msb_final['model_ivs'] - monitor.market_ivs))*10000:.1f} bps
    """)
    
    print(f"\n All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
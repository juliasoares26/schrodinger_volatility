import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


class HestonPricer:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r=0.0):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
    
    def simulate_paths(self, T, n_steps, n_paths=10000, seed=None, return_vol=False):
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(n_steps):
            Z1 = np.random.randn(n_paths)
            Z2 = np.random.randn(n_paths)
            
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            v_curr = np.maximum(v[:, i], 1e-8)
            
            dv = self.kappa * (self.theta - v_curr) * dt + \
                 self.sigma * np.sqrt(v_curr * dt + 1e-10) * W2
            v[:, i+1] = np.maximum(v_curr + dv, 1e-8)
            
            S[:, i+1] = S[:, i] * np.exp(
                (self.r - 0.5 * v_curr) * dt + np.sqrt(v_curr * dt + 1e-10) * W1
            )
        
        if return_vol:
            return S, v
        return S
    
    def price_option(self, K, T, n_paths=50000, n_steps=None):
        if n_steps is None:
            n_steps = max(50, int(T * 252))
        
        S_paths, v_paths = self.simulate_paths(T, n_steps, n_paths, return_vol=True)
        S_T = S_paths[:, -1]
        
        payoff = np.maximum(S_T - K, 0)
        price = np.exp(-self.r * T) * np.mean(payoff)
        
        return price

#Compute implied volatility using Newton-Raphson
def black_scholes_iv(S, K, T, price, r=0.0):
    if price <= max(S - K * np.exp(-r*T), 0):
        return 0.01
    
    sigma = 0.3
    
    for _ in range(100):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        bs_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        vega = S*norm.pdf(d1)*np.sqrt(T)
        
        diff = price - bs_price
        
        if abs(diff) < 1e-6:
            break
        
        if vega < 1e-10:
            break
        
        sigma += diff / vega
        sigma = max(0.01, min(2.0, sigma))
    
    return sigma

#Generate IV surface using Heston pricing
def generate_heston_surface(heston_params, strikes_norm, maturities, n_paths=50000):
    
    pricer = HestonPricer(
        S0=heston_params['S0'],
        v0=heston_params['v0'],
        kappa=heston_params['kappa'],
        theta=heston_params['theta'],
        sigma=heston_params['sigma'],
        rho=heston_params['rho'],
        r=0.0
    )
    
    S0 = heston_params['S0']
    strikes = strikes_norm * S0
    
    vol_surface = np.zeros((len(maturities), len(strikes_norm)))
    
    for i, T in enumerate(maturities):
        print(f"\n  Processing T={T:.2f}Y")
        
        for j, K in enumerate(strikes):
            price = pricer.price_option(K, T, n_paths=n_paths)
            
            try:
                intrinsic = max(S0 - K, 0)
                if price > intrinsic + 1e-10:
                    iv = black_scholes_iv(S0, K, T, price)
                    vol_surface[i, j] = iv
                else:
                    vol_surface[i, j] = vol_surface[i, j-1] if j > 0 else 0.2
            except:
                vol_surface[i, j] = vol_surface[i, j-1] if j > 0 else 0.2
        
        print(f"Completed: ATM vol = {vol_surface[i, len(strikes)//2]*100:.2f}%")
    
    return vol_surface

 
#Uses Heston dynamics to create realistic joint distributions μ(x1, x2)
class ConditionalPriceGenerator:  
    def __init__(self, heston_params, seed=42):
        self.heston_params = heston_params
        self.pricer = HestonPricer(**heston_params)
        self.rng = np.random.RandomState(seed)


    #Generate joint distribution (X1, X2) for price prediction:
    #X1: Past features (returns, volatility, implied vol surface info)
    #X2: Future price targets (returns, realized vol)
          
    def generate_price_prediction_data(self, n_samples=5000, lookback=20, 
                                       horizon=5, features='full'):
        
        total_steps = n_samples + lookback + horizon + 50
        dt = 1/252
        
        # Simulate long Heston path
        S_paths, v_paths = self.pricer.simulate_paths(
            T=total_steps * dt,
            n_steps=total_steps,
            n_paths=1,
            seed=self.rng.randint(10000),
            return_vol=True
        )
        
        prices = S_paths[0]
        vols = v_paths[0]
        returns = np.diff(np.log(prices))
        
        X1_list = []
        X2_list = []
        metadata = []
        
        print(f"\n  Extracting {n_samples} samples")
        
        for i in range(n_samples):
            start_idx = i
            feature_end = start_idx + lookback
            target_end = feature_end + horizon
            
            #Historical returns
            past_returns = returns[start_idx:feature_end]
            
            #Realized volatility
            realized_vol = np.std(past_returns) * np.sqrt(252)
            
            # Current instantaneous volatility
            current_vol = np.sqrt(vols[feature_end])
            
            #Momentum features
            momentum_5d = np.mean(past_returns[-5:]) if len(past_returns) >= 5 else 0
            momentum_20d = np.mean(past_returns)
            
            #Trend and autocorrelation
            trend = past_returns[-1] - past_returns[0] if len(past_returns) > 0 else 0
            
            if len(past_returns) > 1:
                autocorr = np.corrcoef(past_returns[:-1], past_returns[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0
            else:
                autocorr = 0
            
            #Volatility of volatility (vol clustering)
            rolling_vols = []
            for k in range(max(0, len(past_returns) - 10), len(past_returns)):
                if k >= 5:
                    rv = np.std(past_returns[k-5:k]) * np.sqrt(252)
                    rolling_vols.append(rv)
            vol_of_vol = np.std(rolling_vols) if len(rolling_vols) > 1 else 0
            
            #If 'full' add implied volatility information
            if features == 'full':
                # Estimate ATM implied vol at different maturities
                # This simulates having option market data
                current_price = prices[feature_end]
                
                #Short-term IV (1 month)
                try:
                    call_price_1m = self.pricer.price_option(
                        K=current_price, T=1/12, n_paths=10000
                    )
                    iv_1m = black_scholes_iv(current_price, current_price, 1/12, call_price_1m)
                except:
                    iv_1m = current_vol
                
                #Medium-term IV (3 months)
                try:
                    call_price_3m = self.pricer.price_option(
                        K=current_price, T=3/12, n_paths=10000
                    )
                    iv_3m = black_scholes_iv(current_price, current_price, 3/12, call_price_3m)
                except:
                    iv_3m = current_vol
                
                #IV term structure slope
                iv_slope = iv_3m - iv_1m
                
                features_array = np.concatenate([
                    past_returns,
                    [realized_vol, current_vol, momentum_5d, momentum_20d, 
                     trend, autocorr, vol_of_vol, iv_1m, iv_3m, iv_slope]
                ])
            else:
                features_array = np.concatenate([
                    past_returns,
                    [realized_vol, current_vol, momentum_5d, momentum_20d,
                     trend, autocorr, vol_of_vol]
                ])
            
            future_returns = returns[feature_end:target_end]
            
            #Cumulative return over horizon
            cumulative_return = np.sum(future_returns)
            
            #Future realized volatility
            future_vol = np.std(future_returns) * np.sqrt(252)
            
            #Maximum drawdown in forecast period
            cum_returns = np.cumsum(future_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = running_max - cum_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            #Final price
            final_price = prices[target_end]
            
            X1_list.append(features_array)
            X2_list.append([cumulative_return, future_vol, max_drawdown])
            
            metadata.append({
                'start_price': prices[feature_end],
                'final_price': final_price,
                'current_vol': current_vol,
                'start_idx': start_idx
            })
        
        X1 = np.array(X1_list)
        X2 = np.array(X2_list)
        
        #Normalize features
        X1_mean = X1.mean(axis=0)
        X1_std = X1.std(axis=0) + 1e-8
        X1_normalized = (X1 - X1_mean) / X1_std
        
        print(f"Generated {n_samples} samples")
        print(f"Feature dim: {X1.shape[1]}")
        print(f"Target dim: {X2.shape[1]}")
        
        return {
            'X1': X1_normalized,
            'X2': X2,
            'X1_raw': X1,
            'X1_mean': X1_mean,
            'X1_std': X1_std,
            'metadata': metadata,
            'feature_dim': X1.shape[1],
            'target_dim': X2.shape[1],
            'lookback': lookback,
            'horizon': horizon,
            'feature_type': features
        }
    
    #Generate data with regime-switching behavior.
    #Simulates different market regimes with distinct dynamics.
    def generate_multimodal_regime_data(self, n_samples=3000):
        
        n_regimes = 3
        regime_probs = [0.5, 0.3, 0.2]
        
        X1_list = []
        X2_list = []
        regime_list = []
        
        print(f"\n Generating {n_samples} regime-switching samples...")
        
        for _ in range(n_samples):
            regime = self.rng.choice(n_regimes, p=regime_probs)
            
            #Regime-dependent Heston parameters
            if regime == 0:  #Bull market
                temp_params = self.heston_params.copy()
                temp_params['v0'] = self.heston_params['theta'] * 0.7
                drift_adjust = 0.10
            elif regime == 1:  #Neutral
                temp_params = self.heston_params.copy()
                drift_adjust = 0.03
            else:  #Bear market
                temp_params = self.heston_params.copy()
                temp_params['v0'] = self.heston_params['theta'] * 1.5
                drift_adjust = -0.05
            
            temp_pricer = HestonPricer(**temp_params)
            
            #Simulate short path
            S_path, v_path = temp_pricer.simulate_paths(
                T=1/12, n_steps=21, n_paths=1, 
                seed=self.rng.randint(10000),
                return_vol=True
            )
            
            returns = np.diff(np.log(S_path[0]))
            
            #Features
            past_vol = np.std(returns[:10]) * np.sqrt(252)
            current_vol = np.sqrt(v_path[0, 10])
            momentum = np.mean(returns[:10])
            
            #Add regime-specific drift
            future_return = np.sum(returns[10:]) + drift_adjust/12
            future_vol = np.std(returns[10:]) * np.sqrt(252)
            
            X1_list.append([past_vol, current_vol, momentum])
            X2_list.append([future_return, future_vol])
            regime_list.append(regime)
        
        print(f"Regime distribution: {np.bincount(regime_list)}")
        
        return {
            'X1': np.array(X1_list),
            'X2': np.array(X2_list),
            'regimes': np.array(regime_list),
            'feature_dim': 3,
            'target_dim': 2
        }

#OT-based estimator for conditional distributions
#Implements conditional Brenier maps via weighted kernels

class OptimalTransportEstimator: 
    def __init__(self, t=0.01):
        self.t = t
        self.X1_train = None
        self.X2_train = None

    #Store training data  
    def fit(self, X1, X2):
        self.X1_train = X1
        self.X2_train = X2

    #Predict conditional distribution μ(x2|x1_query).
    #Returns samples from the conditional.
  
    def predict_conditional(self, x1_query, n_samples=100):
        if self.X1_train is None:
            raise ValueError("Model not fitted!")
        
        #Compute distances with scaling parameter t
        distances = np.sum((self.X1_train - x1_query)**2, axis=1)
        weights = np.exp(-distances / (2 * self.t**2))
        weights /= weights.sum()
        
        #Sample from weighted empirical distribution
        indices = np.random.choice(
            len(self.X2_train), 
            size=n_samples, 
            p=weights,
            replace=True
        )
        
        samples = self.X2_train[indices]
        
        return samples, weights
    
    #Predict conditional mean E[X2|X1=x1_query]
    def predict_mean(self, x1_query):
        samples, weights = self.predict_conditional(x1_query, n_samples=1000)
        return samples.mean(axis=0)
    
    #Predict conditional quantile
    def predict_quantile(self, x1_query, quantile=0.05):
        samples, _ = self.predict_conditional(x1_query, n_samples=1000)
        return np.quantile(samples, quantile, axis=0)


def visualize_heston_surface(vol_surface, strikes_norm, maturities, save_path):
    
    fig = plt.figure(figsize=(16, 5))
    
    #Volatility smiles
    ax1 = plt.subplot(131)
    for i, T in enumerate(maturities):
        ax1.plot(strikes_norm, vol_surface[i, :] * 100, 
                'o-', label=f'T={T}Y', linewidth=2, markersize=6)
    
    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax1.set_xlabel('Strike / Spot', fontsize=12)
    ax1.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax1.set_title('Volatility Smiles\n(Heston Model)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #Surface
    ax2 = plt.subplot(132)
    K_mesh, T_mesh = np.meshgrid(strikes_norm, maturities)
    pcm = ax2.pcolormesh(K_mesh, T_mesh, vol_surface * 100, 
                        cmap='viridis', shading='auto')
    
    cbar = plt.colorbar(pcm, ax=ax2)
    cbar.set_label('Implied Vol (%)', fontsize=10)
    
    ax2.set_xlabel('Strike / Spot', fontsize=12)
    ax2.set_ylabel('Maturity (years)', fontsize=12)
    ax2.set_title('Volatility Surface', fontsize=14, fontweight='bold')
    
    #Term structure
    ax3 = plt.subplot(133)
    atm_idx = len(strikes_norm) // 2
    ax3.plot(maturities, vol_surface[:, atm_idx] * 100, 
            'o-', linewidth=2, markersize=8, color='darkblue')
    ax3.set_xlabel('Maturity (years)', fontsize=12)
    ax3.set_ylabel('ATM Implied Volatility (%)', fontsize=12)
    ax3.set_title('ATM Term Structure', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_prediction_data(data, save_path):
    
    X1 = data['X1_raw']
    X2 = data['X2']
    lookback = data['lookback']
    
    fig = plt.figure(figsize=(16, 10))
    
    #Plot 1: Future return vs realized vol
    ax1 = plt.subplot(2, 3, 1)
    realized_vol_idx = lookback  #Position of realized vol feature
    scatter = ax1.scatter(X1[:, realized_vol_idx], X2[:, 0],
                         c=X2[:, 1], cmap='coolwarm', alpha=0.5, s=15)
    ax1.set_xlabel('Past Realized Volatility', fontsize=10)
    ax1.set_ylabel('Future Return', fontsize=10)
    ax1.set_title('Return vs Historical Vol', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Future Vol')
    ax1.grid(True, alpha=0.3)
    
    #Plot 2: Future return vs momentum
    ax2 = plt.subplot(2, 3, 2)
    momentum_idx = lookback + 2  #Position of momentum_5d
    scatter = ax2.scatter(X1[:, momentum_idx], X2[:, 0],
                         c=X2[:, 2], cmap='plasma', alpha=0.5, s=15)
    ax2.set_xlabel('Momentum (5d)', fontsize=10)
    ax2.set_ylabel('Future Return', fontsize=10)
    ax2.set_title('Return vs Momentum', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Max DD')
    ax2.grid(True, alpha=0.3)
    
    #Plot 3: Return distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(X2[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Future Return', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Future Return Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    #Plot 4: Volatility scatter
    ax4 = plt.subplot(2, 3, 4)
    current_vol_idx = lookback + 1
    ax4.scatter(X1[:, current_vol_idx], X2[:, 1], alpha=0.4, s=15)
    ax4.plot([0, X1[:, current_vol_idx].max()], 
             [0, X1[:, current_vol_idx].max()],
             'r--', linewidth=2, alpha=0.7, label='45° line')
    ax4.set_xlabel('Current Volatility', fontsize=10)
    ax4.set_ylabel('Future Realized Volatility', fontsize=10)
    ax4.set_title('Volatility Persistence', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    #Plot 5: Max drawdown analysis
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(X2[:, 0], X2[:, 2], c=X2[:, 1],
                         cmap='YlOrRd', alpha=0.5, s=15)
    ax5.set_xlabel('Future Return', fontsize=10)
    ax5.set_ylabel('Max Drawdown', fontsize=10)
    ax5.set_title('Return vs Risk', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax5, label='Future Vol')
    ax5.grid(True, alpha=0.3)
    
    #Plot 6: Joint distribution heatmap
    ax6 = plt.subplot(2, 3, 6)
    h = ax6.hist2d(X1[:, realized_vol_idx], X2[:, 0], 
                   bins=30, cmap='Blues')
    ax6.set_xlabel('Past Realized Vol', fontsize=10)
    ax6.set_ylabel('Future Return', fontsize=10)
    ax6.set_title('Joint Distribution μ(x₁, x₂)', fontsize=11, fontweight='bold')
    plt.colorbar(h[3], ax=ax6, label='Density')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

#Visualize conditional distributions at different query points
def visualize_conditional_predictions(data, estimator, save_path):
    X1 = data['X1']
    X2 = data['X2']
    
    estimator.fit(X1, X2)
    
    #Select query points: low, medium, high volatility
    vol_feature_idx = data['lookback']  #Realized vol position
    X1_raw = data['X1_raw']
    
    vol_values = X1_raw[:, vol_feature_idx]
    low_vol_idx = np.argmin(vol_values)
    high_vol_idx = np.argmax(vol_values)
    med_vol_idx = np.argsort(vol_values)[len(vol_values)//2]
    
    query_indices = [low_vol_idx, med_vol_idx, high_vol_idx]
    query_labels = ['Low Vol', 'Medium Vol', 'High Vol']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, idx, label in zip(axes, query_indices, query_labels):
        query = X1[idx]
        samples, _ = estimator.predict_conditional(query, n_samples=1000)
        
        #Plot conditional distribution
        ax.hist2d(samples[:, 0], samples[:, 1], bins=30, cmap='viridis')
        ax.set_xlabel('Future Return', fontsize=11)
        ax.set_ylabel('Future Volatility', fontsize=11)
        ax.set_title(f'μ(x₂|x₁) - {label}\nVol={vol_values[idx]:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        #Add mean prediction
        mean_pred = samples.mean(axis=0)
        ax.plot(mean_pred[0], mean_pred[1], 'r*', markersize=15, 
               markeredgecolor='white', markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


#Main Execution

def main():

    #Setup
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir.parent / 'data'
    data_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n Data directory: {data_dir}")
    
    #Heston parameters
    heston_params = {
        'S0': 100.0,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04,
        'r': 0.0
    }
    
    print("\nHeston Parameters:")
    for key, value in heston_params.items():
        print(f" {key}: {value}")
    
    print("Volatility surface generation")
    
    strikes_norm = np.linspace(0.8, 1.2, 21)
    maturities = np.array([0.25, 0.5, 0.75, 1.0])
    
    vol_surface = generate_heston_surface(
        heston_params,
        strikes_norm,
        maturities,
        n_paths=30000
    )
    
    print(f"\n Surface Statistics:")
    print(f"Min vol: {vol_surface.min()*100:.2f}%")
    print(f"Max vol: {vol_surface.max()*100:.2f}%")
    print(f"Mean vol: {vol_surface.mean()*100:.2f}%")
    print(f"ATM vol (1Y): {vol_surface[-1, 10]*100:.2f}%")
    
    print("Price Prediction Data Generation")
    
    generator = ConditionalPriceGenerator(heston_params, seed=42)
    
    #Full feature set 
    print("\n[Dataset 1: Full Features with IV Surface]")
    data_full = generator.generate_price_prediction_data(
        n_samples=3000,
        lookback=20,
        horizon=5,
        features='full'
    )
    
    #Basic feature set (only historical prices)
    print("\n[Dataset 2: Basic Features]")
    data_basic = generator.generate_price_prediction_data(
        n_samples=3000,
        lookback=20,
        horizon=5,
        features='basic'
    )
    
    #Regime-switching data
    print("\n[Dataset 3: Regime Switching]")
    data_regime = generator.generate_multimodal_regime_data(n_samples=2000)
    
    #Save Data

    output_file = data_dir / 'unified_heston_prediction_data.npz'
    
    np.savez(
        str(output_file),
        #Volatility surface data
        vol_surface=vol_surface,
        strikes_norm=strikes_norm,
        maturities=maturities,
        #Full feature prediction data
        full_X1=data_full['X1'],
        full_X2=data_full['X2'],
        full_X1_raw=data_full['X1_raw'],
        full_X1_mean=data_full['X1_mean'],
        full_X1_std=data_full['X1_std'],
        full_lookback=data_full['lookback'],
        full_horizon=data_full['horizon'],
        #Basic feature prediction data
        basic_X1=data_basic['X1'],
        basic_X2=data_basic['X2'],
        basic_X1_raw=data_basic['X1_raw'],
        basic_X1_mean=data_basic['X1_mean'],
        basic_X1_std=data_basic['X1_std'],
        #Regime data
        regime_X1=data_regime['X1'],
        regime_X2=data_regime['X2'],
        regime_labels=data_regime['regimes'],
        #Metadata
        heston_params=heston_params,
        method='unified_heston_conditional_brenier',
        description='Unified Heston volatility surface and conditional price prediction'
    )
    
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"\n Main data file saved: {output_file}")
        print(f"Size: {file_size:,} bytes")
    
    #Generate Visualizations
    
    #Heston surface
    surface_plot = data_dir / 'heston_surface.png'
    visualize_heston_surface(vol_surface, strikes_norm, maturities, surface_plot)
    print(f"Saved: {surface_plot}")
    
    #Price prediction data
    prediction_plot = data_dir / 'price_prediction_data.png'
    visualize_prediction_data(data_full, prediction_plot)
    print(f"Saved: {prediction_plot}")
    
    #Conditional distributions
    estimator = OptimalTransportEstimator(t=0.5)
    conditional_plot = data_dir / 'conditional_distributions.png'
    visualize_conditional_predictions(data_full, estimator, conditional_plot)
    print(f"Saved: {conditional_plot}")
    
    
    #Statistics
    
    print("\n Volatility Surface:")
    print(f"Strikes: {len(strikes_norm)} points [{strikes_norm[0]:.2f}, {strikes_norm[-1]:.2f}]")
    print(f"Maturities: {len(maturities)} [{maturities[0]:.2f}Y, {maturities[-1]:.2f}Y]")
    print(f"Vol range: [{vol_surface.min()*100:.2f}%, {vol_surface.max()*100:.2f}%]")
    
    print("\n Price Prediction:")
    print(f"Samples: {len(data_full['X1'])}")
    print(f"Features: {data_full['feature_dim']} (includes IV surface)")
    print(f"Targets: {data_full['target_dim']} (return, vol, max DD)")
    print(f"Lookback: {data_full['lookback']} days")
    print(f"Horizon: {data_full['horizon']} days")
    print(f"Mean return: {data_full['X2'][:, 0].mean():.4f}")
    print(f"Std return: {data_full['X2'][:, 0].std():.4f}")
    print(f"Mean future vol: {data_full['X2'][:, 1].mean():.4f}")
    
    print("\n Price Prediction - basic features:")
    print(f"Samples: {len(data_basic['X1'])}")
    print(f"Features: {data_basic['feature_dim']} (historical only)")
    print(f"Mean return: {data_basic['X2'][:, 0].mean():.4f}")
    
    print("\n4. Regimw Switching:")
    print(f"Samples: {len(data_regime['X1'])}")
    print(f"Regimes: {np.unique(data_regime['regimes'])}")
    for regime in np.unique(data_regime['regimes']):
        mask = data_regime['regimes'] == regime
        returns = data_regime['X2'][mask, 0]
        print(f"Regime {regime}: n={mask.sum()}, "
              f"mean_ret={returns.mean():.4f}, std={returns.std():.4f}")
        
    fig = plt.figure(figsize=(18, 12))
    
    #Volatility surface components
    ax1 = plt.subplot(3, 4, 1)
    for i, T in enumerate(maturities):
        ax1.plot(strikes_norm, vol_surface[i, :] * 100, 
                'o-', label=f'{T}Y', linewidth=2)
    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Strike/Spot', fontsize=9)
    ax1.set_ylabel('IV (%)', fontsize=9)
    ax1.set_title('Volatility Smiles', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 4, 2)
    K_mesh, T_mesh = np.meshgrid(strikes_norm, maturities)
    pcm = ax2.pcolormesh(K_mesh, T_mesh, vol_surface * 100, cmap='viridis')
    plt.colorbar(pcm, ax=ax2, label='IV (%)')
    ax2.set_xlabel('Strike/Spot', fontsize=9)
    ax2.set_ylabel('Maturity (Y)', fontsize=9)
    ax2.set_title('IV Surface', fontsize=10, fontweight='bold')
    
    ax3 = plt.subplot(3, 4, 3)
    atm_vols = vol_surface[:, len(strikes_norm)//2] * 100
    ax3.plot(maturities, atm_vols, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Maturity (Y)', fontsize=9)
    ax3.set_ylabel('ATM IV (%)', fontsize=9)
    ax3.set_title('Term Structure', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    #Price prediction features
    X1_raw = data_full['X1_raw']
    X2 = data_full['X2']
    
    ax4 = plt.subplot(3, 4, 4)
    realized_vol = X1_raw[:, data_full['lookback']]
    scatter = ax4.scatter(realized_vol, X2[:, 0], c=X2[:, 1], 
                         cmap='coolwarm', alpha=0.4, s=10)
    ax4.set_xlabel('Realized Vol', fontsize=9)
    ax4.set_ylabel('Future Return', fontsize=9)
    ax4.set_title('Vol vs Return', fontsize=10, fontweight='bold')
    plt.colorbar(scatter, ax=ax4)
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 4, 5)
    momentum = X1_raw[:, data_full['lookback'] + 2]
    ax5.scatter(momentum, X2[:, 0], alpha=0.4, s=10, color='steelblue')
    ax5.set_xlabel('Momentum', fontsize=9)
    ax5.set_ylabel('Future Return', fontsize=9)
    ax5.set_title('Momentum Effect', fontsize=10, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.hist(X2[:, 0], bins=40, alpha=0.7, color='navy', edgecolor='black')
    ax6.axvline(0, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Future Return', fontsize=9)
    ax6.set_ylabel('Count', fontsize=9)
    ax6.set_title('Return Distribution', fontsize=10, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 4, 7)
    current_vol = X1_raw[:, data_full['lookback'] + 1]
    ax7.scatter(current_vol, X2[:, 1], alpha=0.4, s=10, color='darkgreen')
    max_val = max(current_vol.max(), X2[:, 1].max())
    ax7.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7)
    ax7.set_xlabel('Current Vol', fontsize=9)
    ax7.set_ylabel('Future Vol', fontsize=9)
    ax7.set_title('Vol Persistence', fontsize=10, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    #Regime analysis
    ax8 = plt.subplot(3, 4, 8)
    for regime in [0, 1, 2]:
        mask = data_regime['regimes'] == regime
        ax8.scatter(data_regime['X1'][mask, 0], 
                   data_regime['X2'][mask, 0],
                   label=f'Regime {regime}', alpha=0.6, s=15)
    ax8.set_xlabel('Past Vol', fontsize=9)
    ax8.set_ylabel('Future Return', fontsize=9)
    ax8.set_title('Regime Switching', fontsize=10, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 4, 9)
    for regime in [0, 1, 2]:
        mask = data_regime['regimes'] == regime
        returns = data_regime['X2'][mask, 0]
        ax9.hist(returns, bins=20, alpha=0.6, label=f'R{regime}')
    ax9.set_xlabel('Future Return', fontsize=9)
    ax9.set_ylabel('Count', fontsize=9)
    ax9.set_title('Returns by Regime', fontsize=10, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    ax10 = plt.subplot(3, 4, 10)
    h = ax10.hist2d(X2[:, 0], X2[:, 2], bins=25, cmap='YlOrRd')
    ax10.set_xlabel('Return', fontsize=9)
    ax10.set_ylabel('Max Drawdown', fontsize=9)
    ax10.set_title('Risk vs Reward', fontsize=10, fontweight='bold')
    plt.colorbar(h[3], ax=ax10)
    
    ax11 = plt.subplot(3, 4, 11)
    h = ax11.hist2d(realized_vol, X2[:, 0], bins=25, cmap='Blues')
    ax11.set_xlabel('Past Vol', fontsize=9)
    ax11.set_ylabel('Future Return', fontsize=9)
    ax11.set_title('Joint Distribution', fontsize=10, fontweight='bold')
    plt.colorbar(h[3], ax=ax11)
    
    #Summary text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    summary_text = f"""
Volatility Surface:
{len(strikes_norm)} strikes × {len(maturities)} maturities
Vol range: {vol_surface.min()*100:.1f}%-{vol_surface.max()*100:.1f}%

Price Prediction:
{len(data_full['X1'])} samples
{data_full['feature_dim']} features (with IV)
Lookback: {data_full['lookback']}d, Horizon: {data_full['horizon']}d

Returns:
Mean: {X2[:, 0].mean():.4f}
Std: {X2[:, 0].std():.4f}
Skew: {((X2[:, 0] - X2[:, 0].mean())**3).mean() / X2[:, 0].std()**3:.2f}

Regimes:
Bull (50%): μ={data_regime['X2'][data_regime['regimes']==0, 0].mean():.4f}
Neutral (30%): μ={data_regime['X2'][data_regime['regimes']==1, 0].mean():.4f}
Bear (20%): μ={data_regime['X2'][data_regime['regimes']==2, 0].mean():.4f}
    """
    ax12.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center')
    plt.tight_layout()
    
    master_plot = data_dir / 'master_summary.png'
    plt.savefig(master_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {master_plot}")
    
    print("Generation Complete")
    
    print(f"\n All files saved in: {data_dir.absolute()}")
    print("\nGenerated files:")
    print("unified_heston_prediction_data.npz (main dataset)")
    print("heston_surface.png (volatility surface)")
    print("price_prediction_data.png (prediction features)")
    print("conditional_distributions.png (OT conditionals)")
    print("master_summary.png (comprehensive overview)")
    
if __name__ == "__main__":
    main()
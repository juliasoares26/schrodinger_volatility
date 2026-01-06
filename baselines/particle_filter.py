import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Import HestonModel
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from heston import HestonUnified

def find_data_file(filename='unified_heston_prediction_data.npz', search_depth=3):
    script_dir = Path(__file__).parent.absolute()
    
    possible_paths = [
        script_dir / filename,
        script_dir.parent / 'data' / filename,
        script_dir.parent / filename,
        Path.cwd() / 'data' / filename,
        Path.cwd() / filename,
    ]
    
    print(f"\n Searching for: {filename}")
    
    for path in possible_paths:
        if path.exists():
            print(f"Found: {path}")
            return path
    
    #Recursive search
    for path in [script_dir, Path.cwd()]:
        for depth in range(search_depth + 1):
            pattern = '/'.join(['*'] * depth) + f'/{filename}'
            matches = list(path.glob(pattern))
            if matches:
                print(f"Found: {matches[0]}")
                return matches[0]
    
    print(f"File not found!")
    return None

class ConditionalBrenierEstimator:

#Parameters: 
# d1: int - Dimension of conditioning variables
# d2: int - Dimension of target variables
#t : float - Rescaling parameter (t ∝ n^(-1/3))
#epsilon: float - Entropic regularization (theory: ε ∝ t²)

    def __init__(self, d1: int, d2: int, t: float = 0.01, epsilon: float = 0.001):
        self.d1 = d1
        self.d2 = d2
        self.t = t
        self.epsilon = epsilon
        
        #Rescaling matrix A_t
        d = d1 + d2
        self.A_t = np.eye(d)
        self.A_t[:d1, :d1] *= 1.0
        self.A_t[d1:, d1:] *= np.sqrt(t)
        
        #Storage for training data
        self.X_train = None
        self.Y_train = None
        self.dual_potentials = None
    
    #Compute rescaled cost matrix C_ij = ½‖A_t(X_i - Y_j)‖²
    def compute_rescaled_cost(self, X, Y):
        X_scaled = X @ self.A_t
        Y_scaled = Y @ self.A_t
        
        C = cdist(X_scaled, Y_scaled, metric='sqeuclidean') / 2.0
        return C
    
    #Sinkhorn algorithm for entropic OT dual potentials
    #Solves: min_π <C,π> + ε·KL(π‖ρ⊗μ)
    #Returns: g ∈ ℝⁿ (dual potential on target)

    def sinkhorn_dual_potentials(self, C, max_iter=1000, tol=1e-6):
        n, m = C.shape
        
        f = np.zeros(n)
        g = np.zeros(m)
        
        for iteration in range(max_iter):
            f_old = f.copy()
            
            # Update f
            f = -self.epsilon * np.log(
                np.sum(np.exp((g - C.T) / self.epsilon), axis=1) + 1e-10
            )
            
            # Update g
            g = -self.epsilon * np.log(
                np.sum(np.exp((f[:, np.newaxis] - C) / self.epsilon), axis=0) + 1e-10
            )
            
            # Check convergence
            if np.max(np.abs(f - f_old)) < tol:
                break
        
        return g
    
    #Fit conditional Brenier map to joint samples
    #X1 : ndarray, shape (n, d1) - Past features
    #X2 : ndarray, shape (n, d2) - Future targets
    
    def fit(self, X1, X2):
        print(f"\n Fitting Conditional Brenier Map")
        print(f"d1={self.d1}, d2={self.d2}, t={self.t:.4f}, ε={self.epsilon:.4f}")
        
        n = X1.shape[0]
        
        #Store training data
        self.X_train = np.hstack([X1, X2])
        self.Y_train = np.hstack([X1, X2])
        
        #Compute rescaled cost
        C = self.compute_rescaled_cost(self.X_train, self.Y_train)
        
        #Solve Sinkhorn
        self.dual_potentials = self.sinkhorn_dual_potentials(C)
        
        print(f"Fitted on {n} samples")

    #Predict X₂ given X₁ = x1_query
    #T̂_ε,t(x) = Σᵢ Yᵢ · exp((gᵢ - ½‖A_t(x-Yᵢ)‖²)/ε) / normalization
    #x1_query : ndarray, shape (d1,) - Conditioning variables - If True, returns (mean, samples)
    #prediction : ndarray, shape (d2,) - Predicted target (conditional mean)
    def predict(self, x1_query, return_distribution=False):
        x1_query = np.atleast_2d(x1_query)
        
        if self.dual_potentials is None:
            raise ValueError("Model not fitted, call fit() first")
        
        #Extend query to full dimension (X₁ fixed, X₂ arbitrary)
        x_extended = np.zeros((x1_query.shape[0], self.d1 + self.d2))
        x_extended[:, :self.d1] = x1_query
        
        predictions = []
        
        for x in x_extended:
            #Compute distances in rescaled space
            x_scaled = self.A_t @ x
            Y_scaled = self.Y_train @ self.A_t.T
            
            distances = np.sum((Y_scaled - x_scaled) ** 2, axis=1) / 2.0
            
            #Compute barycentric weights
            log_weights = (self.dual_potentials - distances) / self.epsilon
            log_weights -= np.max(log_weights)  # Numerical stability
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            
            #Barycentric projection
            result = np.sum(self.Y_train * weights[:, np.newaxis], axis=0)
            predictions.append(result[self.d1:])  # Extract X₂ component
        
        prediction = np.array(predictions).squeeze()
        
        if return_distribution:
            #Sample from weighted distribution
            indices = np.random.choice(
                len(self.Y_train), 
                size=1000, 
                p=weights,
                replace=True
            )
            samples = self.Y_train[indices, self.d1:]
            return prediction, samples
        
        return prediction
    
    #Compute Wasserstein-2 distance between true and predicted conditionals.
    def compute_wasserstein_error(self, x1_test, x2_test):
        predictions = []
        for x1 in x1_test:
            pred = self.predict(x1)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        #W₂² ≈ MSE for Gaussians
        mse = np.mean((predictions - x2_test) ** 2)
        return np.sqrt(mse)

#Heston with Drift

class HestonWithDrift:  
    def __init__(self, S0=100.0, v0=0.04, kappa=2.0, theta=0.04, 
                 sigma=0.3, rho=-0.7, r=0.0):
        self.heston = HestonUnified(
            S0=S0, v0=v0, kappa=kappa,
            theta=theta, sigma=sigma, rho=rho, r=r
        )
        
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        
        self.drift_weights = None
    
    #Drift function using RBF basis
    def drift_lambda(self, t, S, v, weights):
        if weights is None or len(weights) == 0:
            return 0.0
        
        S_centers = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * self.S0
        v_centers = np.array([0.02, 0.04, 0.06])
        
        features = []
        for S_c in S_centers:
            for v_c in v_centers:
                rbf = np.exp(-0.5 * (((S - S_c)/20)**2 + ((v - v_c)/0.02)**2))
                features.append(rbf)
        
        return np.dot(weights, np.array(features))
    
    #Simulate paths with drift
    def simulate_paths(self, T, n_steps, n_paths=1, weights=None, 
                      return_full_path=True, seed=None):
        if weights is None:
            return self.heston.simulate_paths(T, n_steps, n_paths, seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        S = np.ones((n_paths, n_steps + 1)) * self.S0
        v = np.ones((n_paths, n_steps + 1)) * self.v0
        
        for i in range(n_steps):
            t = times[i]
            
            dW = np.random.randn(n_paths) * np.sqrt(dt)
            dZ_indep = np.random.randn(n_paths) * np.sqrt(dt)
            dZ = self.rho * dW + np.sqrt(1 - self.rho**2) * dZ_indep
            
            v_current = np.maximum(v[:, i], 1e-8)
            
            lambda_drift = np.array([
                self.drift_lambda(t, S[j, i], v_current[j], weights) 
                for j in range(n_paths)
            ])
            
            drift_v = self.kappa * (self.theta - v_current) + lambda_drift
            dv = drift_v * dt + self.sigma * np.sqrt(v_current) * dZ
            v[:, i + 1] = np.maximum(v_current + dv, 1e-8)
            
            dS = S[:, i] * np.sqrt(v_current) * dW
            S[:, i + 1] = S[:, i] + dS
        
        if return_full_path:
            return S, v, times
        else:
            return S[:, -1], v[:, -1], None
    
    #Price calls with drift
    def price_calls(self, strikes, T, n_paths=10000, weights=None):
        S_T, _, _ = self.simulate_paths(
            T, n_steps=100, n_paths=n_paths, 
            weights=weights, return_full_path=False
        )
        
        prices = []
        for K in strikes:
            payoff = np.maximum(S_T - K, 0)
            prices.append(np.mean(payoff))
        
        return np.array(prices)

#Black-Scholes Implied Volatility 

#Compute implied volatility via Newton-Raphson
def black_scholes_iv(S, K, T, price, r=0.0):
    if price <= max(S - K, 0):
        return 0.01
    
    sigma = 0.3
    
    for _ in range(100):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        bs_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        vega = S*norm.pdf(d1)*np.sqrt(T)
        
        if abs(price - bs_price) < 1e-6 or vega < 1e-10:
            break
        
        sigma += (price - bs_price) / vega
        sigma = max(0.01, min(2.0, sigma))
    
    return sigma

#Load unified data with:
#Volatility surface
#Price prediction features 

def load_unified_data(maturity_idx=3):
    
    print("Loading data")

    data_file = find_data_file('unified_heston_prediction_data.npz')
    if data_file is None:
        raise FileNotFoundError(
            "\n unified_heston_prediction_data.npz not found!\n"
            " Run: python generate_synthetic_data.py\n"
        )
    
    print(f"\n Loading from: {data_file.absolute()}")
    
    data = np.load(data_file, allow_pickle=True)
    
    #Volatility surface data
    strikes_norm = data['strikes_norm']
    maturities = data['maturities']
    vol_surface = data['vol_surface']
    heston_params = data['heston_params'].item()
    
    #Price prediction data
    X1_full = data['full_X1']
    X2_full = data['full_X2']
    X1_raw = data['full_X1_raw']
    lookback = int(data['full_lookback'])
    horizon = int(data['full_horizon'])
    
    print(f"Loaded successfully")
    print(f"\n Volatility Surface:")
    print(f"Strikes: {len(strikes_norm)} points")
    print(f"Maturities: {maturities}")
    
    print(f"\n  Price Prediction Data:")
    print(f"Samples: {len(X1_full)}")
    print(f"Features (X₁): {X1_full.shape[1]} (includes IV surface)")
    print(f"Targets (X₂): {X2_full.shape[1]} (return, vol, drawdown)")
    print(f"Lookback: {lookback} days, Horizon: {horizon} days")
    
    #Extract maturity for drift calibration
    S0 = float(heston_params['S0'])
    T = float(maturities[maturity_idx])
    market_vols = vol_surface[maturity_idx, :]
    strikes = strikes_norm * S0
    
    #Compute prices from IVs
    from scipy.stats import norm as scipy_norm
    market_prices = np.array([
        S0*scipy_norm.cdf((np.log(S0/K) + 0.5*iv**2*T)/(iv*np.sqrt(T))) - 
        K*scipy_norm.cdf((np.log(S0/K) + 0.5*iv**2*T)/(iv*np.sqrt(T)) - iv*np.sqrt(T))
        for K, iv in zip(strikes, market_vols)
    ])
    
    return {
        'strikes': strikes,
        'market_prices': market_prices,
        'market_vols': market_vols,
        'T': T,
        'heston_params': heston_params,
        'X1': X1_full,
        'X2': X2_full,
        'X1_raw': X1_raw,
        'lookback': lookback,
        'horizon': horizon
    }

#Particle Filter with conditional brenier

class HybridCalibration:
    def __init__(self, n_particles=200, n_weights=15, heston_params=None):
        self.n_particles = n_particles
        self.n_weights = n_weights
        
        if heston_params is None:
            heston_params = {
                'S0': 100.0, 'v0': 0.04,
                'kappa': 2.0, 'theta': 0.04,
                'sigma': 0.3, 'rho': -0.7
            }
        
        self.heston = HestonWithDrift(**heston_params)
        
        #Particle system
        self.particles = None
        self.weights = None
        self.calibrated_drift = None
        
        #Conditional Brenier estimator
        self.brenier_estimator = None
    
    def initialize_prior(self):
        self.particles = np.random.randn(self.n_particles, self.n_weights) * 0.1
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    #ESS = (Σw_i)² / Σw_i²
    def effective_sample_size(self):
        return (np.sum(self.weights)**2) / np.sum(self.weights**2)
    
    #Update weights based on likelihood
    def reweight(self, market_price, strike, T, noise_std=0.3):
        likelihoods = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            drift_weights = self.particles[i]
            model_price = self.heston.price_calls(
                [strike], T, n_paths=1000, weights=drift_weights
            )[0]
            likelihoods[i] = norm.pdf(market_price, loc=model_price, scale=noise_std)
        
        self.weights *= likelihoods
        self.weights /= (np.sum(self.weights) + 1e-10)
    
    #Systematic resampling when ESS low
    def resample(self, threshold=0.5):
        ess = self.effective_sample_size()
        
        if ess < threshold * self.n_particles:
            positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
            cumsum = np.cumsum(self.weights)
            indices = np.searchsorted(cumsum, positions)
            
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
            return True
        return False
    
    #MCMC move
    def move(self, step_size=0.01):
        noise = np.random.randn(self.n_particles, self.n_weights) * step_size
        self.particles += noise
    
    def calibrate_drift(self, strikes, market_prices, T):
        self.initialize_prior()
        
        for i, (K, C_mkt) in enumerate(zip(strikes, market_prices)):
            print(f"\n  Strike {i+1}/{len(strikes)}: K={K:.0f}, C={C_mkt:.4f}")
            
            self.reweight(C_mkt, K, T, noise_std=0.3)
            ess = self.effective_sample_size()
            print(f"ESS: {ess:.0f}/{self.n_particles}")
            
            if self.resample(threshold=0.5):
                self.move(step_size=0.02)
        
        self.calibrated_drift = np.average(self.particles, weights=self.weights, axis=0)
        
        print(f"\nDrift calibration complete")
        print(f"Weight range: [{self.calibrated_drift.min():.4f}, "
              f"{self.calibrated_drift.max():.4f}]")
        
        return self.calibrated_drift
    
    #Fit Conditional Brenier Map for price prediction.
    #X1: ndarray, shape (n, d_features)
    # Past features (vol, momentum, IV, etc.)
    #X2 : ndarray, shape (n, 3)
    #Future targets (return, vol, drawdown)
    #d1 : int - Number of conditioning features to use
   
    def fit_brenier_map(self, X1, X2, d1, t=0.01, epsilon=0.001):
    
        # Use subset of features as conditioning variables
        X1_subset = X1[:, -d1:]  # Last d1 features (vol, momentum, etc.)
        d2 = X2.shape[1]
        
        self.brenier_estimator = ConditionalBrenierEstimator(
            d1=d1, d2=d2, t=t, epsilon=epsilon
        )
        
        self.brenier_estimator.fit(X1_subset, X2)
        
        print(f"Brenier map fitted")
        print(f"Input dim (d₁): {d1}")
        print(f"Output dim (d₂): {d2}")
        
        return self.brenier_estimator
    
    #Predict future outcomes using Conditional Brenier Map
    def predict_future_prices(self, current_features, n_samples=1000):
        if self.brenier_estimator is None:
            raise ValueError("Brenier map not fitted, Call fit_brenier_map() first.")
        
        # Predict conditional mean
        pred_mean, samples = self.brenier_estimator.predict(
            current_features, return_distribution=True
        )
        
        return {
            'mean_return': pred_mean[0],
            'mean_vol': pred_mean[1],
            'mean_drawdown': pred_mean[2],
            'samples': samples
        }

#main

if __name__ == "__main__":
    np.random.seed(42)
 
    #Load unified data
    try:
        data = load_unified_data(maturity_idx=3)
    except FileNotFoundError as e:
        print(str(e))
        print("\n💡 Run: python generate_synthetic_data.py")
        exit(1)
    
    #Extract data
    strikes = data['strikes']
    market_prices = data['market_prices']
    market_vols = data['market_vols']
    T = data['T']
    heston_params = data['heston_params']
    X1 = data['X1']
    X2 = data['X2']
    X1_raw = data['X1_raw']
    
   
    print("summary")
    print(f"Market Data:")
    print(f"Strikes: {len(strikes)} options at T={T:.2f}Y")
    print(f"Price range: [{market_prices.min():.4f}, {market_prices.max():.4f}]")
    print(f"\n Prediction Data:")
    print(f"Samples: {len(X1)}")
    print(f"Features: {X1.shape[1]}")
    print(f"Targets: {X2.shape[1]} (return, vol, drawdown)")
    
    #Initialize hybrid calibration
    hybrid = HybridCalibration(
        n_particles=200,
        n_weights=15,
        heston_params=heston_params
    )
    
    #Calibrate drift
    start_time = time.time()
    calibrated_drift = hybrid.calibrate_drift(strikes, market_prices, T)
    drift_time = time.time() - start_time
    
    #Fit Brenier map
    start_time = time.time()
    d1 = 7  #Use last 7 features - realized vol, current vol, momentum, etc
    brenier = hybrid.fit_brenier_map(X1, X2, d1=d1, t=0.02, epsilon=0.004)
    brenier_time = time.time() - start_time
    
    print("Calibration Times")
    print(f"Drift calibration: {drift_time:.1f}s")
    print(f"Brenier map fitting: {brenier_time:.1f}s")
    print(f"Total: {drift_time + brenier_time:.1f}s")
    
    #Drift calibration accuracy
    
    model_prices = hybrid.heston.price_calls(strikes, T, n_paths=20000, 
                                            weights=calibrated_drift)
    
    errors = np.abs(model_prices - market_prices)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    #Price prediction accuracy
    
    # Test on subset
    n_test = min(500, len(X1))
    test_indices = np.random.choice(len(X1), n_test, replace=False)
    
    X1_test = X1[test_indices, -d1:]
    X2_test = X2[test_indices]
    
    predictions = []
    for x1 in X1_test:
        pred = brenier.predict(x1)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    #Compute metrics
    mse_return = np.mean((predictions[:, 0] - X2_test[:, 0])**2)
    mse_vol = np.mean((predictions[:, 1] - X2_test[:, 1])**2)
    mse_dd = np.mean((predictions[:, 2] - X2_test[:, 2])**2)
    
    print(f"\n Prediction MSE:")
    print(f"Return: {mse_return:.6f}")
    print(f"Volatility: {mse_vol:.6f}")
    print(f"Drawdown: {mse_dd:.6f}")
    
    #Wasserstein distance
    w2_error = brenier.compute_wasserstein_error(X1_test, X2_test)
    print(f"\n Wasserstein-2 Error: {w2_error:.6f}")
    
    #Select 3 scenarios: low vol, medium vol, high vol
    vol_feature_idx = data['lookback']  #Position of realized vol in raw features
    realized_vols = X1_raw[:, vol_feature_idx]
    
    low_vol_idx = np.argmin(realized_vols)
    high_vol_idx = np.argmax(realized_vols)
    med_vol_idx = np.argsort(realized_vols)[len(realized_vols)//2]
    
    scenarios = [
        ('Low Volatility', low_vol_idx, realized_vols[low_vol_idx]),
        ('Medium Volatility', med_vol_idx, realized_vols[med_vol_idx]),
        ('High Volatility', high_vol_idx, realized_vols[high_vol_idx])
    ]
    
    for scenario_name, idx, vol_value in scenarios:
        print(f"\n  {scenario_name} (σ={vol_value:.4f}):")
        
        features = X1[idx, -d1:]
        prediction = hybrid.predict_future_prices(features, n_samples=1000)
        
        true_values = X2[idx]
        
        print(f"Predicted Return:   {prediction['mean_return']:+.4f} "
              f"(true: {true_values[0]:+.4f})")
        print(f"Predicted Vol:      {prediction['mean_vol']:.4f} "
              f"(true: {true_values[1]:.4f})")
        print(f"Predicted Drawdown: {prediction['mean_drawdown']:.4f} "
              f"(true: {true_values[2]:.4f})")
    
    print("Generating visualizations")
    
    fig = plt.figure(figsize=(18, 12))
    
    #Plot 1: Price comparison
    ax1 = plt.subplot(3, 4, 1)
    x_pos = np.arange(len(strikes))
    width = 0.35
    ax1.bar(x_pos - width/2, market_prices, width, label='Market', 
            alpha=0.8, color='blue')
    ax1.bar(x_pos + width/2, model_prices, width, label='Calibrated Model', 
            alpha=0.8, color='orange')
    ax1.set_xlabel('Strike Index', fontsize=10)
    ax1.set_ylabel('Call Price', fontsize=10)
    ax1.set_title('Drift Calibration: Price Fit', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    #Plot 2: Pricing errors
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(strikes, errors, 'o-', linewidth=2, markersize=8, color='red')
    ax2.fill_between(strikes, 0, errors, alpha=0.3, color='red')
    ax2.set_xlabel('Strike', fontsize=10)
    ax2.set_ylabel('Absolute Error', fontsize=10)
    ax2.set_title(f'Calibration Errors (MAE={mae:.4f})', 
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    #Plot 3: IV comparison
    ax3 = plt.subplot(3, 4, 3)
    model_vols = np.array([
        black_scholes_iv(heston_params['S0'], K, T, price)
        for K, price in zip(strikes, model_prices)
    ])
    ax3.plot(strikes, market_vols*100, 'o-', linewidth=2, markersize=8,
             label='Market', color='blue')
    ax3.plot(strikes, model_vols*100, 's-', linewidth=2, markersize=8,
             label='Calibrated', color='orange', alpha=0.7)
    ax3.set_xlabel('Strike', fontsize=10)
    ax3.set_ylabel('Implied Volatility (%)', fontsize=10)
    ax3.set_title('Volatility Smile', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    #Plot 4: Learned drift weights
    ax4 = plt.subplot(3, 4, 4)
    ax4.bar(range(len(calibrated_drift)), calibrated_drift, 
            alpha=0.7, color='green')
    ax4.set_xlabel('Basis Function', fontsize=10)
    ax4.set_ylabel('Weight', fontsize=10)
    ax4.set_title('Learned Drift λ(t,S,v)', fontsize=11, fontweight='bold')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    #Plot 5: Return prediction scatter
    ax5 = plt.subplot(3, 4, 5)
    ax5.scatter(X2_test[:, 0], predictions[:, 0], alpha=0.4, s=15)
    lim_min = min(X2_test[:, 0].min(), predictions[:, 0].min())
    lim_max = max(X2_test[:, 0].max(), predictions[:, 0].max())
    ax5.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
             linewidth=2, alpha=0.7, label='Perfect')
    ax5.set_xlabel('True Return', fontsize=10)
    ax5.set_ylabel('Predicted Return', fontsize=10)
    ax5.set_title(f'Return Prediction (MSE={mse_return:.4f})', 
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    #Plot 6: Volatility prediction scatter
    ax6 = plt.subplot(3, 4, 6)
    ax6.scatter(X2_test[:, 1], predictions[:, 1], alpha=0.4, s=15, color='green')
    lim_min = min(X2_test[:, 1].min(), predictions[:, 1].min())
    lim_max = max(X2_test[:, 1].max(), predictions[:, 1].max())
    ax6.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
             linewidth=2, alpha=0.7)
    ax6.set_xlabel('True Volatility', fontsize=10)
    ax6.set_ylabel('Predicted Volatility', fontsize=10)
    ax6.set_title(f'Vol Prediction (MSE={mse_vol:.4f})', 
                  fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    #Plot 7: Drawdown prediction scatter
    ax7 = plt.subplot(3, 4, 7)
    ax7.scatter(X2_test[:, 2], predictions[:, 2], alpha=0.4, s=15, color='purple')
    lim_min = min(X2_test[:, 2].min(), predictions[:, 2].min())
    lim_max = max(X2_test[:, 2].max(), predictions[:, 2].max())
    ax7.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
             linewidth=2, alpha=0.7)
    ax7.set_xlabel('True Max Drawdown', fontsize=10)
    ax7.set_ylabel('Predicted Max Drawdown', fontsize=10)
    ax7.set_title(f'Drawdown Prediction (MSE={mse_dd:.4f})', 
                  fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    #Plot 8: Prediction errors histogram
    ax8 = plt.subplot(3, 4, 8)
    return_errors = predictions[:, 0] - X2_test[:, 0]
    ax8.hist(return_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2)
    ax8.set_xlabel('Prediction Error', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.set_title('Return Prediction Error Distribution', 
                  fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    for plot_idx, (scenario_name, idx, vol_value) in enumerate(scenarios):
        ax = plt.subplot(3, 4, 9 + plot_idx)
        
        features = X1[idx, -d1:]
        prediction = hybrid.predict_future_prices(features, n_samples=1000)
        samples = prediction['samples']
        
        #Plot conditional distribution of returns
        ax.hist(samples[:, 0], bins=30, alpha=0.6, color='skyblue', 
                edgecolor='black', density=True)
        
        #True value
        true_return = X2[idx, 0]
        ax.axvline(true_return, color='red', linestyle='--', 
                   linewidth=2, label=f'True: {true_return:.3f}')
        
        #Predicted mean
        pred_mean = prediction['mean_return']
        ax.axvline(pred_mean, color='green', linestyle='-', 
                   linewidth=2, label=f'Pred: {pred_mean:.3f}')
        
        ax.set_xlabel('Future Return', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{scenario_name}\nσ={vol_value:.3f}', 
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    #Plot 12: Summary statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = f"""
  Particles: {hybrid.n_particles}
  Basis functions: {hybrid.n_weights}
  MAE: {mae:.4f}
  RMSE: {rmse:.4f}
  Time: {drift_time:.1f}s

Brenier map:
  Input dim (d₁): {d1}
  Output dim (d₂): {X2.shape[1]}
  Rescaling t: {brenier.t:.4f}
  Entropy ε: {brenier.epsilon:.4f}
  Time: {brenier_time:.1f}s

Prediction Accuracy:
  Return MSE: {mse_return:.6f}
  Vol MSE: {mse_vol:.6f}
  Drawdown MSE: {mse_dd:.6f}
  W₂ Error: {w2_error:.6f}

Data:
  Training samples: {len(X1)}
  Test samples: {n_test}
  Lookback: {data['lookback']} days
  Horizon: {data['horizon']} days
    """
    
    ax12.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
              verticalalignment='center')
    
    plt.suptitle('Hybrid Method: Particle Filter + Conditional Brenier Maps', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = 'hybrid_pf_brenier_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file}")
    
    #Conditional distributions comparison
    fig2 = plt.figure(figsize=(15, 5))
    
    for plot_idx, (scenario_name, idx, vol_value) in enumerate(scenarios):
        ax = plt.subplot(1, 3, plot_idx + 1)
        
        features = X1[idx, -d1:]
        prediction = hybrid.predict_future_prices(features, n_samples=2000)
        samples = prediction['samples']
        
        #return vs future vol
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, 
                   c=samples[:, 2], cmap='viridis')
        
        #True point
        true_point = X2[idx]
        ax.plot(true_point[0], true_point[1], 'r*', markersize=20,
                markeredgecolor='white', markeredgewidth=2, label='True')
        
        #Predicted mean
        ax.plot(prediction['mean_return'], prediction['mean_vol'], 
                'g*', markersize=20, markeredgecolor='white', 
                markeredgewidth=2, label='Predicted')
        
        ax.set_xlabel('Future Return', fontsize=11)
        ax.set_ylabel('Future Volatility', fontsize=11)
        ax.set_title(f'{scenario_name}\nCurrent σ={vol_value:.3f}', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Conditional Distributions μ(return, vol | past features)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file2 = 'conditional_distributions_2d.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file2}")
    
    #Convergence Analysis

    #Test with different sample sizes
    sample_sizes = [500, 1000, 2000, 3000, len(X1)]
    t_values = [0.1 * (n ** (-1/3)) for n in sample_sizes]
    epsilon_values = [t**2 for t in t_values]
    
    print(f"\n Testing convergence with varying sample sizes")
    
    errors_by_size = []
    
    for n, t_param, eps_param in zip(sample_sizes, t_values, epsilon_values):
        print(f"\n    n={n}, t={t_param:.4f}, ε={eps_param:.6f}")
        
        #Subsample data
        indices = np.random.choice(len(X1), min(n, len(X1)), replace=False)
        X1_sub = X1[indices, -d1:]
        X2_sub = X2[indices]
        
        #Fit Brenier with these parameters
        brenier_temp = ConditionalBrenierEstimator(d1=d1, d2=3, 
                                                    t=t_param, epsilon=eps_param)
        brenier_temp.fit(X1_sub, X2_sub)
        
        #Test error
        test_indices = np.random.choice(len(X1), 200, replace=False)
        X1_test_conv = X1[test_indices, -d1:]
        X2_test_conv = X2[test_indices]
        
        w2_error_temp = brenier_temp.compute_wasserstein_error(
            X1_test_conv, X2_test_conv
        )
        errors_by_size.append(w2_error_temp)
        
        print(f"W₂ error: {w2_error_temp:.6f}")
    
    #Plot convergence rate
    fig3 = plt.figure(figsize=(10, 6))
    
    ax = plt.subplot(111)
    ax.loglog(sample_sizes, errors_by_size, 'o-', linewidth=2, 
              markersize=10, label='Observed Error')
    
    #Theoretical rate: n^(-2/3)
    theoretical = errors_by_size[0] * (np.array(sample_sizes) / sample_sizes[0])**(-2/3)
    ax.loglog(sample_sizes, theoretical, '--', linewidth=2, 
              label='Theory: O(n^(-2/3))', color='red', alpha=0.7)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Wasserstein-2 Error', fontsize=12)
    ax.set_title('Convergence Rate Analysis (Baptista et al. Theorem 4.2)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_file3 = 'convergence_analysis.png'
    plt.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file3}")
    
    #summary
    
    print("summary")
    
    print(f"\n Drift Calibration - Martingale Schrödinger Bridge")
    print(f"Method: Particle Filter with {hybrid.n_particles} particles")
    print(f"Calibrated {hybrid.n_weights} drift basis functions")
    print(f"Price fitting MAE: {mae:.4f}")
    print(f"Vol-of-vol σ={heston_params['sigma']} unchanged")
    
    print(f"\n Price Prediction - Conditional Brenier Maps")
    print(f"Method: Entropic Optimal Transport")
    print(f"Rescaling parameter: t={brenier.t:.4f}")
    print(f"Entropic regularization: ε={brenier.epsilon:.6f}")
    print(f"Return prediction MSE: {mse_return:.6f}")
    print(f"Volatility prediction MSE: {mse_vol:.6f}")
    print(f"Wasserstein-2 error: {w2_error:.6f}")
    
    print(f"\n Computational Efficiency:")
    print(f"Drift calibration: {drift_time:.1f}s")
    print(f"Brenier fitting: {brenier_time:.1f}s")
    print(f"Total time: {drift_time + brenier_time:.1f}s")
    print(f"Training samples: {len(X1)}")
    
    print(f"\n Output Files")
    print(f"{output_file}")
    print(f"{output_file2}")
    print(f"{output_file3}")
    
    print("Experiment Complete")
    
    plt.show()
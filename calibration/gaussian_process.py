import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path
import sys

# HestonUnified lives in base.py (same calibration folder)
sys.path.append(str(Path(__file__).parent))
from base import HestonUnified

# ConditionalBrenierEstimator lives in models/brenier.py
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from brenier import ConditionalBrenierEstimator

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
    
    print(f"File not found")
    return None


#Black-scholes implied volatility

#Compute BS call price
def black_scholes_price(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

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


def load_unified_data(maturity_idx=3):
    data_file = find_data_file('unified_heston_prediction_data.npz')
    if data_file is None:
        raise FileNotFoundError(
            "\n unified_heston_prediction_data.npz not found!\n"
            "   Run: python generate_synthetic_data.py\n"
        )
    
    print(f"\n Loading from: {data_file.absolute()}")
    
    data = np.load(data_file, allow_pickle=True)
    
    strikes_norm = data['strikes_norm']
    maturities = data['maturities']
    vol_surface = data['vol_surface']
    heston_params = data['heston_params'].item()
    
    X1_full = data['full_X1']
    X2_full = data['full_X2']
    X1_raw = data['full_X1_raw']
    lookback = int(data['full_lookback'])
    horizon = int(data['full_horizon'])
    
    print(f"Loaded successfully")
    print(f"\n Volatility Surface:")
    print(f"Strikes: {len(strikes_norm)} points")
    print(f"Maturities: {maturities}")
    
    print(f"\n Price Prediction Data:")
    print(f"Samples: {len(X1_full)}")
    print(f"Features (X₁): {X1_full.shape[1]}")
    print(f"Targets (X₂): {X2_full.shape[1]} (return, vol, drawdown)")
    
    S0 = float(heston_params['S0'])
    T = float(maturities[maturity_idx])
    market_vols = vol_surface[maturity_idx, :]
    strikes = strikes_norm * S0
    
    market_prices = np.array([
        S0*norm.cdf((np.log(S0/K) + 0.5*iv**2*T)/(iv*np.sqrt(T))) - 
        K*norm.cdf((np.log(S0/K) + 0.5*iv**2*T)/(iv*np.sqrt(T)) - iv*np.sqrt(T))
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

#Wrapper around HestonModel with drift for GP calibration
class HestonModelWithDrift:
    def __init__(self, heston_model, n_weights=15):
        self.heston = heston_model
        self.n_weights = n_weights
        # Pre-build RBF centers once (avoids recomputing per step)
        S_centers = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * self.heston.S0
        v_centers = np.array([0.02, 0.04, 0.06])
        self._centers = np.array(
            [[Sc, vc] for Sc in S_centers for vc in v_centers]
        )  # (15, 2)
        self._scales = np.array([20.0, 0.02])

    def _drift_vectorized(self, S_arr, v_arr, weights):
        """Vectorized RBF drift over all paths at once — shape (n_paths,)."""
        sv   = np.stack([S_arr, v_arr], axis=1)               # (n, 2)
        diff = (sv[:, None, :] - self._centers[None, :, :]) / self._scales  # (n, 15, 2)
        rbf  = np.exp(-0.5 * np.sum(diff ** 2, axis=2))       # (n, 15)
        return rbf @ weights                                    # (n,)

    #Simulate paths with drift on variance
    def simulate_paths_with_drift(self, T, n_steps, n_paths, weights=None):
        dt          = T / n_steps
        sqrt_1mrho2 = np.sqrt(1.0 - self.heston.rho ** 2)
        t           = np.linspace(0, T, n_steps + 1)

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.heston.S0
        v[:, 0] = self.heston.v0

        for i in range(n_steps):
            Z1 = np.random.randn(n_paths)
            Z2 = np.random.randn(n_paths)
            W1 = Z1
            W2 = self.heston.rho * Z1 + sqrt_1mrho2 * Z2

            v_curr = np.maximum(v[:, i], 1e-8)

            lambda_drift = (
                self._drift_vectorized(S[:, i], v_curr, weights)
                if weights is not None else 0.0
            )

            drift_v = self.heston.kappa * (self.heston.theta - v_curr) + lambda_drift
            dv = drift_v * dt + self.heston.sigma * np.sqrt(v_curr * dt) * W2
            v[:, i + 1] = np.maximum(v_curr + dv, 1e-8)

            S[:, i + 1] = S[:, i] * np.exp(
                (self.heston.r - 0.5 * v_curr) * dt + np.sqrt(v_curr * dt) * W1
            )

        return S, v, t

    #Price calls with drift — all strikes in one broadcast
    def price_calls(self, strikes, T, n_paths=10000, weights=None):
        S, _, _ = self.simulate_paths_with_drift(
            T, n_steps=100, n_paths=n_paths, weights=weights
        )
        S_T     = S[:, -1]
        strikes = np.asarray(strikes)
        payoffs = np.maximum(S_T[:, None] - strikes[None, :], 0)  # (n_paths, n_strikes)
        return np.exp(-self.heston.r * T) * payoffs.mean(axis=0)



#GP-based Bayesian Optimization for drift learning
class GaussianProcessDriftCalibration:
    def __init__(self, heston_model: HestonUnified, n_weights: int = 15,
                 n_initial: int = 30, n_iterations: int = 70):
        self.n_weights = n_weights
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        
        self.heston_drift = HestonModelWithDrift(heston_model, n_weights)
        
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-4,
            n_restarts_optimizer=10, normalize_y=True
        )
        
        self.X_samples = []
        self.y_samples = []
        self.bounds = np.array([[-1.0, 1.0]] * n_weights)
    
    #Normalize to [0, 1]
    def normalize_weights(self, weights):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return (weights - lower) / (upper - lower)
    
    #Denormalize from [0, 1]
    def denormalize_weights(self, weights_norm):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return lower + weights_norm * (upper - lower)
    
    #Latin Hypercube sampling for good coverage
    def sample_prior(self, n_samples):
        sampler = LatinHypercube(d=self.n_weights, seed=42)
        samples_norm = sampler.random(n=n_samples)
        return self.denormalize_weights(samples_norm)
    
    #MSE loss
    def objective(self, weights, strikes, market_prices, T):
        try:
            model_prices = self.heston_drift.price_calls(
                strikes, T, n_paths=5000, weights=weights
            )
            loss = np.mean((model_prices - market_prices)**2)
        except:
            loss = 1e6
        return loss
    
    #Expected Improvement acquisition function
    def expected_improvement(self, weights_norm, y_best):
        mu, sigma = self.gp.predict(weights_norm.reshape(1, -1), return_std=True)
        mu    = mu[0]
        sigma = sigma[0]
        
        if sigma < 1e-8:
            return 0.0
        
        z = (y_best - mu) / sigma
        ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
    
    #Select next weights via EI maximization
    def select_next_sample(self, y_best):
        best_ei = -np.inf
        best_weights = None
        
        for _ in range(10):
            x0 = np.random.uniform(0, 1, self.n_weights)
            
            result = minimize(
                lambda x: -self.expected_improvement(x, y_best),
                x0, bounds=[(0, 1)] * self.n_weights,
                method='L-BFGS-B'
            )
            
            if -result.fun > best_ei:
                best_ei = -result.fun
                best_weights = result.x
        
        return self.denormalize_weights(best_weights)
    
    #Calibrate drift via Bayesian Optimization
    def calibrate(self, strikes, market_prices, T, verbose=True):
        if verbose:
            print(f"Basis functions: {self.n_weights}")
            print(f"Initial samples: {self.n_initial}")
            print(f"BO iterations: {self.n_iterations}")
        
        #Initial sampling
        if verbose:
            print(f"\n Phase 1: Initial sampling")
        
        initial_samples = self.sample_prior(self.n_initial)
        
        for weights in tqdm(initial_samples, desc="Evaluating", disable=not verbose):
            loss = self.objective(weights, strikes, market_prices, T)
            self.X_samples.append(self.normalize_weights(weights))
            self.y_samples.append(loss)
        
        X_train = np.array(self.X_samples)
        y_train = np.array(self.y_samples)
        
        if verbose:
            print(f"Best initial loss: {np.min(y_train):.6f}")
            print(f"\n Phase 2: Bayesian optimization")
        
        #Phase 2: BO
        for iteration in tqdm(range(self.n_iterations), desc="Optimizing", disable=not verbose):
            self.gp.fit(X_train, y_train)
            
            y_best = np.min(y_train)
            next_weights = self.select_next_sample(y_best)
            next_loss = self.objective(next_weights, strikes, market_prices, T)
            
            X_train = np.vstack([X_train, self.normalize_weights(next_weights)])
            y_train = np.append(y_train, next_loss)
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: best loss = {np.min(y_train):.6f}")
        
        best_idx = np.argmin(y_train)
        best_weights = self.denormalize_weights(X_train[best_idx])
        best_loss = y_train[best_idx]
        
        if verbose:
            print(f"\n Best Loss: {best_loss:.6f}")
            print(f"Total evaluations: {len(y_train)}")
        
        return best_weights, {'X': X_train, 'y': y_train, 'best_idx': best_idx}
    
    #Reprice with calibrated drift
    def reprice(self, best_weights, strikes, T):
        return self.heston_drift.price_calls(strikes, T, n_paths=20000, 
                                            weights=best_weights)


class HybridGPBrenierMethod:
    def __init__(self, heston_model: HestonUnified, n_weights=15):
        self.gp_calibrator = GaussianProcessDriftCalibration(
            heston_model, n_weights=n_weights, n_initial=30, n_iterations=70
        )
        self.brenier_estimator = None
        self.heston_model = heston_model
    
    def calibrate_gp_drift(self, strikes, market_prices, T):
        start = time.time()
        best_weights, history = self.gp_calibrator.calibrate(
            strikes, market_prices, T, verbose=True
        )
        gp_time = time.time() - start
        
        print(f"GP calibration complete ({gp_time:.1f}s)")
        
        return best_weights, history, gp_time
    
    def fit_brenier_map(self, X1, X2, d1, t=None, epsilon=None):
        """Fit the conditional Brenier map.

        t and epsilon default to the theory-optimal values:
            t*       = 0.1 * n^(-1/3)
            epsilon* = t*^2
        which shrink the bandwidth as n grows and avoids over-smoothing
        (the main cause of bias in the medium-volatility regime).
        """
        X1_subset = X1[:, -d1:]
        d2 = X2.shape[1]
        n  = X1_subset.shape[0]

        if t is None:
            t = 0.1 * (n ** (-1/3))
        if epsilon is None:
            epsilon = t ** 2

        print(f"  Brenier hyperparams: n={n}  t={t:.4f}  epsilon={epsilon:.6f}")

        start = time.time()
        self.brenier_estimator = ConditionalBrenierEstimator(
            d1=d1, d2=d2, t=t, epsilon=epsilon
        )
        self.brenier_estimator.fit(X1_subset, X2)
        brenier_time = time.time() - start

        print(f"Brenier map fitted ({brenier_time:.1f}s)")

        return self.brenier_estimator, brenier_time
    
    #Predict future outcomes
    def predict_future_prices(self, current_features, n_samples=1000):
        if self.brenier_estimator is None:
            raise ValueError("Brenier map not fitted")
        
        pred_mean, samples = self.brenier_estimator.predict(
            current_features, return_distribution=True, n_samples=n_samples
        )
        
        return {
            'mean_return': pred_mean[0],
            'mean_vol': pred_mean[1],
            'mean_drawdown': pred_mean[2],
            'samples': samples
        }

#main experiment
if __name__ == "__main__":
    np.random.seed(42)

    # Load data
    try:
        data = load_unified_data(maturity_idx=3)
    except FileNotFoundError as e:
        print(str(e))
        print("\n Run: python generate_synthetic_data.py")
        sys.exit(1)
    
    strikes = data['strikes']
    market_prices = data['market_prices']
    market_vols = data['market_vols']
    T = data['T']
    heston_params = data['heston_params']
    X1 = data['X1']
    X2 = data['X2']
    X1_raw = data['X1_raw']
    
    print("summary")
    print(f"Market Data: {len(strikes)} options at T={T:.2f}Y")
    print(f"Prediction Data: {len(X1)} samples, {X1.shape[1]} features")
    
    #Initialize hybrid
    heston_model = HestonUnified(**heston_params)
    hybrid = HybridGPBrenierMethod(heston_model, n_weights=15)
    
    #GP calibration
    best_weights, history, gp_time = hybrid.calibrate_gp_drift(
        strikes, market_prices, T
    )
    
    #Brenier map — d1=10 for richer conditioning; t/epsilon auto-tuned by theory
    d1 = 10
    brenier, brenier_time = hybrid.fit_brenier_map(X1, X2, d1=d1)
    
    print(f"GP calibration: {gp_time:.1f}s")
    print(f"Brenier fitting: {brenier_time:.1f}s")
    print(f"Total: {gp_time + brenier_time:.1f}s")
    
    #Validation 1: GP drift
    print("Validation 1: GP drift")
    
    model_prices = hybrid.gp_calibrator.reprice(best_weights, strikes, T)
    errors = np.abs(model_prices - market_prices)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
  
    print("#Validation 2: Price prediction")
    
    n_test = min(500, len(X1))
    test_indices = np.random.choice(len(X1), n_test, replace=False)
    
    X1_test = X1[test_indices, -d1:]
    X2_test = X2[test_indices]
    
    predictions = []
    for x1 in X1_test:
        pred = brenier.predict(x1)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    mse_return = np.mean((predictions[:, 0] - X2_test[:, 0])**2)
    mse_vol = np.mean((predictions[:, 1] - X2_test[:, 1])**2)
    mse_dd = np.mean((predictions[:, 2] - X2_test[:, 2])**2)
    
    print(f"\n Prediction MSE:")
    print(f"Return: {mse_return:.6f}")
    print(f"Volatility: {mse_vol:.6f}")
    print(f"Drawdown: {mse_dd:.6f}")
    
    w2_error = brenier.compute_wasserstein_error(X1_test, X2_test)
    print(f"\n Wasserstein-2 Error: {w2_error:.6f}")
    
    #Scenarios
    print("Conditional Predictions")
    
    vol_feature_idx = data['lookback']
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
        
        print(f"    Predicted Return:   {prediction['mean_return']:+.4f} "
              f"(true: {true_values[0]:+.4f})")
        print(f"    Predicted Vol:      {prediction['mean_vol']:.4f} "
              f"(true: {true_values[1]:.4f})")
        print(f"    Predicted Drawdown: {prediction['mean_drawdown']:.4f} "
              f"(true: {true_values[2]:.4f})")
    
    #Visualizations
    print("Generating Visualizations")
    
    fig = plt.figure(figsize=(18, 12))
    
    #Plot 1: GP convergence
    ax1 = plt.subplot(3, 4, 1)
    iterations = np.arange(len(history['y']))
    min_loss = np.minimum.accumulate(history['y'])
    ax1.semilogy(iterations, history['y'], 'o', alpha=0.3, markersize=4, 
                 label='Samples', color='gray')
    ax1.semilogy(iterations, min_loss, 'r-', linewidth=2, label='Best so far')
    ax1.axvline(hybrid.gp_calibrator.n_initial, color='blue', linestyle='--', 
                alpha=0.5, label='BO starts')
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Loss (MSE)', fontsize=10)
    ax1.set_title('GP Bayesian Optimization\nConvergence', 
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    #Plot 2: Price comparison
    ax2 = plt.subplot(3, 4, 2)
    x_pos = np.arange(len(strikes))
    width = 0.35
    ax2.bar(x_pos - width/2, market_prices, width, label='Market', 
            alpha=0.8, color='blue')
    ax2.bar(x_pos + width/2, model_prices, width, label='GP Model', 
            alpha=0.8, color='orange')
    ax2.set_xlabel('Strike Index', fontsize=10)
    ax2.set_ylabel('Call Price', fontsize=10)
    ax2.set_title('GP Calibration: Price Fit', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    #Plot 3: Learned weights
    ax3 = plt.subplot(3, 4, 3)
    ax3.bar(range(len(best_weights)), best_weights, alpha=0.7, color='green')
    ax3.set_xlabel('Basis Function', fontsize=10)
    ax3.set_ylabel('Weight', fontsize=10)
    ax3.set_title('Learned Drift Weights', fontsize=11, fontweight='bold')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    #Plot 4: IV comparison
    ax4 = plt.subplot(3, 4, 4)
    model_vols = np.array([
        black_scholes_iv(heston_params['S0'], K, T, price)
        for K, price in zip(strikes, model_prices)
    ])
    strikes_norm = strikes / heston_params['S0']
    ax4.plot(strikes_norm, market_vols*100, 'o-', linewidth=2.5, 
             markersize=8, color='black', label='Market')
    ax4.plot(strikes_norm, model_vols*100, 's--', linewidth=2, 
             markersize=6, alpha=0.7, color='orange', label='GP Model')
    ax4.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Strike / Spot', fontsize=10)
    ax4.set_ylabel('IV (%)', fontsize=10)
    ax4.set_title('Volatility Smile', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    #Plot 5: Return prediction
    ax5 = plt.subplot(3, 4, 5)
    ax5.scatter(X2_test[:, 0], predictions[:, 0], alpha=0.4, s=15)
    lim_min = min(X2_test[:, 0].min(), predictions[:, 0].min())
    lim_max = max(X2_test[:, 0].max(), predictions[:, 0].max())
    ax5.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
             linewidth=2, alpha=0.7, label='Perfect')
    ax5.set_xlabel('True Return', fontsize=10)
    ax5.set_ylabel('Predicted Return', fontsize=10)
    ax5.set_title(f'Return Prediction\n(MSE={mse_return:.4f})', 
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    #Plot 6: Volatility prediction
    ax6 = plt.subplot(3, 4, 6)
    ax6.scatter(X2_test[:, 1], predictions[:, 1], alpha=0.4, s=15, color='green')
    lim_min = min(X2_test[:, 1].min(), predictions[:, 1].min())
    lim_max = max(X2_test[:, 1].max(), predictions[:, 1].max())
    ax6.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
             linewidth=2, alpha=0.7)
    ax6.set_xlabel('True Volatility', fontsize=10)
    ax6.set_ylabel('Predicted Volatility', fontsize=10)
    ax6.set_title(f'Vol Prediction\n(MSE={mse_vol:.4f})', 
                  fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    #Plot 7: Drawdown prediction
    ax7 = plt.subplot(3, 4, 7)
    ax7.scatter(X2_test[:, 2], predictions[:, 2], alpha=0.4, s=15, color='purple')
    lim_min = min(X2_test[:, 2].min(), predictions[:, 2].min())
    lim_max = max(X2_test[:, 2].max(), predictions[:, 2].max())
    ax7.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
             linewidth=2, alpha=0.7)
    ax7.set_xlabel('True Drawdown', fontsize=10)
    ax7.set_ylabel('Predicted Drawdown', fontsize=10)
    ax7.set_title(f'DD Prediction\n(MSE={mse_dd:.4f})', 
                  fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    #Plot 8: Prediction errors
    ax8 = plt.subplot(3, 4, 8)
    return_errors = predictions[:, 0] - X2_test[:, 0]
    ax8.hist(return_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2)
    ax8.set_xlabel('Prediction Error', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.set_title('Return Error Distribution', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    for plot_idx, (scenario_name, idx, vol_value) in enumerate(scenarios):
        ax = plt.subplot(3, 4, 9 + plot_idx)
        
        features = X1[idx, -d1:]
        prediction = hybrid.predict_future_prices(features, n_samples=1000)
        samples = prediction['samples']
        
        ax.hist(samples[:, 0], bins=30, alpha=0.6, color='skyblue', 
                edgecolor='black', density=True)
        
        true_return = X2[idx, 0]
        ax.axvline(true_return, color='red', linestyle='--', 
                   linewidth=2, label=f'True: {true_return:.3f}')
        
        pred_mean = prediction['mean_return']
        ax.axvline(pred_mean, color='green', linestyle='-', 
                   linewidth=2, label=f'Pred: {pred_mean:.3f}')
        
        ax.set_xlabel('Future Return', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{scenario_name}\nσ={vol_value:.3f}', 
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    #Plot 12: Summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = f"""
GP Calibration
Basis functions: {len(best_weights)}
Initial samples: {hybrid.gp_calibrator.n_initial}
BO iterations: {hybrid.gp_calibrator.n_iterations}
MAE: {mae:.4f}
RMSE: {rmse:.4f}
Time: {gp_time:.1f}s

Brenier Map
Input dim: {d1}
Output dim: {X2.shape[1]}
Rescaling t: {brenier.t:.4f}
Entropy ε: {brenier.epsilon:.4f}
Time: {brenier_time:.1f}s

Prediction Metrics:
Return MSE: {mse_return:.6f}
Vol MSE: {mse_vol:.6f}
DD MSE: {mse_dd:.6f}
W₂ Error: {w2_error:.6f}
    """
    
    ax12.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
              verticalalignment='center')
    
    plt.tight_layout()
    
    output_file = 'hybrid_gp_brenier_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file}")
    
    #Conditional Distributions
    fig2 = plt.figure(figsize=(15, 5))
    
    for plot_idx, (scenario_name, idx, vol_value) in enumerate(scenarios):
        ax = plt.subplot(1, 3, plot_idx + 1)
        
        features = X1[idx, -d1:]
        prediction = hybrid.predict_future_prices(features, n_samples=2000)
        samples = prediction['samples']
        
        scatter = ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, 
                           c=samples[:, 2], cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Drawdown')
        
        true_point = X2[idx]
        ax.plot(true_point[0], true_point[1], 'r*', markersize=20,
                markeredgecolor='white', markeredgewidth=2, label='True')
        
        ax.plot(prediction['mean_return'], prediction['mean_vol'], 
                'g*', markersize=20, markeredgecolor='white', 
                markeredgewidth=2, label='Predicted')
        
        ax.set_xlabel('Future Return', fontsize=11)
        ax.set_ylabel('Future Volatility', fontsize=11)
        ax.set_title(f'{scenario_name}\nCurrent σ={vol_value:.3f}', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Conditional Distributions: μ(return, vol | features)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file2 = 'gp_conditional_distributions_2d.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file2}")
    
    #Convergence Analysis
    
    # Hold out a fixed test set BEFORE subsampling so that when
    # n == len(X1) the test queries are disjoint from training.
    n_conv_test     = 300
    all_idx         = np.arange(len(X1))
    conv_test_idx   = np.random.choice(all_idx, n_conv_test, replace=False)
    conv_train_pool = np.setdiff1d(all_idx, conv_test_idx)

    X1_conv_test = X1[conv_test_idx, -d1:]
    X2_conv_test = X2[conv_test_idx]

    max_n        = len(conv_train_pool)
    sample_sizes = [s for s in [500, 1000, 2000] if s <= max_n]
    if max_n not in sample_sizes:
        sample_sizes.append(max_n)

    t_values = [0.1 * (n ** (-1/3)) for n in sample_sizes]

    print(f"\n Testing theoretical rate: error ∝ n^(-2/3)")
    print(f"  Held-out test: {n_conv_test} samples (disjoint from training)")

    errors_by_size = []

    for n, t_param in zip(sample_sizes, t_values):
        print(f"\n    n={n}, t={t_param:.4f}")

        # Auto-scale epsilon from the data geometry (same as particle_filter.py)
        train_idx    = np.random.choice(conv_train_pool, n, replace=False)
        X1_sub       = X1[train_idx, -d1:]
        X2_sub       = X2[train_idx]

        d_full        = d1 + X2_sub.shape[1]
        A_t           = np.eye(d_full)
        A_t[d1:, d1:] *= np.sqrt(t_param)
        Z             = np.hstack([X1_sub, X2_sub]) @ A_t
        mean_sq_dist  = np.mean(np.sum(Z ** 2, axis=1))
        eps_param     = t_param ** 2 * max(mean_sq_dist, 0.01)

        brenier_temp = ConditionalBrenierEstimator(
            d1=d1, d2=X2_sub.shape[1], t=t_param, epsilon=eps_param
        )
        brenier_temp.fit(X1_sub, X2_sub)

        w2_error_temp = brenier_temp.compute_wasserstein_error(
            X1_conv_test, X2_conv_test
        )
        errors_by_size.append(w2_error_temp)
        print(f"W₂ error: {w2_error_temp:.6f}")
    
    # Plot convergence
    fig3 = plt.figure(figsize=(10, 6))
    
    ax = plt.subplot(111)
    ax.loglog(sample_sizes, errors_by_size, 'o-', linewidth=2, 
              markersize=10, label='Observed Error', color='blue')
    
    theoretical = errors_by_size[0] * (np.array(sample_sizes) / sample_sizes[0])**(-2/3)
    ax.loglog(sample_sizes, theoretical, '--', linewidth=2, 
              label='Theory: O(n^(-2/3))', color='red', alpha=0.7)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Wasserstein-2 Error', fontsize=12)
    ax.set_title('Convergence Rate Analysis\n', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_file3 = 'gp_convergence_analysis.png'
    plt.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file3}")
    
    #GP Uncertainty Analysis

    #Analyze GP predictions at different points
    X_test_gp = hybrid.gp_calibrator.normalize_weights(
        np.random.randn(100, len(best_weights)) * 0.1
    )
    
    means, stds = hybrid.gp_calibrator.gp.predict(X_test_gp, return_std=True)
    
    print(f"\n  GP Prediction Statistics:")
    print(f"Mean loss: {means.mean():.6f}")
    print(f"Std loss: {means.std():.6f}")
    print(f"Mean uncertainty: {stds.mean():.6f}")
    print(f"Max uncertainty: {stds.max():.6f}")
    
    #Plot uncertainty landscape
    fig4 = plt.figure(figsize=(12, 5))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(means, stds, alpha=0.5, s=20)
    ax1.set_xlabel('Predicted Loss', fontsize=11)
    ax1.set_ylabel('Prediction Uncertainty (σ)', fontsize=11)
    ax1.set_title('GP Uncertainty Landscape', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(1, 2, 2)
    sorted_idx = np.argsort(means)
    ax2.plot(means[sorted_idx], color='blue', label='Mean', linewidth=2)
    ax2.fill_between(range(len(means)), 
                     (means - 2*stds)[sorted_idx],
                     (means + 2*stds)[sorted_idx],
                     alpha=0.3, color='blue', label='95% CI')
    ax2.set_xlabel('Sample (sorted by mean)', fontsize=11)
    ax2.set_ylabel('Predicted Loss', fontsize=11)
    ax2.set_title('GP Predictions with Uncertainty', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file4 = 'gp_uncertainty_analysis.png'
    plt.savefig(output_file4, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file4}")

    # Summary

    print(f"Kernel: Constant × Matérn(ν=2.5)")
    print(f"Initial samples: {hybrid.gp_calibrator.n_initial} (Latin Hypercube)")
    print(f"BO iterations: {hybrid.gp_calibrator.n_iterations}")
    print(f"Total evaluations: {len(history['y'])}")
    print(f"Best loss: {np.min(history['y']):.6f}")
    print(f"Price MAE: {mae:.4f}")
    print(f"Calibration time: {gp_time:.1f}s")
    
    print(f"\n Conditional Price Prediction:")
    print(f"Rescaling: t={brenier.t:.4f}")
    print(f"Regularization: ε={brenier.epsilon:.6f}")
    print(f"Return MSE: {mse_return:.6f}")
    print(f"Volatility MSE: {mse_vol:.6f}")
    print(f"Wasserstein-2 error: {w2_error:.6f}")
    print(f"Fitting time: {brenier_time:.1f}s")
    
    print(f"\n Computational Efficiency:")
    print(f"Total time: {gp_time + brenier_time:.1f}s")
    print(f"GP evaluations: {len(history['y'])}")
    print(f"Sample efficiency: ~{len(strikes) * len(history['y'])//1000}k MC paths")
    print(f"Training samples: {len(X1)}")
    
    print(f"\n Output Files:")
    print(f"{output_file}")
    print(f"{output_file2}")
    print(f"{output_file3}")
    print(f"{output_file4}")
    
    print("Experiment Complete")

    plt.show()
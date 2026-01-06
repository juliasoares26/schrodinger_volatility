import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

# Import HestonModel
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from heston import HestonUnified


#Search for data file in multiple possible locations
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

#Conditional Brenier Map Estimator 

#Entropic Optimal Transport for conditional price prediction
#Implements rescaled quadratic cost approach from Baptista et al. (2024):
#Cost: c_t(x,y) = ½‖A_t(x-y)‖² where A_t = diag(1_d₁, √t·1_d₂)
#Sinkhorn algorithm for entropic regularization
#Barycentric projection for conditional mean prediction
#With t ∝ n^(-1/3) and ε ∝ t², achieves O(n^(-2/3)) rate

class ConditionalBrenierEstimator:
#Parameters - d1: int - Dimension of conditioning variables
#d2: int - Dimension of target variables
#t : float - Rescaling parameter - t ∝ n^(-1/3))
#epsilon : float - Entropic regularization -  ε ∝ t²
    def __init__(self, d1: int, d2: int, t: float = 0.01, epsilon: float = 0.001):
        self.d1 = d1
        self.d2 = d2
        self.t = t
        self.epsilon = epsilon
        
        # Rescaling matrix A_t
        d = d1 + d2
        self.A_t = np.eye(d)
        self.A_t[:d1, :d1] *= 1.0
        self.A_t[d1:, d1:] *= np.sqrt(t)
        
        #Training data storage
        self.X_train = None
        self.Y_train = None
        self.dual_potentials = None
    
    #Compute rescaled cost matrix C_ij = ½‖A_t(X_i - Y_j)‖²
    def compute_rescaled_cost(self, X, Y):
        X_scaled = X @ self.A_t
        Y_scaled = Y @ self.A_t
        
        C = cdist(X_scaled, Y_scaled, metric='sqeuclidean') / 2.0
        return C
    
     
    #Sinkhorn algorithm for entropic OT dual potentials.
    #Solves: min_π <C,π> + ε·KL(π‖ρ⊗μ)
    #Returns dual potential g satisfying:
    #π_ij ∝ exp((f_i + g_j - C_ij)/ε)
    
    def sinkhorn_dual_potentials(self, C, max_iter=1000, tol=1e-6):
        n, m = C.shape
        
        f = np.zeros(n)
        g = np.zeros(m)
        
        for iteration in range(max_iter):
            f_old = f.copy()
            
            #Update f
            f = -self.epsilon * np.log(
                np.sum(np.exp((g - C.T) / self.epsilon), axis=1) + 1e-10
            )
            
            #Update g
            g = -self.epsilon * np.log(
                np.sum(np.exp((f[:, np.newaxis] - C) / self.epsilon), axis=0) + 1e-10
            )
            
            #Convergence check
            if np.max(np.abs(f - f_old)) < tol:
                break
        
        return g
    
     
    #Fit conditional Brenier map to joint samples (X₁, X₂).
    #X1: ndarray, shape (n, d1) - Conditioning variables (past features)
    #X2: ndarray, shape (n, d2) - Target variables (future outcomes)
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
        start = time.time()
        self.dual_potentials = self.sinkhorn_dual_potentials(C)
        sinkhorn_time = time.time() - start
        
        print(f"Fitted on {n} samples ({sinkhorn_time:.1f}s)")
    
      #Predict X₂ given X₁ = x1_query via entropic Brenier map.
      #T̂_ε,t(x) = Σᵢ Yᵢ · wᵢ(x)
      #where wᵢ(x) ∝ exp((gᵢ - ½‖A_t(x-Yᵢ)‖²)/ε)
      #x1_query : ndarray, shape (d1,) or (n_query, d1)
      #prediction : ndarray, shape (d2,) or (n_query, d2)
      #Predicted target (conditional mean E[X₂|X₁])

    def predict(self, x1_query, return_distribution=False, n_samples=1000):
        x1_query = np.atleast_2d(x1_query)
        
        if self.dual_potentials is None:
            raise ValueError("Model not fitted, call fit() first.")
        
        #Extend query to full dimension
        x_extended = np.zeros((x1_query.shape[0], self.d1 + self.d2))
        x_extended[:, :self.d1] = x1_query
        
        predictions = []
        all_samples = [] if return_distribution else None
        
        for x in x_extended:
            #Compute barycentric weights
            x_scaled = self.A_t @ x
            Y_scaled = self.Y_train @ self.A_t.T
            
            distances = np.sum((Y_scaled - x_scaled) ** 2, axis=1) / 2.0
            
            #Weights
            log_weights = (self.dual_potentials - distances) / self.epsilon
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            
            #Barycentric projection (conditional mean)
            result = np.sum(self.Y_train * weights[:, np.newaxis], axis=0)
            predictions.append(result[self.d1:])
            
            #Sample from weighted distribution if requested
            if return_distribution:
                indices = np.random.choice(
                    len(self.Y_train), 
                    size=n_samples, 
                    p=weights,
                    replace=True
                )
                samples = self.Y_train[indices, self.d1:]
                all_samples.append(samples)
        
        prediction = np.array(predictions).squeeze()
        
        if return_distribution:
            return prediction, all_samples[0] if len(all_samples) == 1 else all_samples
        
        return prediction
    
    #Compute Wasserstein-2 distance between true and predicted conditionals
    def compute_wasserstein_error(self, x1_test, x2_test):
        predictions = []
        for x1 in x1_test:
            pred = self.predict(x1)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # W₂² ≈ MSE for Gaussians (approximation)
        mse = np.mean((predictions - x2_test) ** 2)
        return np.sqrt(mse)

#Compute Black-Scholes call option price
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
    
    print(f"\n Price Prediction Data:")
    print(f"Samples: {len(X1_full)}")
    print(f"Features (X₁): {X1_full.shape[1]} (includes IV surface)")
    print(f"Targets (X₂): {X2_full.shape[1]} (return, vol, drawdown)")
    print(f"Lookback: {lookback} days, Horizon: {horizon} days")
    
    #Extract maturity for SVI calibration
    S0 = float(heston_params['S0'])
    T = float(maturities[maturity_idx])
    market_vols = vol_surface[maturity_idx, :]
    strikes = strikes_norm * S0
    
    #Compute prices from IVs
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

#Wrapper around HestonModel with SVI-parametrized drift
#Drift function: λ(k) = a + b(ρ(k-m) + √((k-m)² + σ²)) where k = log(S/S0) is log-moneyness
class HestonModelWithSVIDrift:
    def __init__(self, heston_model):
        self.heston = heston_model

   #SVI-parametrized drift function
    def drift_lambda(self, t, S, v, svi_params):
        if svi_params is None or len(svi_params) == 0:
            return 0.0
        
        a, b, rho_svi, m, sigma_svi = svi_params
        
        k = np.log(S / self.heston.S0)
        drift = a + b * (rho_svi * (k - m) + np.sqrt((k - m)**2 + sigma_svi**2))
        
        return drift * np.sqrt(v / self.heston.theta)
    
    #Simulate paths with SVI drift on variance
    def simulate_paths_with_svi_drift(self, T, n_steps, n_paths, svi_params=None):
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.heston.S0
        v[:, 0] = self.heston.v0
        
        for i in range(n_steps):
            t_curr = times[i]
            
            Z1 = np.random.randn(n_paths)
            Z2 = np.random.randn(n_paths)
            
            W1 = Z1
            W2 = self.heston.rho * Z1 + np.sqrt(1 - self.heston.rho**2) * Z2
            
            v_curr = np.maximum(v[:, i], 1e-8)
            
            #SVI drift
            if svi_params is not None:
                lambda_drift = np.array([
                    self.drift_lambda(t_curr, S[j, i], v_curr[j], svi_params) 
                    for j in range(n_paths)
                ])
            else:
                lambda_drift = 0.0
            
            #Variance with drift
            drift_v = self.heston.kappa * (self.heston.theta - v_curr) + lambda_drift
            dv = drift_v * dt + self.heston.sigma * np.sqrt(v_curr * dt) * W2
            v[:, i + 1] = np.maximum(v_curr + dv, 1e-8)
            
            #Price (martingale)
            S[:, i + 1] = S[:, i] * np.exp(
                (self.heston.r - 0.5 * v_curr) * dt + np.sqrt(v_curr * dt) * W1
            )
        
        return S, v, times
    
    #Worker function for parallel pricing
    def _simulate_and_price_worker(self, n_paths_chunk, T, n_steps, strikes, svi_params):
        S, _, _ = self.simulate_paths_with_svi_drift(
            T, n_steps=n_steps, n_paths=n_paths_chunk, svi_params=svi_params
        )
        S_T = S[:, -1]
        
        #Compute payoffs for all strikes
        payoffs = np.maximum(S_T[:, np.newaxis] - strikes[np.newaxis, :], 0)
        return payoffs
    
    def price_calls(self, strikes, T, n_paths=10000, svi_params=None, n_jobs=-1):
        if n_jobs == -1:
            n_jobs = cpu_count()
        
        if n_jobs == 1:
            #Sequential fallback
            S, _, _ = self.simulate_paths_with_svi_drift(
                T, n_steps=100, n_paths=n_paths, svi_params=svi_params
            )
            S_T = S[:, -1]
            prices = []
            for K in strikes:
                payoff = np.maximum(S_T - K, 0)
                prices.append(np.exp(-self.heston.r * T) * np.mean(payoff))
            return np.array(prices)
        
        #Split paths across workers
        paths_per_job = n_paths // n_jobs
        remaining = n_paths % n_jobs
        
        job_sizes = [paths_per_job + (1 if i < remaining else 0) 
                     for i in range(n_jobs)]
        
        #Parallel simulation
        simulate_partial = partial(
            self._simulate_and_price_worker,
            T=T, n_steps=100, strikes=strikes, svi_params=svi_params
        )
        
        with Pool(n_jobs) as pool:
            results = pool.map(simulate_partial, job_sizes)
        
        #Aggregate results
        all_payoffs = np.concatenate(results, axis=0)
        prices = np.mean(all_payoffs, axis=0) * np.exp(-self.heston.r * T)
        
        return prices

#Parametrizes drift as: λ(k) = a + b(ρ(k-m) + √((k-m)² + σ²))
class SVIDriftCalibration:
    
    def __init__(self, heston_model: HestonUnified, n_jobs=-1):
        self.heston_svi = HestonModelWithSVIDrift(heston_model)
        self.svi_params = None
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        
        self.strikes = None
        self.market_prices = None
        self.T = None
        
        print(f"\n Parallel mode: {self.n_jobs} CPUs")
    
    def _objective_function(self, params):
        try:
            model_prices = self.heston_svi.price_calls(
                self.strikes, self.T, n_paths=5000, 
                svi_params=params, n_jobs=1  
            )
            loss = np.mean((model_prices - self.market_prices)**2)
        except:
            loss = 1e6
        return loss
   
    #Calibrate SVI parameters to market prices.
    #Minimizes: Σ [C_model(K; θ) - C_market(K)]² where θ = (a, b, ρ, m, σ) are SVI parameters
    
    def calibrate(self, strikes, market_prices, T, verbose=True):
        if verbose:
            print(f"Strikes: {len(strikes)}")
            print(f"Maturity: T={T:.2f}Y")
        
        #Initial guess
        initial_guess = [0.0, 0.1, -0.1, 0.0, 0.3]
        
        #Bounds
        bounds = [
            (-0.2, 0.2),   # a
            (0.01, 1.0),   # b
            (-0.9, 0.9),   # ρ
            (-1.0, 1.0),   # m
            (0.05, 2.0)    # σ
        ]
        
        def objective(params):
            try:
                model_prices = self.heston_svi.price_calls(
                    strikes, T, n_paths=5000, svi_params=params, n_jobs=self.n_jobs
                )
                loss = np.mean((model_prices - market_prices)**2)
            except:
                loss = 1e6
            return loss
        
        #No-arbitrage constraints
        def constraint1(params):
            a, b, rho, m, sigma = params
            return 4 - b * (1 + np.abs(rho))
        
        def constraint2(params):
            a, b, rho, m, sigma = params
            return a + b * sigma * np.sqrt(1 - rho**2) + 0.1
        
        constraints = [
            {'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2}
        ]
        
        if verbose:
            print(f"\n Optimizing SVI parameters...")
        
        start_time = time.time()
        
        #Multiple starts
        best_result = None
        best_loss = np.inf
        
        for attempt in range(3):
            if attempt == 0:
                x0 = initial_guess
            else:
                x0 = [np.random.uniform(bounds[i][0], bounds[i][1]) 
                      for i in range(len(bounds))]
            
            try:
                result = minimize(
                    objective, x0=x0, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 150, 'ftol': 1e-6}
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
                
                if result.success and verbose:
                    print(f"Attempt {attempt + 1}: Success (loss={result.fun:.6f})")
                    break
            except Exception as e:
                if verbose:
                    print(f"Attempt {attempt + 1}: {e}")
        
        result = best_result if best_result is not None else result
        
        #Fallback to global optimization
        if not result.success or result.fun > 0.1:
            if verbose:
                print(f"Trying Differential Evolution (parallel)...")
            
            result = differential_evolution(self._objective_function, bounds=bounds, maxiter=100, seed=42, workers=1, polish=True)
        
        calib_time = time.time() - start_time
        
        self.svi_params = result.x
        
        if verbose:
            a, b, rho, m, sigma = self.svi_params
            print(f"\n Calibration complete ({calib_time:.1f}s)")
            print(f"Loss (MSE): {result.fun:.6f}")
            print(f"Parameters: a={a:.4f}, b={b:.4f}, ρ={rho:.4f}, m={m:.4f}, σ={sigma:.4f}")
        
        return self.svi_params
    
    #Reprice options with calibrated SVI drift
    def reprice(self, strikes, T):
        if self.svi_params is None:
            raise ValueError("Must calibrate first")
        
        return self.heston_svi.price_calls(
            strikes, T, n_paths=20000, svi_params=self.svi_params, n_jobs=self.n_jobs
        )


class HybridSVIBrenierMethod:
  
    def __init__(self, heston_model: HestonUnified, n_jobs=-1):
        self.svi_calibrator = SVIDriftCalibration(heston_model, n_jobs=n_jobs)
        self.brenier_estimator = None
        self.heston_model = heston_model
    
    def calibrate_svi_drift(self, strikes, market_prices, T):
        start = time.time()
        svi_params = self.svi_calibrator.calibrate(strikes, market_prices, T)
        svi_time = time.time() - start
        
        print(f"SVI calibration complete ({svi_time:.1f}s)")
        
        return svi_params, svi_time
    
    def fit_brenier_map(self, X1, X2, d1, t=0.01, epsilon=0.001):
        
        X1_subset = X1[:, -d1:]
        d2 = X2.shape[1]
        
        start = time.time()
        self.brenier_estimator = ConditionalBrenierEstimator(
            d1=d1, d2=d2, t=t, epsilon=epsilon
        )
        self.brenier_estimator.fit(X1_subset, X2)
        brenier_time = time.time() - start
        
        print(f"Brenier map fitted ({brenier_time:.1f}s)")
        print(f"Input dim: {d1}, Output dim: {d2}")
        
        return self.brenier_estimator, brenier_time
    
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

if __name__ == "__main__":
    np.random.seed(42)

    try:
        data = load_unified_data(maturity_idx=3)
    except FileNotFoundError as e:
        print(str(e))
        print("\n Run: python generate_synthetic_data.py")
        sys.exit(1)
    
    #Extract data
    strikes = data['strikes']
    market_prices = data['market_prices']
    market_vols = data['market_vols']
    T = data['T']
    heston_params = data['heston_params']
    X1 = data['X1']
    X2 = data['X2']
    X1_raw = data['X1_raw']
    
    print(f"  Market Data:")
    print(f"Strikes: {len(strikes)} options at T={T:.2f}Y")
    print(f"Price range: [{market_prices.min():.4f}, {market_prices.max():.4f}]")
    print(f"\n Prediction Data:")
    print(f"Samples: {len(X1)}")
    print(f"Features: {X1.shape[1]}")
    print(f"Targets: {X2.shape[1]} (return, vol, drawdown)")
    
    #Initialize hybrid method with parallelization
    heston_model = HestonUnified(**heston_params)
    n_cpus = cpu_count()
    print(f"\n Available CPUs: {n_cpus}")
    hybrid = HybridSVIBrenierMethod(heston_model, n_jobs=n_cpus)
    
    #SVI Calibration
    svi_params, svi_time = hybrid.calibrate_svi_drift(strikes, market_prices, T)
    
    #Brenier Map
    d1 = 7  #Use last 7 features
    brenier, brenier_time = hybrid.fit_brenier_map(X1, X2, d1=d1, t=0.02, epsilon=0.004)

    print("Calibration Times")
    print(f"SVI calibration: {svi_time:.1f}s")
    print(f"Brenier fitting: {brenier_time:.1f}s")
    print(f"Total: {svi_time + brenier_time:.1f}s")
    
    #Validation 1: SVI drift accuracy
    model_prices = hybrid.svi_calibrator.reprice(strikes, T)
    
    errors = np.abs(model_prices - market_prices)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    #Validation 2: Price prediction accuracy
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
    
    #Demonstration; Scenarios

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
        
        print(f"Predicted Return: {prediction['mean_return']:+.4f} "
              f"(true: {true_values[0]:+.4f})")
        print(f"Predicted Vol: {prediction['mean_vol']:.4f} "
              f"(true: {true_values[1]:.4f})")
        print(f"Predicted Drawdown: {prediction['mean_drawdown']:.4f} "
              f"(true: {true_values[2]:.4f})")
    
    #Visualizations
    fig = plt.figure(figsize=(18, 12))
    
    #Plot 1: Price comparison
    ax1 = plt.subplot(3, 4, 1)
    x_pos = np.arange(len(strikes))
    width = 0.35
    ax1.bar(x_pos - width/2, market_prices, width, label='Market', 
            alpha=0.8, color='blue')
    ax1.bar(x_pos + width/2, model_prices, width, label='SVI Model', 
            alpha=0.8, color='orange')
    ax1.set_xlabel('Strike Index', fontsize=10)
    ax1.set_ylabel('Call Price', fontsize=10)
    ax1.set_title('SVI Calibration: Price Fit', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    #Plot 2: SVI drift function
    ax2 = plt.subplot(3, 4, 2)
    k_range = np.linspace(-0.3, 0.3, 100)
    a, b, rho_svi, m, sigma_svi = svi_params
    drift_values = a + b * (rho_svi * (k_range - m) + 
                           np.sqrt((k_range - m)**2 + sigma_svi**2))
    
    ax2.plot(k_range, drift_values, linewidth=2, color='green')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Log-Moneyness k', fontsize=10)
    ax2.set_ylabel('Drift λ(k)', fontsize=10)
    ax2.set_title('SVI Drift Function', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    #Plot 3: IV comparison
    ax3 = plt.subplot(3, 4, 3)
    model_vols = np.array([
        black_scholes_iv(heston_params['S0'], K, T, price)
        for K, price in zip(strikes, model_prices)
    ])
    strikes_norm = strikes / heston_params['S0']
    ax3.plot(strikes_norm, market_vols*100, 'o-', linewidth=2.5, 
             markersize=8, color='black', label='Market')
    ax3.plot(strikes_norm, model_vols*100, 's--', linewidth=2, 
             markersize=6, alpha=0.7, color='orange', label='SVI Model')
    ax3.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Strike / Spot', fontsize=10)
    ax3.set_ylabel('Implied Volatility (%)', fontsize=10)
    ax3.set_title('Volatility Smile', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    #Plot 4: Calibration errors
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(strikes, errors, 'o-', linewidth=2, markersize=8, color='red')
    ax4.fill_between(strikes, 0, errors, alpha=0.3, color='red')
    ax4.set_xlabel('Strike', fontsize=10)
    ax4.set_ylabel('Absolute Error', fontsize=10)
    ax4.set_title(f'SVI Errors (MAE={mae:.4f})', fontsize=11, fontweight='bold')
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
    ax5.set_title(f'Return Prediction (MSE={mse_return:.4f})', 
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
    ax6.set_title(f'Vol Prediction (MSE={mse_vol:.4f})', 
                  fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    #Plot 7: Drawdown prediction
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
SVI Calibration:
Parameters: 5 (a, b, ρ, m, σ)
MAE: {mae:.4f}
RMSE: {rmse:.4f}
Time: {svi_time:.1f}s

Brenier Map:
Input dim: {d1}
Output dim: {X2.shape[1]}
Rescaling t: {brenier.t:.4f}
Entropy ε: {brenier.epsilon:.4f}
Time: {brenier_time:.1f}s

Prediction Accuracy:
Return MSE: {mse_return:.6f}
Vol MSE: {mse_vol:.6f}
DD MSE: {mse_dd:.6f}
W₂ Error: {w2_error:.6f}
    """
    
    ax12.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
              verticalalignment='center')
    
    plt.suptitle('Hybrid Method: SVI + Conditional Brenier Maps (Parallelized)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = 'hybrid_svi_brenier_parallel_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file}")
    
    #Conditional Distributions
    
    fig2 = plt.figure(figsize=(15, 5))
    
    for plot_idx, (scenario_name, idx, vol_value) in enumerate(scenarios):
        ax = plt.subplot(1, 3, plot_idx + 1)
        
        features = X1[idx, -d1:]
        prediction = hybrid.predict_future_prices(features, n_samples=2000)
        samples = prediction['samples']
        
        #return vs future vol
        scatter = ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, 
                           c=samples[:, 2], cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Drawdown')
        
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
    
    plt.suptitle('Conditional Distributions: μ(return, vol | features)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file2 = 'svi_conditional_distributions_2d_parallel.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file2}")
    
    #Convergence Analysis
    
    sample_sizes = [500, 1000, 2000, 3000, len(X1)]
    t_values = [0.1 * (n ** (-1/3)) for n in sample_sizes]
    epsilon_values = [t**2 for t in t_values]
    
    print(f"\n Testing theoretical rate: error ∝ n^(-2/3)")
    
    errors_by_size = []
    
    for n, t_param, eps_param in zip(sample_sizes, t_values, epsilon_values):
        print(f"\n    n={n}, t={t_param:.4f}, ε={eps_param:.6f}")
        
        indices = np.random.choice(len(X1), min(n, len(X1)), replace=False)
        X1_sub = X1[indices, -d1:]
        X2_sub = X2[indices]
        
        brenier_temp = ConditionalBrenierEstimator(d1=d1, d2=3, 
                                                    t=t_param, epsilon=eps_param)
        brenier_temp.fit(X1_sub, X2_sub)
        
        test_indices = np.random.choice(len(X1), 200, replace=False)
        X1_test_conv = X1[test_indices, -d1:]
        X2_test_conv = X2[test_indices]
        
        w2_error_temp = brenier_temp.compute_wasserstein_error(
            X1_test_conv, X2_test_conv
        )
        errors_by_size.append(w2_error_temp)
        
        print(f"W₂ error: {w2_error_temp:.6f}")
    
    #Plot convergence
    fig3 = plt.figure(figsize=(10, 6))
    
    ax = plt.subplot(111)
    ax.loglog(sample_sizes, errors_by_size, 'o-', linewidth=2, 
              markersize=10, label='Observed Error', color='blue')
    
    #Theoretical rate
    theoretical = errors_by_size[0] * (np.array(sample_sizes) / sample_sizes[0])**(-2/3)
    ax.loglog(sample_sizes, theoretical, '--', linewidth=2, 
              label='Theory: O(n^(-2/3))', color='red', alpha=0.7)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Wasserstein-2 Error', fontsize=12)
    ax.set_title('Convergence Rate Analysis\n(Baptista et al. Theorem 4.2)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_file3 = 'svi_convergence_analysis_parallel.png'
    plt.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file3}")
    
#summary
    
    print(f"\n✅ SVI Drift Calibration:")
    print(f"Parameters: a={svi_params[0]:.4f}, b={svi_params[1]:.4f}, "
          f"ρ={svi_params[2]:.4f}, m={svi_params[3]:.4f}, σ={svi_params[4]:.4f}")
    print(f" Price fitting MAE: {mae:.4f}")
    print(f"   • Calibration time: {svi_time:.1f}s ({n_cpus} CPUs)")
    
    print(f"\n Conditional Price Prediction:")
    print(f"Method: Entropic Optimal Transport")
    print(f"Rescaling: t={brenier.t:.4f} ∝ n^(-1/3)")
    print(f"Regularization: ε={brenier.epsilon:.6f} ∝ t²")
    print(f"Return MSE: {mse_return:.6f}")
    print(f"Volatility MSE: {mse_vol:.6f}")
    print(f"Wasserstein-2 error: {w2_error:.6f}")
    print(f"Fitting time: {brenier_time:.1f}s")
    
    print(f"\n Computational Efficiency:")
    print(f"Total time: {svi_time + brenier_time:.1f}s")
    print(f"SVI: ~{len(strikes)} MC simulations per iteration")
    print(f"Parallelization: {n_cpus} CPUs")
    print(f"Training samples: {len(X1)}")
    
    print(f"\n Output Files")
    print(f"{output_file}")
    print(f"{output_file2}")
    print(f"{output_file3}")

    plt.show()
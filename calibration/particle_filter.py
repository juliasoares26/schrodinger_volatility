import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import sys

# HestonUnified lives in calibration/base.py
sys.path.insert(0, str(Path(__file__).parent))
from base import HestonUnified

# models/ for any other imports
sys.path.append(str(Path(__file__).parent.parent / 'models'))

_N_CPUS = cpu_count()

# Module-level RBF constants — inherited by workers without pickling overhead
_S_CENTERS_NORM  = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
_V_CENTERS = np.array([0.02, 0.04, 0.06])
_RBF_CENTERS_UNIT = np.array(
    [[Sc, vc] for Sc in _S_CENTERS_NORM for vc in _V_CENTERS]
)  # (15, 2) — S column scaled by S0 at call time
_RBF_SCALES = np.array([20.0, 0.02])

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
    
    # Recursive search
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
    # t : float - Rescaling parameter (t ∝ n^(-1/3))
    # epsilon: float - Entropic regularization (theory: ε ∝ t²)

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
        
        # Storage for training data
        self.X_train = None
        self.Y_train = None
        self.dual_potentials = None

        # Cached arrays for fast predict (filled by fit())
        self._Y_scaled = None
        self._Y_target = None
    
    # Compute rescaled cost matrix C_ij = ½‖A_t(X_i - Y_j)‖²
    def compute_rescaled_cost(self, X, Y):
        X_scaled = X @ self.A_t
        Y_scaled = Y @ self.A_t
        C = cdist(X_scaled, Y_scaled, metric='sqeuclidean') / 2.0
        return C
    
    # Sinkhorn algorithm for entropic OT dual potentials
    # Solves: min_π <C,π> + ε·KL(π‖ρ⊗μ)
    # Returns: g ∈ ℝⁿ (dual potential on target)
    #
    # Optimized: precompute K = exp(-C/ε) once, iterate via K-matvecs

    def sinkhorn_dual_potentials(self, C, max_iter=1000, tol=1e-6):
        n, m = C.shape
        K = np.exp(-C / self.epsilon)   # precomputed once
        f = np.zeros(n)
        g = np.zeros(m)
        for iteration in range(max_iter):
            f_old = f.copy()
            u = np.exp(g / self.epsilon)
            f = -self.epsilon * np.log(K @ u + 1e-300)
            v = np.exp(f / self.epsilon)
            g = -self.epsilon * np.log(K.T @ v + 1e-300)
            if np.max(np.abs(f - f_old)) < tol:
                break
        return g
    
    # Fit conditional Brenier map to joint samples
    # X1: ndarray, shape (n, d1) - Past features
    # X2: ndarray, shape (n, d2) - Future targets
    
    def fit(self, X1, X2):
        print(f"\n Fitting Conditional Brenier Map")
        print(f"d1={self.d1}, d2={self.d2}, t={self.t:.4f}, ε={self.epsilon:.4f}")
        n = X1.shape[0]
        self.X_train = np.hstack([X1, X2])
        self.Y_train = np.hstack([X1, X2])
        C = self.compute_rescaled_cost(self.X_train, self.Y_train)
        self.dual_potentials = self.sinkhorn_dual_potentials(C)
        #Cache for predict: avoid recomputing per call
        self._Y_scaled = self.Y_train @ self.A_t.T   # (n, d)
        self._Y_target = self.Y_train[:, self.d1:]   # (n, d2)
        print(f"Fitted on {n} samples")

    # Predict X₂ given X₁ = x1_query — fully batched over all queries
    # T̂_ε,t(x) = Σᵢ Yᵢ · wᵢ(x)  where wᵢ ∝ exp((gᵢ - ½‖A_t(x-Yᵢ)‖²)/ε)

    def predict(self, x1_query, return_distribution=False):
        x1_query = np.atleast_2d(x1_query)
        n_q = x1_query.shape[0]
        
        if self.dual_potentials is None:
            raise ValueError("Model not fitted, call fit() first")
        
        # Extend queries to full dimension (X₁ fixed, X₂ = 0 placeholder)
        x_extended = np.zeros((n_q, self.d1 + self.d2))
        x_extended[:, :self.d1] = x1_query
        
        # Scale all queries at once
        XS = x_extended @ self.A_t.T                                          # (n_q, d)

        # Batched squared distances: ½‖XS[i] - Y_scaled[j]‖²
        sq_X    = np.sum(XS**2, axis=1, keepdims=True)                        # (n_q, 1)
        sq_Y    = np.sum(self._Y_scaled**2, axis=1, keepdims=True).T          # (1, n_train)
        dist_mat = (sq_X + sq_Y - 2.0 * (XS @ self._Y_scaled.T)) / 2.0       # (n_q, n_train)

        # Numerically stable softmax
        log_w  = (self.dual_potentials[None, :] - dist_mat) / self.epsilon    # (n_q, n_train)
        log_w -= log_w.max(axis=1, keepdims=True)
        w_mat  = np.exp(log_w)
        w_mat /= w_mat.sum(axis=1, keepdims=True)
        
        # Barycentric projection
        predictions = w_mat @ self._Y_target                                   # (n_q, d2)
        prediction  = predictions.squeeze()
        
        if return_distribution:
            #Sample from the query's weight distribution
            weights_q = w_mat[-1]
            indices = np.random.choice(
                len(self.Y_train), size=1000, p=weights_q, replace=True
            )
            samples = self._Y_target[indices]
            return prediction, samples
        
        return prediction
    
    # Compute Wasserstein-2 distance — single batched call, no Python loop
    def compute_wasserstein_error(self, x1_test, x2_test):
        predictions = self.predict(x1_test)
        if predictions.ndim == 1:
            predictions = predictions[np.newaxis, :]
        mse = np.mean((predictions - x2_test) ** 2)
        return np.sqrt(mse)

# Top-level worker — required for multiprocessing pickle on Windows (spawn).
# Prices a chunk of particles for one (strike, T) using antithetic variates and fully vectorized drift across all paths simultaneously.

def _price_particle_chunk(args):
    (weights_chunk, S0, v0, kappa, theta, sigma, rho,
     strike, T, n_paths, n_steps) = args

    dt          = T / n_steps
    sqrt_dt     = np.sqrt(dt)
    sqrt_1mrho2 = np.sqrt(1.0 - rho**2)
    n_p         = len(weights_chunk)
    total       = n_p * n_paths

    # Antithetic variates: mirror noise to halve MC variance
    half = total // 2
    Z1 = np.random.randn(n_steps, half)
    Z2 = np.random.randn(n_steps, half)
    rW = np.concatenate([Z1, -Z1], axis=1)
    rZ = np.concatenate([Z2, -Z2], axis=1)

    S = np.full(total, S0, dtype=np.float64)
    v = np.full(total, v0, dtype=np.float64)
    pid = np.repeat(np.arange(n_p), n_paths)   # path → particle index

    centers = _RBF_CENTERS_UNIT.copy()
    centers[:, 0]  *= S0
    sv_buf = np.empty((total, 2))

    for i in range(n_steps):
        dW = rW[i] * sqrt_dt
        dZ = rho * rW[i] * sqrt_dt + sqrt_1mrho2 * rZ[i] * sqrt_dt
        vc = np.maximum(v, 1e-8)

        sv_buf[:, 0] = S
        sv_buf[:, 1] = vc
        diff = (sv_buf[:, None, :] - centers[None, :, :]) / _RBF_SCALES  # (total,15,2)
        rbf = np.exp(-0.5 * np.sum(diff**2, axis=2))                     # (total,15)
        drift = (rbf * weights_chunk[pid]).sum(axis=1)                     # (total,)

        v = np.maximum(vc + (kappa*(theta - vc) + drift)*dt
                          + sigma*np.sqrt(vc)*dZ, 1e-8)
        S = S * np.exp(-0.5*vc*dt + np.sqrt(vc)*dW)

    payoffs = np.maximum(S.reshape(n_p, n_paths) - strike, 0.0)
    return payoffs.mean(axis=1)   # (n_p,)


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

        #Pre-scaled RBF centers for this S0
        self._centers = _RBF_CENTERS_UNIT.copy()
        self._centers[:, 0]  *= S0

    #Vectorized drift over a batch of paths — (n,) → (n,)
    def drift_lambda_vectorized(self, S_arr, v_arr, weights):
        if weights is None or len(weights) == 0:
            return np.zeros(len(S_arr))
        sv   = np.stack([S_arr, v_arr], axis=1)
        diff = (sv[:, None, :] - self._centers[None, :, :]) / _RBF_SCALES
        rbf  = np.exp(-0.5 * np.sum(diff**2, axis=2))
        return rbf @ weights

    #Simulate paths — pre-allocated noise, vectorized drift
    def simulate_paths(self, T, n_steps, n_paths=1, weights=None, 
                       return_full_path=True, seed=None):
        if weights is None:
            return self.heston.simulate_paths(T, n_steps, n_paths, seed=seed)
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        sqrt_1mrho2 = np.sqrt(1.0 - self.rho**2)
        times = np.linspace(0, T, n_steps + 1)
        S = np.ones((n_paths, n_steps + 1)) * self.S0
        v = np.ones((n_paths, n_steps + 1)) * self.v0
        Z1_all = np.random.randn(n_steps, n_paths)
        Z2_all = np.random.randn(n_steps, n_paths)
        for i in range(n_steps):
            dW = Z1_all[i] * sqrt_dt
            dZ = self.rho * Z1_all[i] * sqrt_dt + sqrt_1mrho2 * Z2_all[i] * sqrt_dt
            vc = np.maximum(v[:, i], 1e-8)
            lam = self.drift_lambda_vectorized(S[:, i], vc, weights)
            v[:, i+1] = np.maximum(
                vc + (self.kappa*(self.theta - vc) + lam)*dt + self.sigma*np.sqrt(vc)*dZ,
                1e-8)
            S[:, i+1] = S[:, i] * np.exp(-0.5*vc*dt + np.sqrt(vc)*dW)
        if return_full_path:
            return S, v, times
        return S[:, -1], v[:, -1], None

    #Price calls — antithetic variates + all strikes broadcast
    def price_calls(self, strikes, T, n_paths=10000, weights=None):
        half = n_paths // 2
        n_steps = 100
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        sqrt_1mrho2 = np.sqrt(1.0 - self.rho**2)
        Z1 = np.random.randn(n_steps, half)
        Z2 = np.random.randn(n_steps, half)
        Z1f = np.concatenate([Z1, -Z1], axis=1)
        Z2f = np.concatenate([Z2, -Z2], axis=1)
        S_arr = np.full(n_paths, self.S0)
        v_arr = np.full(n_paths, self.v0)
        for i in range(n_steps):
            dW = Z1f[i] * sqrt_dt
            dZ = self.rho * Z1f[i] * sqrt_dt + sqrt_1mrho2 * Z2f[i] * sqrt_dt
            vc = np.maximum(v_arr, 1e-8)
            lam = self.drift_lambda_vectorized(S_arr, vc, weights) if weights is not None else 0.0
            v_arr = np.maximum(
                vc + (self.kappa*(self.theta - vc) + lam)*dt + self.sigma*np.sqrt(vc)*dZ,
                1e-8)
            S_arr = S_arr * np.exp(-0.5*vc*dt + np.sqrt(vc)*dW)
        payoffs = np.maximum(S_arr[:, None] - np.array(strikes)[None, :], 0.0)
        return payoffs.mean(axis=0)

# Black-Scholes Implied Volatility 

# Compute implied volatility via Newton-Raphson
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

# Load unified data with:
# Volatility surface
# Price prediction features 

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
    def __init__(self, n_particles=200, n_weights=15, heston_params=None,
                 n_cpus=None):
        self.n_particles = n_particles
        self.n_weights   = n_weights
        self.n_cpus      = n_cpus or _N_CPUS
        
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
    
    # ESS = (Σw_i)² / Σw_i²
    def effective_sample_size(self):
        return (np.sum(self.weights)**2) / np.sum(self.weights**2)
    
    # Update weights based on likelihood
    # All particles priced in parallel — each worker simulates a chunk
    # with antithetic variates and vectorized drift.
    def reweight(self, market_price, strike, T, noise_std=0.3,
                 n_paths=1000, n_steps=50):
        h = self.heston
        chunk_size = max(1, self.n_particles // self.n_cpus)
        chunks = [self.particles[i:i+chunk_size]
                  for i in range(0, self.n_particles, chunk_size)]
        args = [(chunk, h.S0, h.v0, h.kappa, h.theta, h.sigma, h.rho,
                 strike, T, n_paths, n_steps)
                for chunk in chunks]
        if self.n_cpus > 1:
            with Pool(self.n_cpus) as pool:
                results = pool.map(_price_particle_chunk, args)
        else:
            results = [_price_particle_chunk(a) for a in args]
        model_prices = np.concatenate(results)
        likelihoods  = norm.pdf(market_price, loc=model_prices, scale=noise_std)
        self.weights *= likelihoods
        self.weights /= (np.sum(self.weights) + 1e-10)
    
    # Systematic resampling when ESS low
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
    
    # MCMC move
    def move(self, step_size=0.01):
        noise = np.random.randn(self.n_particles, self.n_weights) * step_size
        self.particles += noise
    
    def calibrate_drift(self, strikes, market_prices, T):
        self.initialize_prior()
        
        for i, (K, C_mkt) in enumerate(zip(strikes, market_prices)):
            print(f"\n  Strike {i+1}/{len(strikes)}: K={K:.0f}, C={C_mkt:.4f}")
            
            # noise_std scales with price — avoids ESS collapse on deep-ITM strikes
            noise_std = max(0.05 * C_mkt, 0.05)
            self.reweight(C_mkt, K, T, noise_std=noise_std)
            ess = self.effective_sample_size()
            print(f"ESS: {ess:.0f}/{self.n_particles}")
            
            if self.resample(threshold=0.5):
                self.move(step_size=0.02)
        
        self.calibrated_drift = np.average(self.particles, weights=self.weights, axis=0)
        
        print(f"\nDrift calibration complete")
        print(f"Weight range: [{self.calibrated_drift.min():.4f}, "
              f"{self.calibrated_drift.max():.4f}]")
        
        return self.calibrated_drift
    
    # Fit Conditional Brenier Map for price prediction.
    # X1: ndarray, shape (n, d_features)
    # Past features (vol, momentum, IV, etc.)
    # X2 : ndarray, shape (n, 3)
    # Future targets (return, vol, drawdown)
    # d1 : int - Number of conditioning features to use
   
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
 
    # Load unified data
    try:
        data = load_unified_data(maturity_idx=3)
    except FileNotFoundError as e:
        print(str(e))
        print("\n💡 Run: python generate_synthetic_data.py")
        exit(1)
    
    # Extract data
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
    
    # Initialize hybrid calibration
    hybrid = HybridCalibration(
        n_particles=500,
        n_weights=15,
        heston_params=heston_params
    )
    
    # Calibrate drift
    start_time = time.time()
    calibrated_drift = hybrid.calibrate_drift(strikes, market_prices, T)
    drift_time = time.time() - start_time
    
    # Fit Brenier map
    start_time = time.time()
    d1 = 7  # last 7 features: realized vol, current vol, momentum, etc.
    # t and epsilon auto-tuned: t* = 0.1*n^(-1/3), epsilon* = t*^2
    n_train = len(X1)
    t_auto  = 0.1 * (n_train ** (-1/3))
    eps_auto = t_auto ** 2
    print(f"  Brenier auto-params: t={t_auto:.4f}  ε={eps_auto:.6f}")
    brenier = hybrid.fit_brenier_map(X1, X2, d1=d1, t=t_auto, epsilon=eps_auto)
    brenier_time = time.time() - start_time
    
    print("Calibration Times")
    print(f"Drift calibration: {drift_time:.1f}s")
    print(f"Brenier map fitting: {brenier_time:.1f}s")
    print(f"Total: {drift_time + brenier_time:.1f}s")
    
    # Drift calibration accuracy
    
    model_prices = hybrid.heston.price_calls(strikes, T, n_paths=20000, 
                                            weights=calibrated_drift)
    
    errors = np.abs(model_prices - market_prices)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Price prediction accuracy
    
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
    
    # Compute metrics
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
    
    # Plot 1: Price comparison
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
    
    # Plot 2: Pricing errors
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(strikes, errors, 'o-', linewidth=2, markersize=8, color='red')
    ax2.fill_between(strikes, 0, errors, alpha=0.3, color='red')
    ax2.set_xlabel('Strike', fontsize=10)
    ax2.set_ylabel('Absolute Error', fontsize=10)
    ax2.set_title(f'Calibration Errors (MAE={mae:.4f})', 
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: IV comparison
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
    
    # Plot 4: Learned drift weights
    ax4 = plt.subplot(3, 4, 4)
    ax4.bar(range(len(calibrated_drift)), calibrated_drift, 
            alpha=0.7, color='green')
    ax4.set_xlabel('Basis Function', fontsize=10)
    ax4.set_ylabel('Weight', fontsize=10)
    ax4.set_title('Learned Drift λ(t,S,v)', fontsize=11, fontweight='bold')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Return prediction scatter
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
    
    # Plot 6: Volatility prediction scatter
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
    
    # Plot 7: Drawdown prediction scatter
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
    
    # Plot 8: Prediction errors histogram
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
    print(f"Saved: {output_file2}")
    
    # Convergence Analysis

    # Hold out a fixed test set BEFORE subsampling. Without this, when
    # n == len(X1) the test queries are inside training and W₂ collapses.
    n_conv_test = 300
    all_idx = np.arange(len(X1))
    conv_test_idx  = np.random.choice(all_idx, n_conv_test, replace=False)
    conv_train_pool = np.setdiff1d(all_idx, conv_test_idx)

    X1_conv_test = X1[conv_test_idx, -d1:]
    X2_conv_test = X2[conv_test_idx]

    max_n = len(conv_train_pool)
    sample_sizes = [s for s in [500, 1000, 2000] if s <= max_n]
    if max_n not in sample_sizes:
        sample_sizes.append(max_n)

    t_values = [0.1 * (n ** (-1/3)) for n in sample_sizes]

    print(f"\n Testing convergence with varying sample sizes")
    print(f"Held-out test: {n_conv_test} samples (disjoint from training)")

    errors_by_size = []

    for n, t_param in zip(sample_sizes, t_values):
        #Subsample strictly from training pool
        train_idx = np.random.choice(conv_train_pool, n, replace=False)
        X1_sub = X1[train_idx, -d1:]
        X2_sub = X2[train_idx]

        #Auto-scale epsilon identically to how fit_brenier_map does it
        d_full = d1 + X2_sub.shape[1]
        A_t = np.eye(d_full)
        A_t[d1:, d1:] *= np.sqrt(t_param)
        Z = np.hstack([X1_sub, X2_sub]) @ A_t
        mean_sq_dist = np.mean(np.sum(Z**2, axis=1))
        eps_param = t_param**2 * max(mean_sq_dist, 0.01)

        print(f"\n n={n}, t={t_param:.4f}, ε={eps_param:.6f}")

        brenier_temp = ConditionalBrenierEstimator(
            d1=d1, d2=3, t=t_param, epsilon=eps_param
        )
        brenier_temp.fit(X1_sub, X2_sub)

        w2_error_temp = brenier_temp.compute_wasserstein_error(
            X1_conv_test, X2_conv_test
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
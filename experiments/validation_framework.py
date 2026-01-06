import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

N_CPUS = cpu_count()
print(f"Parallelization enabled: {N_CPUS} CPUs available")

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
    
    for path in possible_paths:
        if path.exists():
            return path
    
    for path in [script_dir, Path.cwd()]:
        for depth in range(search_depth + 1):
            pattern = '/'.join(['*'] * depth) + f'/{filename}'
            matches = list(path.glob(pattern))
            if matches:
                return matches[0]
    
    return None

# Black-scholes implied volatility

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

#Conditional Brenier Estimator

class ConditionalBrenierEstimator:
    def __init__(self, d1: int, d2: int, t: float = 0.01, epsilon: float = 0.001):
        self.d1 = d1
        self.d2 = d2
        self.t = t
        self.epsilon = epsilon
        
        d = d1 + d2
        self.A_t = np.eye(d)
        self.A_t[:d1, :d1] *= 1.0
        self.A_t[d1:, d1:] *= np.sqrt(t)
        
        self.X_train = None
        self.Y_train = None
        self.dual_potentials = None
    
    def compute_rescaled_cost(self, X, Y):
        X_scaled = X @ self.A_t
        Y_scaled = Y @ self.A_t
        return cdist(X_scaled, Y_scaled, metric='sqeuclidean') / 2.0
    
    def sinkhorn_dual_potentials(self, C, max_iter=1000, tol=1e-6):
        n, m = C.shape
        f = np.zeros(n)
        g = np.zeros(m)
        
        for _ in range(max_iter):
            f_old = f.copy()
            f = -self.epsilon * np.log(np.sum(np.exp((g - C.T) / self.epsilon), axis=1) + 1e-10)
            g = -self.epsilon * np.log(np.sum(np.exp((f[:, np.newaxis] - C) / self.epsilon), axis=0) + 1e-10)
            if np.max(np.abs(f - f_old)) < tol:
                break
        
        return g
    
#Fit conditional Brenier map
    def fit(self, X1, X2, verbose=False):
        n = X1.shape[0]
        self.X_train = np.hstack([X1, X2])
        self.Y_train = np.hstack([X1, X2])
        
        C = self.compute_rescaled_cost(self.X_train, self.Y_train)
        self.dual_potentials = self.sinkhorn_dual_potentials(C)
        
        if verbose:
            print(f"Brenier fitted on {n} samples")

    #Predict X₂ given X₁  
    def predict(self, x1_query):
        x1_query = np.atleast_2d(x1_query)
        x_extended = np.zeros((x1_query.shape[0], self.d1 + self.d2))
        x_extended[:, :self.d1] = x1_query
        
        predictions = []
        for x in x_extended:
            x_scaled = self.A_t @ x
            Y_scaled = self.Y_train @ self.A_t.T
            distances = np.sum((Y_scaled - x_scaled) ** 2, axis=1) / 2.0
            
            log_weights = (self.dual_potentials - distances) / self.epsilon
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            
            result = np.sum(self.Y_train * weights[:, np.newaxis], axis=0)
            predictions.append(result[self.d1:])
        
        return np.array(predictions).squeeze()
    
    #Compute W₂ error
    def compute_wasserstein_error(self, x1_test, x2_test):
        predictions = np.array([self.predict(x1) for x1 in x1_test])
        mse = np.mean((predictions - x2_test) ** 2)
        return np.sqrt(mse)


# Heston Wrapper for drift networks 
class HestonWithDrift:
    def __init__(self, heston_model, n_weights=15):
        self.heston = heston_model
        self.n_weights = n_weights
    
    #RBF-based drift (for PF, GP)
    def drift_lambda_rbf(self, t, S, v, weights):
        if weights is None or len(weights) == 0:
            return 0.0
        
        S_centers = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * self.heston.S0
        v_centers = np.array([0.02, 0.04, 0.06])
        
        features = []
        for S_c in S_centers:
            for v_c in v_centers:
                rbf = np.exp(-0.5 * (((S - S_c)/20)**2 + ((v - v_c)/0.02)**2))
                features.append(rbf)
        
        return np.dot(weights, np.array(features))
    
    #SVI-based drift (for SVI method)
    def drift_lambda_svi(self, t, S, v, svi_params):
        if svi_params is None or len(svi_params) == 0:
            return 0.0
        
        a, b, rho_svi, m, sigma_svi = svi_params
        k = np.log(S / self.heston.S0)
        drift = a + b * (rho_svi * (k - m) + np.sqrt((k - m)**2 + sigma_svi**2))
        return drift * np.sqrt(v / self.heston.theta)
    
    #Worker function for parallel path simulation
    def _simulate_single_path_batch(self, args):
        n_paths_chunk, T, n_steps, weights, is_svi, seed = args
        
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        S = np.ones((n_paths_chunk, n_steps + 1)) * self.heston.S0
        v = np.ones((n_paths_chunk, n_steps + 1)) * self.heston.v0
        
        for i in range(n_steps):
            t_curr = times[i]
            
            dW = np.random.randn(n_paths_chunk) * np.sqrt(dt)
            dZ_indep = np.random.randn(n_paths_chunk) * np.sqrt(dt)
            dZ = self.heston.rho * dW + np.sqrt(1 - self.heston.rho**2) * dZ_indep
            
            v_current = np.maximum(v[:, i], 1e-8)
            
            if weights is not None:
                if is_svi:
                    lambda_drift = np.array([
                        self.drift_lambda_svi(t_curr, S[j, i], v_current[j], weights) 
                        for j in range(n_paths_chunk)
                    ])
                else:
                    lambda_drift = np.array([
                        self.drift_lambda_rbf(t_curr, S[j, i], v_current[j], weights) 
                        for j in range(n_paths_chunk)
                    ])
            else:
                lambda_drift = 0.0
            
            drift_v = self.heston.kappa * (self.heston.theta - v_current) + lambda_drift
            dv = drift_v * dt + self.heston.sigma * np.sqrt(v_current) * dZ
            v[:, i + 1] = np.maximum(v_current + dv, 1e-8)
            
            dS = S[:, i] * np.sqrt(v_current) * dW
            S[:, i + 1] = S[:, i] + dS
        
        return S[:, -1]  #Return only terminal values
    
    def simulate_paths(self, T, n_steps, n_paths, weights=None, is_svi=False, n_jobs=-1):
        if n_jobs == -1:
            n_jobs = N_CPUS
        
        #Sequential fallback for small jobs
        if n_jobs == 1 or n_paths < 1000:
            dt = T / n_steps
            times = np.linspace(0, T, n_steps + 1)
            
            S = np.ones((n_paths, n_steps + 1)) * self.heston.S0
            v = np.ones((n_paths, n_steps + 1)) * self.heston.v0
            
            for i in range(n_steps):
                t_curr = times[i]
                
                dW = np.random.randn(n_paths) * np.sqrt(dt)
                dZ_indep = np.random.randn(n_paths) * np.sqrt(dt)
                dZ = self.heston.rho * dW + np.sqrt(1 - self.heston.rho**2) * dZ_indep
                
                v_current = np.maximum(v[:, i], 1e-8)
                
                if weights is not None:
                    if is_svi:
                        lambda_drift = np.array([
                            self.drift_lambda_svi(t_curr, S[j, i], v_current[j], weights) 
                            for j in range(n_paths)
                        ])
                    else:
                        lambda_drift = np.array([
                            self.drift_lambda_rbf(t_curr, S[j, i], v_current[j], weights) 
                            for j in range(n_paths)
                        ])
                else:
                    lambda_drift = 0.0
                
                drift_v = self.heston.kappa * (self.heston.theta - v_current) + lambda_drift
                dv = drift_v * dt + self.heston.sigma * np.sqrt(v_current) * dZ
                v[:, i + 1] = np.maximum(v_current + dv, 1e-8)
                
                dS = S[:, i] * np.sqrt(v_current) * dW
                S[:, i + 1] = S[:, i] + dS
            
            return S, v, times
        
        #Parallel execution
        paths_per_job = n_paths // n_jobs
        remaining = n_paths % n_jobs
        
        job_sizes = [paths_per_job + (1 if i < remaining else 0) for i in range(n_jobs)]
        
        args_list = [
            (size, T, n_steps, weights, is_svi, 42 + i) 
            for i, size in enumerate(job_sizes) if size > 0
        ]
        
        with Pool(n_jobs) as pool:
            results = pool.map(self._simulate_single_path_batch, args_list)
        
        #Combine results
        S_T = np.concatenate(results)
        
        #Return in expected format
        S_full = np.zeros((n_paths, 2))
        S_full[:, 0] = self.heston.S0
        S_full[:, 1] = S_T
        
        v_full = np.ones((n_paths, 2)) * self.heston.v0
        times = np.array([0, T])
        
        return S_full, v_full, times
    
    #Price calls with drift
    def price_calls(self, strikes, T, n_paths=5000, weights=None, is_svi=False, n_jobs=-1):
        S, _, _ = self.simulate_paths(T, 100, n_paths, weights, is_svi, n_jobs=n_jobs)
        S_T = S[:, -1]
        
        prices = []
        for K in strikes:
            payoff = np.maximum(S_T - K, 0)
            prices.append(np.mean(payoff))
        
        return np.array(prices)

#Particle Filter

class ParticleFilterCalibration:
    def __init__(self, heston_model, n_particles=100, n_weights=15):
        self.heston_drift = HestonWithDrift(heston_model, n_weights)
        self.n_particles = n_particles
        self.n_weights = n_weights
        self.particles = None
        self.weights = None
    
    #Initialize particles from prior
    def initialize_prior(self):
        self.particles = np.random.randn(self.n_particles, self.n_weights) * 0.1
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    #ESS = (Σw_i)² / Σw_i²
    def effective_sample_size(self):
        return (np.sum(self.weights)**2) / np.sum(self.weights**2)
    
    def _compute_likelihood(self, args):
        i, drift_weights, market_price, strike, T = args
        model_price = self.heston_drift.price_calls(
            [strike], T, n_paths=1000, weights=drift_weights, n_jobs=1
        )[0]
        return norm.pdf(market_price, loc=model_price, scale=0.3)
    
    #Prepare arguments for parallel computation
    def reweight(self, market_price, strike, T, noise_std=0.3):
        args_list = [
            (i, self.particles[i], market_price, strike, T) 
            for i in range(self.n_particles)
        ]
        
        #Parallel likelihood computation
        with Pool(min(N_CPUS, self.n_particles)) as pool:
            likelihoods = pool.map(self._compute_likelihood, args_list)
        
        likelihoods = np.array(likelihoods)
        
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
    
    #Calibrate drift via Particle Filter
    def calibrate(self, strikes, market_prices, T):
        self.initialize_prior()
        
        for K, C_mkt in zip(strikes, market_prices):
            self.reweight(C_mkt, K, T, noise_std=0.3)
            if self.resample(threshold=0.5):
                self.move(step_size=0.02)
        
        calibrated_drift = np.average(self.particles, weights=self.weights, axis=0)
        return calibrated_drift

#Gaussian Process

class GaussianProcessCalibration:
    def __init__(self, heston_model, n_weights=15, n_initial=20, n_iterations=50):
        self.heston_drift = HestonWithDrift(heston_model, n_weights)
        self.n_weights = n_weights
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-4, n_restarts_optimizer=5, normalize_y=True
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
    
    #Latin Hypercube sampling
    def sample_prior(self, n_samples):
        sampler = LatinHypercube(d=self.n_weights, seed=42)
        samples_norm = sampler.random(n=n_samples)
        return self.denormalize_weights(samples_norm)
    
    #MSE loss
    def objective(self, weights, strikes, market_prices, T):
        try:
            model_prices = self.heston_drift.price_calls(
                strikes, T, n_paths=3000, weights=weights
            )
            loss = np.mean((model_prices - market_prices)**2)
        except:
            loss = 1e6
        return loss
    
    #EI acquisition function
    def expected_improvement(self, weights_norm, y_best):
        mu, sigma = self.gp.predict(weights_norm.reshape(1, -1), return_std=True)
        mu = mu[0]
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
        
        for _ in range(5):
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
    def calibrate(self, strikes, market_prices, T):
        #Initial sampling
        initial_samples = self.sample_prior(self.n_initial)
        
        for weights in initial_samples:
            loss = self.objective(weights, strikes, market_prices, T)
            self.X_samples.append(self.normalize_weights(weights))
            self.y_samples.append(loss)
        
        X_train = np.array(self.X_samples)
        y_train = np.array(self.y_samples)
        
        #BO iterations
        for _ in range(self.n_iterations):
            self.gp.fit(X_train, y_train)
            
            y_best = np.min(y_train)
            next_weights = self.select_next_sample(y_best)
            next_loss = self.objective(next_weights, strikes, market_prices, T)
            
            X_train = np.vstack([X_train, self.normalize_weights(next_weights)])
            y_train = np.append(y_train, next_loss)
        
        best_idx = np.argmin(y_train)
        best_weights = self.denormalize_weights(X_train[best_idx])
        
        return best_weights

#SVI Parametrization

class SVICalibration:
    def __init__(self, heston_model):
        self.heston_drift = HestonWithDrift(heston_model, n_weights=5)
        self.svi_params = None
    
    def calibrate(self, strikes, market_prices, T):
        initial_guess = [0.0, 0.1, -0.1, 0.0, 0.3]
        
        bounds = [
            (-0.2, 0.2),   #a
            (0.01, 1.0),   #b
            (-0.9, 0.9),   #ρ
            (-1.0, 1.0),   #m
            (0.05, 2.0)    #σ
        ]
        
        def objective(params):
            try:
                model_prices = self.heston_drift.price_calls(
                    strikes, T, n_paths=3000, weights=params, is_svi=True
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
        
        result = minimize(
            objective, x0=initial_guess, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        #Fallback to global optimization
        if not result.success or result.fun > 0.1:
            result = differential_evolution(
                objective, bounds=bounds, maxiter=50, seed=42, polish=True
            )
        
        self.svi_params = result.x
        return self.svi_params


#MSB using potential optimization only
class MSBCalibration:
    def __init__(self, heston_model):
        self.heston = heston_model
    
    def potential(self, S, weights, strikes):
        f = np.zeros_like(S)
        for i, K in enumerate(strikes):
            f += weights[i] * np.maximum(S - K, 0)
        return np.clip(f, -50, 50)
    
    def objective(self, weights, strikes, market_prices, T):
        market_term = np.sum(weights * market_prices)
        
        S, _, _ = self.heston.simulate_paths(T=T, n_steps=30, n_paths=1000)
        S_T = S[:, -1]
        
        f_vals = self.potential(S_T, weights, strikes)
        u_value = -np.log(np.mean(np.exp(-f_vals)) + 1e-10)
        
        regularization = 0.1 * np.sum(weights**2)
        
        return -(market_term - u_value) + regularization
    
    #Optimize Schrödinger potential
    def calibrate(self, strikes, market_prices, T):
        n_strikes = len(strikes)
        bounds = [(-3, 3) for _ in range(n_strikes)]
        
        result = minimize(
            lambda w: self.objective(w, strikes, market_prices, T),
            x0=np.zeros(n_strikes),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30}
        )
        
        return result.x

#Results

@dataclass
class MethodResults:
    name: str
    drift_weights: np.ndarray
    model_prices: np.ndarray
    calibration_time: float
    mae: float
    rmse: float
    brenier_time: float
    prediction_mse_return: float
    prediction_mse_vol: float
    w2_error: float

#Comparison Framework

def load_data(maturity_idx=3):
    data_file = find_data_file('unified_heston_prediction_data.npz')
    if data_file is None:
        raise FileNotFoundError("Data file not found")
    
    data = np.load(data_file, allow_pickle=True)
    
    strikes_norm = data['strikes_norm']
    maturities = data['maturities']
    vol_surface = data['vol_surface']
    heston_params = data['heston_params'].item()
    
    X1_full = data['full_X1']
    X2_full = data['full_X2']
    X1_raw = data['full_X1_raw']
    lookback = int(data['full_lookback'])
    
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
        'X2': X2_full[:, :2],  #Only return and vol
        'X1_raw': X1_raw,
        'lookback': lookback
    }


def run_comparison(data: Dict) -> List[MethodResults]:   
    strikes = data['strikes']
    market_prices = data['market_prices']
    T = data['T']
    heston_params = data['heston_params']
    X1 = data['X1']
    X2 = data['X2']
    
    heston_model = HestonUnified(**heston_params)
    
    results = []
    d1 = 7  #Features for Brenier
    
    #Prepare test set
    n_test = 200
    test_idx = np.random.choice(len(X1), n_test, replace=False)
    X1_test = X1[test_idx, -d1:]
    X2_test = X2[test_idx]
   
    print("Method 1: Particle Filter")
    
    start = time.time()
    pf = ParticleFilterCalibration(heston_model, n_particles=100, n_weights=15)
    pf_weights = pf.calibrate(strikes, market_prices, T)
    pf_time = time.time() - start
    
    pf_model_prices = pf.heston_drift.price_calls(
        strikes, T, n_paths=5000, weights=pf_weights, n_jobs=N_CPUS
    )
    
    pf_mae = np.mean(np.abs(pf_model_prices - market_prices))
    pf_rmse = np.sqrt(np.mean((pf_model_prices - market_prices)**2))
    
    print(f"Calibration: {pf_time:.1f}s, MAE={pf_mae:.4f}, RMSE={pf_rmse:.4f}")

    #Brenier prediction
    start = time.time()
    brenier_pf = ConditionalBrenierEstimator(d1=d1, d2=2, t=0.02, epsilon=0.004)
    brenier_pf.fit(X1[:, -d1:], X2)
    pf_brenier_time = time.time() - start
    
    pf_predictions = np.array([brenier_pf.predict(x1) for x1 in X1_test])
    pf_mse_return = np.mean((pf_predictions[:, 0] - X2_test[:, 0])**2)
    pf_mse_vol = np.mean((pf_predictions[:, 1] - X2_test[:, 1])**2)
    pf_w2 = brenier_pf.compute_wasserstein_error(X1_test, X2_test)
    
    print(f"Prediction: {pf_brenier_time:.1f}s, Return MSE={pf_mse_return:.6f}, W2={pf_w2:.6f}")
    
    results.append(MethodResults(
        name="Particle Filter",
        drift_weights=pf_weights,
        model_prices=pf_model_prices,
        calibration_time=pf_time,
        mae=pf_mae,
        rmse=pf_rmse,
        brenier_time=pf_brenier_time,
        prediction_mse_return=pf_mse_return,
        prediction_mse_vol=pf_mse_vol,
        w2_error=pf_w2
    ))
    
    print("Method 2: Gaussian Process")
    
    start = time.time()
    gp = GaussianProcessCalibration(heston_model, n_weights=15, n_initial=20, n_iterations=50)
    gp_weights = gp.calibrate(strikes, market_prices, T)
    gp_time = time.time() - start
    
    gp_model_prices = gp.heston_drift.price_calls(
        strikes, T, n_paths=5000, weights=gp_weights, n_jobs=N_CPUS
    )
    
    gp_mae = np.mean(np.abs(gp_model_prices - market_prices))
    gp_rmse = np.sqrt(np.mean((gp_model_prices - market_prices)**2))
    
    print(f"Calibration: {gp_time:.1f}s, MAE={gp_mae:.4f}, RMSE={gp_rmse:.4f}")
    
    # Brenier prediction
    start = time.time()
    brenier_gp = ConditionalBrenierEstimator(d1=d1, d2=2, t=0.02, epsilon=0.004)
    brenier_gp.fit(X1[:, -d1:], X2)
    gp_brenier_time = time.time() - start
    
    gp_predictions = np.array([brenier_gp.predict(x1) for x1 in X1_test])
    gp_mse_return = np.mean((gp_predictions[:, 0] - X2_test[:, 0])**2)
    gp_mse_vol = np.mean((gp_predictions[:, 1] - X2_test[:, 1])**2)
    gp_w2 = brenier_gp.compute_wasserstein_error(X1_test, X2_test)
    
    print(f"Prediction: {gp_brenier_time:.1f}s, Return MSE={gp_mse_return:.6f}, W2={gp_w2:.6f}")
    
    results.append(MethodResults(
        name="GP Bayesian Opt",
        drift_weights=gp_weights,
        model_prices=gp_model_prices,
        calibration_time=gp_time,
        mae=gp_mae,
        rmse=gp_rmse,
        brenier_time=gp_brenier_time,
        prediction_mse_return=gp_mse_return,
        prediction_mse_vol=gp_mse_vol,
        w2_error=gp_w2
    ))
    
    print("Method 3: SVI")
    
    start = time.time()
    svi = SVICalibration(heston_model)
    svi_params = svi.calibrate(strikes, market_prices, T)
    svi_time = time.time() - start
    
    svi_model_prices = svi.heston_drift.price_calls(
        strikes, T, n_paths=5000, weights=svi_params, is_svi=True, n_jobs=N_CPUS
    )
    
    svi_mae = np.mean(np.abs(svi_model_prices - market_prices))
    svi_rmse = np.sqrt(np.mean((svi_model_prices - market_prices)**2))
    
    print(f"Calibration: {svi_time:.1f}s, MAE={svi_mae:.4f}, RMSE={svi_rmse:.4f}")
    
    #Brenier prediction
    start = time.time()
    brenier_svi = ConditionalBrenierEstimator(d1=d1, d2=2, t=0.02, epsilon=0.004)
    brenier_svi.fit(X1[:, -d1:], X2)
    svi_brenier_time = time.time() - start
    
    svi_predictions = np.array([brenier_svi.predict(x1) for x1 in X1_test])
    svi_mse_return = np.mean((svi_predictions[:, 0] - X2_test[:, 0])**2)
    svi_mse_vol = np.mean((svi_predictions[:, 1] - X2_test[:, 1])**2)
    svi_w2 = brenier_svi.compute_wasserstein_error(X1_test, X2_test)
    
    print(f"Prediction: {svi_brenier_time:.1f}s, Return MSE={svi_mse_return:.6f}, W2={svi_w2:.6f}")
    
    results.append(MethodResults(
        name="SVI",
        drift_weights=svi_params,
        model_prices=svi_model_prices,
        calibration_time=svi_time,
        mae=svi_mae,
        rmse=svi_rmse,
        brenier_time=svi_brenier_time,
        prediction_mse_return=svi_mse_return,
        prediction_mse_vol=svi_mse_vol,
        w2_error=svi_w2
    ))
    
    #Methof 4: MSB

    print("Method 4: Martingale Schodinger Bridge")
    
    start = time.time()
    msb = MSBCalibration(heston_model)
    msb_weights = msb.calibrate(strikes, market_prices, T)
    msb_time = time.time() - start

    S, _, _ = heston_model.simulate_paths(T=T, n_steps=100, n_paths=5000)
    S_T = S[:, -1]
    msb_model_prices = []
    for K in strikes:
        payoff = np.maximum(S_T - K, 0)
        msb_model_prices.append(np.mean(payoff))
    msb_model_prices = np.array(msb_model_prices)
    
    msb_mae = np.mean(np.abs(msb_model_prices - market_prices))
    msb_rmse = np.sqrt(np.mean((msb_model_prices - market_prices)**2))
    
    print(f"Calibration: {msb_time:.1f}s, MAE={msb_mae:.4f}, RMSE={msb_rmse:.4f}")
    
    #Brenier prediction
    start = time.time()
    brenier_msb = ConditionalBrenierEstimator(d1=d1, d2=2, t=0.02, epsilon=0.004)
    brenier_msb.fit(X1[:, -d1:], X2)
    msb_brenier_time = time.time() - start
    
    msb_predictions = np.array([brenier_msb.predict(x1) for x1 in X1_test])
    msb_mse_return = np.mean((msb_predictions[:, 0] - X2_test[:, 0])**2)
    msb_mse_vol = np.mean((msb_predictions[:, 1] - X2_test[:, 1])**2)
    msb_w2 = brenier_msb.compute_wasserstein_error(X1_test, X2_test)
    
    print(f"Prediction: {msb_brenier_time:.1f}s, Return MSE={msb_mse_return:.6f}, W2={msb_w2:.6f}")
    
    results.append(MethodResults(
        name="MSB",
        drift_weights=msb_weights,
        model_prices=msb_model_prices,
        calibration_time=msb_time,
        mae=msb_mae,
        rmse=msb_rmse,
        brenier_time=msb_brenier_time,
        prediction_mse_return=msb_mse_return,
        prediction_mse_vol=msb_mse_vol,
        w2_error=msb_w2
    ))
    
    return results


def visualize_comparison(results: List[MethodResults], data: Dict):
    strikes = data['strikes']
    market_prices = data['market_prices']
    market_vols = data['market_vols']
    T = data['T']
    S0 = data['heston_params']['S0']
    
    fig = plt.figure(figsize=(20, 12))
    
    #Plot 1: Price fits
    ax1 = plt.subplot(3, 4, 1)
    colors = ['blue', 'orange', 'green', 'red']
    markers = ['o', 's', '^', 'D']
    
    ax1.plot(strikes, market_prices, 'k*-', linewidth=3, markersize=12, 
             label='Market', zorder=10)
    
    for i, result in enumerate(results):
        ax1.plot(strikes, result.model_prices, 
                marker=markers[i], linestyle='--', linewidth=2, 
                markersize=8, alpha=0.7, color=colors[i],
                label=result.name)
    
    ax1.set_xlabel('Strike', fontsize=11)
    ax1.set_ylabel('Call Price', fontsize=11)
    ax1.set_title('Calibration: Price Fit', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    #Plot 2: Calibration errors
    ax2 = plt.subplot(3, 4, 2)
    x = np.arange(len(results))
    width = 0.35
    
    mae_values = [r.mae for r in results]
    rmse_values = [r.rmse for r in results]
    
    ax2.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8, color='coral')
    
    ax2.set_ylabel('Error', fontsize=11)
    ax2.set_title('Calibration Errors', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    #Plot 3: Calibration times
    ax3 = plt.subplot(3, 4, 3)
    times = [r.calibration_time for r in results]
    bars = ax3.bar(range(len(results)), times, alpha=0.7, color=colors)
    
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Calibration Speed', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    #Plot 4: IV comparison
    ax4 = plt.subplot(3, 4, 4)
    strikes_norm = strikes / S0
    
    ax4.plot(strikes_norm, market_vols*100, 'k*-', linewidth=3, 
             markersize=12, label='Market', zorder=10)
    
    for i, result in enumerate(results):
        model_vols = np.array([
            black_scholes_iv(S0, K, T, price)
            for K, price in zip(strikes, result.model_prices)
        ])
        ax4.plot(strikes_norm, model_vols*100, 
                marker=markers[i], linestyle='--', linewidth=2,
                markersize=8, alpha=0.7, color=colors[i],
                label=result.name)
    
    ax4.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Strike / Spot', fontsize=11)
    ax4.set_ylabel('Implied Volatility (%)', fontsize=11)
    ax4.set_title('Volatility Smile', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)
    
    #Plot 5: Return MSE
    ax5 = plt.subplot(3, 4, 5)
    return_mse = [r.prediction_mse_return for r in results]
    bars = ax5.bar(range(len(results)), return_mse, alpha=0.7, color=colors)
    
    for bar, mse_val in zip(bars, return_mse):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse_val:.4f}', ha='center', va='bottom', fontsize=8)
    
    ax5.set_ylabel('MSE', fontsize=11)
    ax5.set_title('Return Prediction Error', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(results)))
    ax5.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    #Plot 6: Vol MSE
    ax6 = plt.subplot(3, 4, 6)
    vol_mse = [r.prediction_mse_vol for r in results]
    bars = ax6.bar(range(len(results)), vol_mse, alpha=0.7, color=colors)
    
    for bar, mse_val in zip(bars, vol_mse):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse_val:.4f}', ha='center', va='bottom', fontsize=8)
    
    ax6.set_ylabel('MSE', fontsize=11)
    ax6.set_title('Volatility Prediction Error', fontsize=12, fontweight='bold')
    ax6.set_xticks(range(len(results)))
    ax6.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    #Plot 7: Wasserstein-2 error
    ax7 = plt.subplot(3, 4, 7)
    w2_errors = [r.w2_error for r in results]
    bars = ax7.bar(range(len(results)), w2_errors, alpha=0.7, color=colors)
    
    for bar, w2_val in zip(bars, w2_errors):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{w2_val:.4f}', ha='center', va='bottom', fontsize=8)
    
    ax7.set_ylabel('W₂ Distance', fontsize=11)
    ax7.set_title('Wasserstein-2 Error', fontsize=12, fontweight='bold')
    ax7.set_xticks(range(len(results)))
    ax7.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')
    
    #Plot 8: Brenier fitting time
    ax8 = plt.subplot(3, 4, 8)
    brenier_times = [r.brenier_time for r in results]
    bars = ax8.bar(range(len(results)), brenier_times, alpha=0.7, color=colors)
    
    for bar, time_val in zip(bars, brenier_times):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    ax8.set_ylabel('Time (seconds)', fontsize=11)
    ax8.set_title('Brenier Fitting Time', fontsize=12, fontweight='bold')
    ax8.set_xticks(range(len(results)))
    ax8.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax8.grid(True, alpha=0.3, axis='y')
    
    #Plot 9: Total time
    ax9 = plt.subplot(3, 4, 9)
    total_times = [r.calibration_time + r.brenier_time for r in results]
    bars = ax9.bar(range(len(results)), total_times, alpha=0.7, color=colors)
    
    for i, (bar, time_val) in enumerate(zip(bars, total_times)):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
        
        #Add breakdown
        calib_time = results[i].calibration_time
        brenier_time = results[i].brenier_time
        ax9.bar(i, calib_time, alpha=0.5, color='blue', width=0.8)
        ax9.bar(i, brenier_time, bottom=calib_time, alpha=0.5, 
               color='orange', width=0.8)
    
    ax9.set_ylabel('Time (seconds)', fontsize=11)
    ax9.set_title('Total Computation Time\n(Blue: Calib, Orange: Brenier)', 
                  fontsize=11, fontweight='bold')
    ax9.set_xticks(range(len(results)))
    ax9.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax9.grid(True, alpha=0.3, axis='y')
    
    #Plot 10: Normalized performance
    ax10 = plt.subplot(3, 4, 10)
    
    #Normalize metrics
    mae_norm = np.array([r.mae for r in results]) / np.min([r.mae for r in results])
    time_norm = np.array(total_times) / np.min(total_times)
    pred_norm = np.array([r.prediction_mse_return for r in results]) / \
                np.min([r.prediction_mse_return for r in results])
    
    x = np.arange(len(results))
    width = 0.25
    
    ax10.bar(x - width, mae_norm, width, label='Calib MAE', alpha=0.8, color='skyblue')
    ax10.bar(x, time_norm, width, label='Total Time', alpha=0.8, color='coral')
    ax10.bar(x + width, pred_norm, width, label='Pred MSE', alpha=0.8, color='lightgreen')
    
    ax10.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax10.set_ylabel('Normalized Score\n(1.0 = best)', fontsize=11)
    ax10.set_title('Normalized Performance\n(Lower is Better)', 
                   fontsize=11, fontweight='bold')
    ax10.set_xticks(x)
    ax10.set_xticklabels([r.name for r in results], rotation=15, ha='right')
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3, axis='y')
    
    #Plot 11: Radar chart
    ax11 = plt.subplot(3, 4, 11, projection='polar')
    
    categories = ['Calib\nAccuracy', 'Calib\nSpeed', 'Pred\nAccuracy', 
                  'Pred\nSpeed', 'Overall']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for i, result in enumerate(results):
        calib_acc = 1 / (1 + result.mae)
        calib_speed = 1 / (1 + result.calibration_time / 10)
        pred_acc = 1 / (1 + result.prediction_mse_return * 100)
        pred_speed = 1 / (1 + result.brenier_time / 10)
        overall = (calib_acc + calib_speed + pred_acc + pred_speed) / 4
        
        values = [calib_acc, calib_speed, pred_acc, pred_speed, overall]
        values += values[:1]
        
        ax11.plot(angles, values, 'o-', linewidth=2, label=result.name, 
                 color=colors[i], alpha=0.7)
        ax11.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax11.set_xticks(angles[:-1])
    ax11.set_xticklabels(categories, fontsize=9)
    ax11.set_ylim(0, 1)
    ax11.set_title('Multi-Dimensional\nPerformance', fontsize=11, fontweight='bold')
    ax11.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax11.grid(True)
    
    # Plot 12: Summary table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = "Comparative Summary\n" + "="*50 + "\n\n"
    
    for i, result in enumerate(results):
        summary_text += f"{i+1}. {result.name}:\n"
        summary_text += f"   Calibration: MAE={result.mae:.4f}, "
        summary_text += f"Time={result.calibration_time:.1f}s\n"
        summary_text += f"   Prediction: Return MSE={result.prediction_mse_return:.6f}\n"
        summary_text += f"   W₂ Error: {result.w2_error:.6f}\n"
        summary_text += f"   Total Time: {result.calibration_time + result.brenier_time:.1f}s\n\n"
    
    #Best performers
    best_calib = min(results, key=lambda r: r.mae)
    best_pred = min(results, key=lambda r: r.prediction_mse_return)
    fastest = min(results, key=lambda r: r.calibration_time + r.brenier_time)
    
    summary_text += "Best Performers:\n"
    summary_text += f"• Best Calibration: {best_calib.name}\n"
    summary_text += f"• Best Prediction: {best_pred.name}\n"
    summary_text += f"• Fastest: {fastest.name}\n\n"
    
    ax12.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
              verticalalignment='top', transform=ax12.transAxes)
    
    output_file = 'comparative_analysis_all_methods.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_file}")
    
    return fig


def print_detailed_comparison(results: List[MethodResults]):   

    print("Comparison Table")
    
    #Header
    print(f"{'Method':<20} {'Calib MAE':<12} {'Calib Time':<12} "
          f"{'Pred MSE':<12} {'W₂ Error':<12} {'Total Time':<12}")
    print(f"{'-'*80}")
    
    #Data rows
    for result in results:
        total_time = result.calibration_time + result.brenier_time
        print(f"{result.name:<20} {result.mae:<12.4f} {result.calibration_time:<12.1f} "
              f"{result.prediction_mse_return:<12.6f} {result.w2_error:<12.6f} {total_time:<12.1f}")
    
    print("Rankings")

    #Rankings
    sorted_by_calib = sorted(results, key=lambda r: r.mae)
    sorted_by_pred = sorted(results, key=lambda r: r.prediction_mse_return)
    sorted_by_time = sorted(results, key=lambda r: r.calibration_time + r.brenier_time)
    sorted_by_w2 = sorted(results, key=lambda r: r.w2_error)
    
    print("Best Calibration Accuracy:")
    for i, r in enumerate(sorted_by_calib[:3], 1):
        print(f"  {i}. {r.name:<20} (MAE = {r.mae:.4f})")
    
    print("\nBest Prediction Accuracy:")
    for i, r in enumerate(sorted_by_pred[:3], 1):
        print(f"  {i}. {r.name:<20} (MSE = {r.prediction_mse_return:.6f})")
    
    print("\nFastest Methods:")
    for i, r in enumerate(sorted_by_time[:3], 1):
        total = r.calibration_time + r.brenier_time
        print(f"  {i}. {r.name:<20} (Total = {total:.1f}s)")
    
    print("\nBest W₂ Distance:")
    for i, r in enumerate(sorted_by_w2[:3], 1):
        print(f"  {i}. {r.name:<20} (W₂ = {r.w2_error:.6f})")

#Main Execution
if __name__ == "__main__":
    np.random.seed(42)

    #Load data
    try:
        data = load_data(maturity_idx=3)
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\n Please run generate_synthetic_data.py first")
        sys.exit(1)
    
    print(f"\n Data loaded:")
    print(f"Strikes: {len(data['strikes'])}")
    print(f"Maturity: T={data['T']:.2f}Y")
    print(f"Training samples: {len(data['X1'])}")
    print(f"Features: {data['X1'].shape[1]}")
    
    #Run comparison
    print(f"\n Starting comparison")
    results = run_comparison(data)
    
    #Print comparison
    print_detailed_comparison(results)
    
    #Visualize
    print(f"\n Generating visualizations")
    visualize_comparison(results, data)
    
    #Final summary
    print("Experiment Complete")
    
    print("Results saved to: comparative_analysis_all_methods.png")
    plt.show()
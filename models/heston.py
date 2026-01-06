import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, Optional

import sys
sys.path.append('..')

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#Unified Heston stochastic volatility model
#Naked Model (P₀ measure):
#dS_t = S_t a_t dW¹_t
#da_t = κ(θ - a_t)dt + σ√a_t dZ_t
#d⟨W¹, Z⟩_t = ρdt
#Calibrated Model (P measure):
#dS_t = S_t a_t dW_t  (still martingale)
#da_t = (κ(θ - a_t) + λ(t,S,a))dt + σ√a_t dZ_t
#Key: Drift λ added ONLY to variance, preserving vol-of-vol σ.
    
#Parameters:
#S0: float - Initial asset price
#v0: float - Initial variance (not volatility!)
#kappa: float - Mean reversion speed
#theta: float -Long-term variance
#sigma: float - Volatility of volatility (vol-of-vol)
#rho: float -Correlation between price and variance innovations
#r: float - Risk-free rate
#mode: str - 'financial':  option pricing, 'conditional': Baptista et al. conditional simulation
    
class HestonUnified:
    def __init__(self, 
                 S0: float = 100, 
                 v0: float = 0.04, 
                 kappa: float = 2.0, 
                 theta: float = 0.04, 
                 sigma: float = 0.3,
                 rho: float = -0.7, 
                 r: float = 0.0,
                 mode: str = 'financial'):
        
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.mode = mode
        
        #Calibration storage
        self.potentials = {}  #f*(K) - Schrödinger potentials
        self.drift_corrections = {}  #λ(t,S,v) - drift on variance
        
        #Conditional simulation storage
        self.entropic_maps = {}  #T̂_ε,t for conditional sampling
        self.rescaling_params = {}  #(t, ε, A_t) parameters
  
    #Naked Model Simulation (P₀ measure)
    #This is the baseline model before any calibration

    #Parameters
    #T: float - Time horizon
    #n_steps: int - Number of time steps
    #n_paths: int - Number of Monte Carlo paths
            
    #Returns
    #S: ndarray, shape (n_paths, n_steps+1) Stock price paths
    #v: ndarray, shape (n_paths, n_steps+1) Variance paths
    #t: ndarray, shape (n_steps+1,) Time grid

    def simulate_paths(self, T: float = 1.0, n_steps: int = 252, 
                      n_paths: int = 10000, seed: Optional[int] = 42) -> Tuple:
 
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(n_steps):
            #Correlated Brownian motions
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)
            
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            #Euler-Maruyama for variance with Feller correction
            v_current = np.maximum(v[:, i], 0)
            
            v[:, i+1] = v_current + self.kappa * (self.theta - v_current) * dt \
                        + self.sigma * np.sqrt(v_current * dt) * W2
            
            #Stock dynamics (log-normal discretization)
            S[:, i+1] = S[:, i] * np.exp(
                (self.r - 0.5 * v_current) * dt + np.sqrt(v_current * dt) * W1
            )
        
        return S, v, t
    
    
    #Martingale Schrodinger Bridge 
    
    #Calibrate Schrödinger potential f* to match market vanilla prices
    #P₁ = sup_{f,Δ} { -E_μ[f] - ln E_P₀[exp(-f(S_T) - ∫Δ_s dS_s)] }
    #This is equivalent to entropy minimization with martingale constraint
    #Returns
    #potential: ndarray, shape (N,) Calibrated potential f*(K)
    #weights: ndarray, shape (N,) Decomposition weights: f(s) = Σ ω_α (s - K_α)⁺
    
    def calibrate_schrodinger_potential(self, 
                                       market_prices: np.ndarray,
                                       strikes: np.ndarray,
                                       T: float,
                                       max_iter: int = 100,
                                       tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        
        N = len(strikes)
        
        #Potential decomposition over calls 
        #Negative dual objective (for minimization).
        #Computes: E_μ[f] + ln E_P₀[exp(-f(S_T) - ∫Δ*dS)]

        def objective(omega):
            #Construct potential from weights
            f_vals = np.zeros(N)
            for i, K in enumerate(strikes):
                f_vals[i] = np.sum(omega * np.maximum(strikes - K, 0))
            
            #Market term: E_μ[f] where μ is risk-neutral density
            market_term = np.sum(f_vals * market_prices / np.sum(market_prices))
            
            #Simulation term: ln E_P₀[exp(-f(S_T))]
            n_mc = 10000
            S, _, _ = self.simulate_paths(T=T, n_paths=n_mc, seed=42)
            S_T = S[:, -1]
            
            #Interpolate potential at terminal prices
            f_ST = np.interp(S_T, strikes, f_vals, left=f_vals[0], right=f_vals[-1])
            
            #Log expectation 
            max_val = np.max(-f_ST)
            log_expectation = max_val + np.log(np.mean(np.exp(-f_ST - max_val)))
            
            return market_term + log_expectation
        
        #Optimize over call weights
        omega0 = np.zeros(N)
        result = minimize(objective, omega0, method='BFGS',
                         options={'maxiter': max_iter, 'gtol': tol})
        
        weights = result.x
        
        #Compute final potential
        potential = np.zeros(N)
        for i, K in enumerate(strikes):
            potential[i] = np.sum(weights * np.maximum(strikes - K, 0))
        
        self.potentials[T] = {
            'strikes': strikes,
            'values': potential,
            'weights': weights,
            'market_prices': market_prices
        }
        
        return potential, weights
    
        #Compute drift correction λ(t,S,v) for the variance process.
        #Calibrated variance: da_t = (κ(θ-a) + λ(t,S,a))dt + σ√a dZ_t
        #where λ = (1-ρ²)σ² ∂_v ln E_P₀[exp(-f*(S_T) - ∫Δ*dS)|S_t,v_t]
        
        #Drift added only to variance, not to price
        #This preserves the vol-of-vol structure σ 
        
        #Parameters
        #t: float Current time
        #S: float Current stock price
        #v: float Current variance
        #T: float Target maturity
            
        #Returns
        #lambda_drift: float - Drift correction for variance process
    def compute_drift_correction(self, t: float, S: float, v: float, T: float) -> float:
        if T not in self.potentials:
            return 0.0
        
        dt = T - t
        if dt <= 0:
            return 0.0
        
        #Finite difference approximation of ∂_v ln E[...]
        epsilon = max(1e-4, 0.01 * v)
        
        #Simulate forward conditionally - simplified MC approximation
        n_mc = 1000
        n_steps = max(10, int(dt * 252))
        dt_step = dt / n_steps
        
        #Compute E_P₀[exp(-f*(S_T) - ∫Δ*dS)|S_t, v_t]
        def conditional_expectation(v_init):
            S_path = np.full((n_mc, n_steps + 1), S)
            v_path = np.full((n_mc, n_steps + 1), v_init)
            
            for i in range(n_steps):
                Z1 = np.random.randn(n_mc)
                Z2 = np.random.randn(n_mc)
                
                W1 = Z1
                W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
                
                v_curr = np.maximum(v_path[:, i], 0)
                
                v_path[:, i+1] = v_curr + self.kappa * (self.theta - v_curr) * dt_step \
                                + self.sigma * np.sqrt(v_curr * dt_step) * W2
                
                S_path[:, i+1] = S_path[:, i] * np.exp(
                    (self.r - 0.5 * v_curr) * dt_step + np.sqrt(v_curr * dt_step) * W1
                )
            
            return S_path[:, -1]
        
        #Evaluate at v and v + ε
        potential = self.potentials[T]
        
        S_T = conditional_expectation(v)
        f_vals = np.interp(S_T, potential['strikes'], potential['values'])
        E_v = np.mean(np.exp(-f_vals))
        
        S_T_plus = conditional_expectation(v + epsilon)
        f_vals_plus = np.interp(S_T_plus, potential['strikes'], potential['values'])
        E_v_plus = np.mean(np.exp(-f_vals_plus))
        
        #Finite difference derivative
        if E_v > 1e-10 and E_v_plus > 1e-10:
            d_log_E = (np.log(E_v_plus) - np.log(E_v)) / epsilon
        else:
            d_log_E = 0.0
        
        #Drift correction
        lambda_drift = (1 - self.rho**2) * self.sigma**2 * d_log_E
        
        return lambda_drift
    
   
    #Simulate paths under calibrated measure P (with drift λ)
    #Returns
    #S, v, t: ndarrays - Calibrated price/variance paths
    
    def simulate_calibrated_paths(self, T: float = 1.0, n_steps: int = 252, 
                                 n_paths: int = 10000) -> Tuple:
        
        if T not in self.potentials:
            raise ValueError(f"Model not calibrated for maturity T={T}"
                           "Call calibrate_schrodinger_potential first")
        
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(n_steps):
            Z1 = np.random.randn(n_paths)
            Z2 = np.random.randn(n_paths)
            
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            v_curr = np.maximum(v[:, i], 0)
            
            #Compute drift correction for each path
            lambda_drift = np.array([
                self.compute_drift_correction(t[i], S[j, i], v_curr[j], T)
                for j in range(n_paths)
            ])
            
            #Calibrated variance dynamics with drift correction
            v[:, i+1] = v_curr + (self.kappa * (self.theta - v_curr) + lambda_drift) * dt \
                        + self.sigma * np.sqrt(v_curr * dt) * W2
            
            S[:, i+1] = S[:, i] * np.exp(
                (self.r - 0.5 * v_curr) * dt + np.sqrt(v_curr * dt) * W1
            )
        
        return S, v, t
    
    
    #Conditional Entropic Brenier Maps 

    #Fit conditional Brenier map T_CB via entropic OT with rescaled cost
    #Algorithm 
    #Scale data with A_t = diag(1_d₁, √t·1_d₂)
    #Solve entropic OT with ε-regularization (Sinkhorn)
    #Return barycentric map T̂_ε,t(x) = E_π_ε[Y|X=x]

    #Parameters
    #X_samples: ndarray, shape (n, d) - Source samples 
    #Y_samples: ndarray, shape (n, d)

    #Target samples
    #d1: int - Dimension of conditioning variables
    #d2: int - Dimension of parameter/response variables
    #t: float, optional - Rescaling parameter (if None, use t ∝ n^(-1/3))
    #epsilon: float, optional
    #Entropic regularization (if None, use ε ∝ t²)
    #method: str
    #'theory': Use theoretical rates from Theorem 4.2
    #'practical': Use conservative rates from experiments
            
    #Returns
    #T_hat: callable
    #Conditional Brenier map T̂: x → conditional sample
    
    def fit_conditional_brenier_map(self,
                                   X_samples: np.ndarray,
                                   Y_samples: np.ndarray,
                                   d1: int,
                                   d2: int,
                                   t: Optional[float] = None,
                                   epsilon: Optional[float] = None,
                                   method: str = 'theory') -> Callable:
        
        n = X_samples.shape[0]
        
        #Auto-select parameters if not provided -. Theorem 4.2)
        if t is None:
            if method == 'theory':
                t = 0.1 * (n ** (-1/3))  #Theorem 4.2
            else:
                t = 0.06  #Practical choice from experiments (Sec 5.1.1)
        
        if epsilon is None:
            if method == 'theory':
                epsilon = t ** 2  #Theorem 4.7
            else:
                epsilon = t / 5  
        
        #Compute rescaled cost matrix
        from metrics import compute_rescaled_cost_matrix, sinkhorn_dual_potentials
        
        C, A_t = compute_rescaled_cost_matrix(X_samples, Y_samples, t, d1, d2)
        
        #Solve Sinkhorn algorithm for dual potentials
        dual_g = sinkhorn_dual_potentials(C, epsilon, max_iter=1000, tol=1e-6)
        
        #Define entropic Brenier map
        
        #Entropic Brenier map: T̂_ε,t(x) = Σᵢ Yᵢ · wᵢ(x)
        #where wᵢ(x) ∝ exp((ĝᵢ - ½‖Aₜ(x-Yᵢ)‖²)/ε)
        def T_epsilon_t(x):
            
            x = np.atleast_2d(x)
            n_query = x.shape[0]
            results = []
            
            for i in range(n_query):
                #Compute distances in rescaled space
                x_scaled = A_t @ x[i]
                Y_scaled = Y_samples @ A_t.T
                
                distances = np.sum((Y_scaled - x_scaled) ** 2, axis=1) / 2.0
                
                #Compute barycentric weights
                log_weights = (dual_g - distances) / epsilon
                log_weights -= np.max(log_weights)  # Numerical stability
                weights = np.exp(log_weights)
                weights /= np.sum(weights)
                
                #Barycentric projection
                result = np.sum(Y_samples * weights[:, np.newaxis], axis=0)
                results.append(result)
            
            return np.array(results).squeeze()
        
        #Store for analysis
        self.entropic_maps[t] = T_epsilon_t
        self.rescaling_params[t] = {'t': t, 'epsilon': epsilon, 'A_t': A_t}
        
        return T_epsilon_t
    
    #Generate conditional samples μ₂|₁(·|x₁) using fitted map
    #T₂(X₂; x₁) ~ μ₂|₁(·|T₁(x₁)) for X₂ ~ ρ₂
    #For Heston: Sample from conditional posterior given observation
        
    #Parameters
    #x_condition: ndarray - Conditioning variable (e.g., observation or price level)
    #map_func: callable - Fitted conditional Brenier map
    #n_samples: int - Number of conditional samples
            
    #Returns
    #samples: ndarray, shape (n_samples, d2) - Samples from conditional distribution
    
    def conditional_sample(self, 
                          x_condition: np.ndarray,
                          map_func: Callable,
                          n_samples: int = 1000) -> np.ndarray:
        
        #Sample from reference (e.g., standard Gaussian for parameters)
        noise_samples = np.random.randn(n_samples, x_condition.shape[0])
        
        #Apply conditional map
        conditional_samples = np.array([
            map_func(np.concatenate([x_condition, noise]))
            for noise in noise_samples
        ])
        
        return conditional_samples
    
    #Option Pricing
    
    #Price European call options via Monte Carlo
    def option_prices_mc(self, strikes: np.ndarray, T: float = 1.0, 
                        n_paths: int = 100000, calibrated: bool = False) -> np.ndarray:
       
        if calibrated:
            S, _, _ = self.simulate_calibrated_paths(T=T, n_steps=252, n_paths=n_paths)
        else:
            S, _, _ = self.simulate_paths(T=T, n_steps=252, n_paths=n_paths)
        
        S_T = S[:, -1]
        
        call_prices = np.array([
            np.exp(-self.r * T) * np.mean(np.maximum(S_T - K, 0))
            for K in strikes
        ])
        
        return call_prices


#example usage

if __name__ == "__main__":
    
    #Initialize model
    heston = HestonUnified(
        S0=100, v0=0.04, kappa=2.0, theta=0.04,
        sigma=0.3, rho=-0.7, mode='financial'
    )
    
    print("\nModel Parameters:")
    print(f"S₀ = {heston.S0}")
    print(f"v₀ = {heston.v0} (σ₀ = {np.sqrt(heston.v0)*100:.1f}%)")
    print(f"κ = {heston.kappa}, θ = {heston.theta}, σ = {heston.sigma}, ρ = {heston.rho}")
    
    #Martingale Schrödinger Bridge

    #Generate "market" prices from naked model
    strikes = np.linspace(80, 120, 10)
    T = 1.0
    market_prices = heston.option_prices_mc(strikes, T, n_paths=50000)
    
    print(f"\nCalibrating to {len(strikes)} vanilla options at T={T}")
    potential, weights = heston.calibrate_schrodinger_potential(
        market_prices, strikes, T, max_iter=50
    )
    
    print("Calibration complete!")
    print(f" Potential range: [{np.min(potential):.4f}, {np.max(potential):.4f}]")
    
    # Price with calibrated model
    calibrated_prices = heston.option_prices_mc(strikes, T, n_paths=50000, calibrated=True)
    
    print("\nCalibration Quality:")
    print(f"MAE = {np.mean(np.abs(calibrated_prices - market_prices)):.6f}")
    print(f"Max Error = {np.max(np.abs(calibrated_prices - market_prices)):.6f}")
    

    #Conditional Entropic Brenier Map 

    #Generate joint samples (X₁ = price observation, X₂ = parameter)
    n_samples = 5000
    X1 = np.random.uniform(-3, 3, (n_samples, 1))  #Conditioning variable
    X2 = np.tanh(X1) + np.random.gamma(1, 0.3, (n_samples, 1))  # arameter
    
    X_joint = np.hstack([X1, X2])
    Y_joint = X_joint.copy() 
    
    print(f"\nFitting conditional Brenier map to {n_samples} samples")
    T_hat = heston.fit_conditional_brenier_map(
        X_joint, Y_joint, d1=1, d2=1, method='practical'
    )
    
    print("Map fitted successfully!")
    
    #Generate conditional samples
    x_test = np.array([0.5])
    conditional_samples = heston.conditional_sample(x_test, T_hat, n_samples=1000)
    
    print(f"\nConditional sampling at x₁ = {x_test[0]}:")
    print(f" Mean = {np.mean(conditional_samples):.4f}")
    print(f"Std = {np.std(conditional_samples):.4f}")

    print("Both methods implemented!")

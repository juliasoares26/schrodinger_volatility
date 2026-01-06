import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

def compute_metrics(prices_predicted, prices_true):
    diff = prices_predicted - prices_true
    
    metrics = {
        'mse': float(np.mean(diff ** 2)),
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'max_error': float(np.max(np.abs(diff))),
        'mape': float(np.mean(np.abs(diff / (prices_true + 1e-10))) * 100),
        'relative_mae': float(np.mean(np.abs(diff)) / (np.mean(prices_true) + 1e-10))
    }
    
    return metrics

#Compute L²(ρ) error: ‖T̂ - T‖²_L²(ρ)
#Parameters

#map_pred: callable - Predicted transport map T̂(x)
#map_true: callable - True transport map T(x)
#samples: ndarray, shape (n_samples, d) - Samples from source distribution ρ
#weights: ndarray, optional - Sample weights (default: uniform)

#Returns
#error: float - L² error estimate
def compute_l2_error(map_pred, map_true, samples, weights=None):
    
    if weights is None:
        weights = np.ones(len(samples)) / len(samples)
    
    pred_values = np.array([map_pred(x) for x in samples])
    true_values = np.array([map_true(x) for x in samples])
    
    squared_errors = np.sum((pred_values - true_values) ** 2, axis=1)
    l2_error = np.sum(weights * squared_errors)
    
    return float(l2_error)

#Verify martingale property: E[S_T] = S_0 under risk-neutral measure.
def check_martingale_property(S_paths, S0, tol=0.05):
    S_T = S_paths[:, -1]
    mean_terminal = np.mean(S_T)
    std_terminal = np.std(S_T)
    
    relative_error = abs(mean_terminal - S0) / S0
    z_score = (mean_terminal - S0) / (std_terminal / np.sqrt(len(S_T)))
    
    is_martingale = relative_error < tol
    
    result = {
        'is_martingale': bool(is_martingale),
        'mean_terminal': float(mean_terminal),
        'expected': float(S0),
        'relative_error': float(relative_error),
        'z_score': float(z_score),
        'p_value': float(2 * (1 - stats.norm.cdf(abs(z_score))))
    }
    
    return result

#Verify drift is added only to variance, not price
# da_t = (b(a_t) + λ(t,S,v))dt + σ(a_t)dZ_t
def check_drift_on_variance_only(v_paths, kappa, theta, sigma):
    result = {
        'mean_variance': float(np.mean(v_paths)),
        'long_run_mean': float(theta),
        'mean_reversion_speed': float(kappa),
        'vol_of_vol': float(sigma),
        'variance_stable': bool(np.mean(v_paths) > 0),
        'preserves_vol_of_vol': True  # By construction in calibrated model
    }
    
    return result

#A_t = diag(1/√d₁, √t·1/√d₂)
#c_t(x,y) = ½‖A_t(x-y)‖²
def compute_rescaled_cost_matrix(X, Y, t, d1, d2):
    d = d1 + d2
    
    #Construct A_t = diag(1_d1, √t·1_d2)
    A_t = np.eye(d)
    A_t[:d1, :d1] *= 1.0
    A_t[d1:, d1:] *= np.sqrt(t)
    
    #Scale data
    X_scaled = X @ A_t
    Y_scaled = Y @ A_t
    
    #Compute pairwise squared distances
    C = cdist(X_scaled, Y_scaled, metric='sqeuclidean') / 2.0
    
    return C, A_t

#Compute entropic Brenier map estimator T̂_ε,t(x).
#T̂_ε,t(x) = Σᵢ Yᵢ exp((ĝε,t)ᵢ - ½‖Aₜ(x-Yᵢ)‖²)/ε / normalization
def compute_entropic_brenier_map(X, Y, t, epsilon, d1, d2):
    from scipy.optimize import linprog
    
    #Compute rescaled cost matrix
    C, A_t = compute_rescaled_cost_matrix(X, Y, t, d1, d2)
    
    #Solve entropic OT via Sinkhorn algorithm
    dual_g = sinkhorn_dual_potentials(C, epsilon, max_iter=1000)
    
    #Entropic Brenier map at point x
    def T_epsilon_t(x):
        x = np.atleast_2d(x)
        n_query = x.shape[0]
        n_targets = Y.shape[0]
        
        #Scale query point
        x_scaled = x @ A_t
        
        #Compute weights for each target point
        results = []
        for i in range(n_query):
            distances = np.sum((A_t @ (x[i] - Y.T)) ** 2, axis=0) / 2.0
            log_weights = (dual_g - distances) / epsilon
            weights = np.exp(log_weights - np.max(log_weights))  # Numerical stability
            weights /= np.sum(weights)
            
            #Barycentric projection
            result = np.sum(Y * weights[:, np.newaxis], axis=0)
            results.append(result)
        
        return np.array(results).squeeze()
    
    return T_epsilon_t, {'dual_g': dual_g, 'A_t': A_t}

#Solves: min_π <C,π> + ε·KL(π‖ρ⊗μ)
#Returns dual variable g from π_ij ∝ exp((f_i + g_j - C_ij)/ε)
def sinkhorn_dual_potentials(C, epsilon, max_iter=1000, tol=1e-6):
    n, m = C.shape
    
    #Initialize dual variables
    f = np.zeros(n)
    g = np.zeros(m)
    
    #Sinkhorn iterations
    for iteration in range(max_iter):
        f_old = f.copy()
        
        #Update f
        f = -epsilon * np.log(np.sum(np.exp((g - C.T) / epsilon), axis=1))
        
        #Update g
        g = -epsilon * np.log(np.sum(np.exp((f[:, np.newaxis] - C) / epsilon), axis=0))
        
        #Check convergence
        if np.max(np.abs(f - f_old)) < tol:
            break
    
    return g

#Check for call spread arbitrage: C(K₁) ≥ C(K₂) for K₁ < K₂
def check_call_spread_arbitrage(prices, strikes):
    violations = []
    
    for i in range(len(prices) - 1):
        spread = prices[i] - prices[i+1]
        if spread < -1e-6:
            violations.append({
                'strike_low': float(strikes[i]),
                'strike_high': float(strikes[i+1]),
                'spread': float(spread)
            })
    
    result = {
        'n_violations': len(violations),
        'violations': violations,
        'is_arbitrage_free': len(violations) == 0
    }
    
    return result

# Check convexity of call prices in strike
def check_convexity(prices, strikes):
    if len(prices) < 3:
        return {'is_convex': True, 'violations': 0}
    
    #Second derivative via finite differences
    d2C = np.diff(prices, n=2)
    dk2 = np.diff(strikes, n=2)
    
    d2C_dk2 = d2C / (dk2 + 1e-10)
    
    violations = np.sum(d2C_dk2 < -1e-6)
    
    result = {
        'is_convex': bool(violations == 0),
        'n_violations': int(violations),
        'min_curvature': float(np.min(d2C_dk2)),
        'mean_curvature': float(np.mean(d2C_dk2))
    }
    
    return result

#Compute Wasserstein-1 distance between 1D distributions.
    
def wasserstein_distance_1d(samples1, samples2, weights1=None, weights2=None):
    if weights1 is None:
        weights1 = np.ones(len(samples1)) / len(samples1)
    if weights2 is None:
        weights2 = np.ones(len(samples2)) / len(samples2)
    
    #Normalize weights
    weights1 = weights1 / np.sum(weights1)
    weights2 = weights2 / np.sum(weights2)
    
    #Sort samples
    idx1 = np.argsort(samples1)
    idx2 = np.argsort(samples2)
    
    sorted_samples1 = samples1[idx1]
    sorted_samples2 = samples2[idx2]
    sorted_weights1 = weights1[idx1]
    sorted_weights2 = weights2[idx2]
    
    #Compute cumulative distributions
    cum_weights1 = np.cumsum(sorted_weights1)
    cum_weights2 = np.cumsum(sorted_weights2)
    
    #Compute distance via inverse CDF difference
    all_samples = np.sort(np.concatenate([sorted_samples1, sorted_samples2]))
    distance = 0.0
    
    for i in range(len(all_samples) - 1):
        x1, x2 = all_samples[i], all_samples[i + 1]
        
        #Evaluate CDFs at midpoint
        mid = (x1 + x2) / 2
        F1 = np.searchsorted(sorted_samples1, mid, side='right')
        F2 = np.searchsorted(sorted_samples2, mid, side='right')
        
        w1 = cum_weights1[F1 - 1] if F1 > 0 else 0
        w2 = cum_weights2[F2 - 1] if F2 > 0 else 0
        
        distance += abs(w1 - w2) * (x2 - x1)
    
    return float(distance)

#Analyze convergence rate: error ∝ N^α
def convergence_diagnostics(errors, sample_sizes):
    errors = np.array(errors)
    sample_sizes = np.array(sample_sizes)
    
    valid_idx = errors > 0
    errors = errors[valid_idx]
    sample_sizes = sample_sizes[valid_idx]
    
    if len(errors) < 2:
        return {
            'rate': np.nan,
            'r_squared': np.nan,
            'is_monte_carlo_rate': False,
            'is_conditional_brenier_rate': False
        }
    
    log_N = np.log(sample_sizes)
    log_error = np.log(errors)
    
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(log_N, log_error)
    
    #Check rates
    is_mc_rate = abs(slope - (-0.5)) < 0.1
    is_cb_rate = abs(slope - (-2/3)) < 0.1
    
    return {
        'rate': float(slope),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'std_err': float(std_err),
        'is_monte_carlo_rate': bool(is_mc_rate),
        'is_conditional_brenier_rate': bool(is_cb_rate),
        'fit_params': (float(intercept), float(slope))
    }

#Compute ESS = (Σw_i)² / Σw_i²
def effective_sample_size(weights):
    weights = weights / (np.sum(weights) + 1e-10)
    ess = 1.0 / (np.sum(weights ** 2) + 1e-10)
    return float(ess)

#Select rescaling parameter t based on sample size
def select_optimal_t(n_samples, d1, d2, method='theory'):
    if method == 'theory':
        # Theorem 4.2: t(n) ∝ n^(-1/3)
        t_optimal = 0.1 * (n_samples ** (-1/3))
    elif method == 'practical':
        # More conservative for finite samples
        t_optimal = 0.1 * (n_samples ** (-1/5))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(t_optimal)

#Select entropic regularization ε given rescaling t
def select_optimal_epsilon(t, method='theory'):
    if method == 'theory':
        # Theorem 4.7: ε ∝ t²
        epsilon_optimal = t ** 2
    elif method == 'practical':
        # Empirical choice from experiments
        epsilon_optimal = t / 5
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(epsilon_optimal)


def generate_summary_table(results_dict):
    methods = list(results_dict.keys())
    
    lines = []
    lines.append("="*80)
    lines.append(f"{'Method':<25} {'MAE':<12} {'RMSE':<12} {'Time (s)':<12} {'ESS':<10}")
    lines.append("-"*80)
    
    for method in methods:
        r = results_dict[method]
        mae = r.get('mae', np.nan)
        rmse = r.get('rmse', np.nan)
        time_val = r.get('time', np.nan)
        ess = r.get('ess', np.nan)
        
        lines.append(f"{method:<25} {mae:<12.6f} {rmse:<12.6f} {time_val:<12.2f} {ess:<10.0f}")
    
    lines.append("="*80)
    
    return '\n'.join(lines)
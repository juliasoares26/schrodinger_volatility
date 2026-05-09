import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import warnings
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from multiprocessing import cpu_count
warnings.filterwarnings('ignore')

N_CPUS = cpu_count()

sys.path.insert(0, str(Path(__file__).parent.parent / 'calibration'))  # HestonUnified
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))       # brenier.py
sys.path.append(str(Path(__file__).parent))
from base import HestonUnified
from brenier import ConditionalBrenierEstimator, make_brenier_estimator

#Heston Wrapper for drift networks

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
    
    def simulate_paths(self, T, n_steps, n_paths, weights=None, is_svi=False, n_jobs=1):
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

#Particle Filter Calibration

class ParticleFilterCalibration:
    def __init__(self, heston_model, n_particles=100, n_weights=15):
        self.heston_drift = HestonWithDrift(heston_model, n_weights)
        self.n_particles = n_particles
        self.n_weights = n_weights
        self.particles = None
        self.weights = None
    
    def initialize_prior(self):
        self.particles = np.random.randn(self.n_particles, self.n_weights) * 0.1
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def effective_sample_size(self):
        return (np.sum(self.weights)**2) / np.sum(self.weights**2)
    
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

#Gaussian Process Calibration

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

#SVI Calibration

class SVICalibration:
    def __init__(self, heston_model):
        self.heston_drift = HestonWithDrift(heston_model, n_weights=5)
        self.svi_params = None

#MSB Calibration

class MSBCalibration:
    def __init__(self, heston_model):
        self.heston = heston_model
    
    def potential(self, S, weights, strikes):
        f = np.zeros_like(S)
        for i, K in enumerate(strikes):
            f += weights[i] * np.maximum(S - K, 0)
        return np.clip(f, -50, 50)

#Moment Stability Analyzer for All Methods

class MomentStabilityAnalyzer:
    def __init__(self, heston_params: Dict):
        self.heston_params = heston_params
        self.heston = HestonUnified(**{k: v for k, v in heston_params.items()
                                       if k in ('S0', 'v0', 'kappa', 'theta', 'sigma', 'rho', 'r')})
    
    #Run particle filter (used as baseline filtering method)
    def _run_particle_filter(self, S_obs, times, n_particles):
        n_steps = len(times)
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        
        #Initialize particles
        S0 = self.heston_params['S0']
        v0 = self.heston_params['v0']
        
        particles = np.zeros((n_particles, 2))
        particles[:, 0] = S0 + np.random.randn(n_particles) * 0.1
        particles[:, 1] = np.maximum(v0 + np.random.randn(n_particles) * 0.001, 1e-6)
        weights = np.ones(n_particles) / n_particles
        
        all_particles = [particles.copy()]
        all_weights = [weights.copy()]
        
        #Sequential filtering
        for t_idx in range(1, n_steps):
            #Prediction step
            particles = self._heston_transition(particles, dt)
            
            #Update step
            obs = S_obs[t_idx]
            likelihoods = np.array([
                self._heston_likelihood(particles[i], obs, dt)
                for i in range(n_particles)
            ])
            
            weights *= likelihoods
            weights /= (np.sum(weights) + 1e-10)
            
            #Resample if ESS low
            ess = (np.sum(weights)**2) / np.sum(weights**2)
            if ess < 0.5 * n_particles:
                positions = (np.arange(n_particles) + np.random.uniform()) / n_particles
                cumsum = np.cumsum(weights)
                indices = np.searchsorted(cumsum, positions)
                particles = particles[indices]
                weights = np.ones(n_particles) / n_particles
            
            all_particles.append(particles.copy())
            all_weights.append(weights.copy())
        
        return {
            'particles': all_particles,
            'weights': all_weights,
            'times': times
        }
    
    def _heston_likelihood(self, state, observation, dt):
        S, v = state[0], state[1]
        
        if isinstance(observation, np.ndarray):
            S_obs = float(observation.flat[0])
        else:
            S_obs = float(observation)
        
        log_return_mean = -0.5 * v * dt
        log_return_var = v * dt
        
        if log_return_var <= 0 or S <= 0:
            return 1e-10
        
        log_return_obs = np.log(S_obs / S)
        diff = log_return_obs - log_return_mean
        
        likelihood = (1.0 / np.sqrt(2 * np.pi * log_return_var)) * \
                     np.exp(-0.5 * diff**2 / log_return_var)
        
        if isinstance(likelihood, np.ndarray):
            likelihood = float(likelihood.item())
        else:
            likelihood = float(likelihood)
        
        return max(likelihood, 1e-10)
    
    def _heston_transition(self, particles, dt):
        N = len(particles)
        new_particles = np.zeros_like(particles)
        
        for i in range(N):
            S, v = particles[i]
            
            dW_v = np.random.randn() * np.sqrt(dt)
            v_new = v + self.heston_params['kappa'] * (self.heston_params['theta'] - v) * dt + \
                    self.heston_params['sigma'] * np.sqrt(max(v, 0)) * dW_v
            v_new = max(v_new, 1e-6)
            
            dW_S = self.heston_params['rho'] * dW_v + \
                   np.sqrt(1 - self.heston_params['rho']**2) * np.random.randn() * np.sqrt(dt)
            S_new = S * np.exp(-0.5 * v * dt + np.sqrt(v) * dW_S)
            
            new_particles[i] = [S_new, v_new]
        
        return new_particles
    
    #Generate training data for Brenier (simple synthetic data)
    def _generate_training_data(self, n_samples=1000, lookback=7, seed=None):
        X1 = []
        X2 = []
        rng = np.random.default_rng(seed)

        for i in range(n_samples):
            s1 = int(rng.integers(0, 2**31))
            s2 = int(rng.integers(0, 2**31))
            S, v, _ = self.heston.simulate_paths(T=0.1, n_steps=lookback, n_paths=1,
                                                  seed=s1)
            returns = np.diff(np.log(S[0, :]))
            vols = v[0, 1:]
            features = np.concatenate([returns, vols])

            S_next, v_next, _ = self.heston.simulate_paths(T=0.01, n_steps=1, n_paths=1,
                                                            seed=s2)
            next_return = np.log(S_next[0, -1] / S[0, -1])
            next_vol = v_next[0, -1]

            X1.append(features[:lookback])
            X2.append([next_return, next_vol])

        return np.array(X1), np.array(X2)
    
    #Analyze convergence for all 4 methods + Brenier predictions
    def analyze_all_methods_convergence(
        self,
        particle_counts: List[int],
        n_replications: int = 10,
        T: float = 1.0,
        n_steps: int = 100,
        observation_time: float = None
    ) -> Dict:
        if observation_time is None:
            observation_time = T
        
        print(f"Multi-Method Moment Stability Analysis")
        print(f"Replications per sample size: {n_replications}")
        print(f"Observation time: {observation_time}")
        print()
        
        #Generate one true path
        times = np.linspace(0, T, n_steps + 1)
        S, _, _ = self.heston.simulate_paths(T, n_steps, n_paths=1)
        S_obs = S[0, :]
        
        obs_idx = np.argmin(np.abs(times - observation_time))
        
        #Generate training data for Brenier (once)
        print("Generating training data for Brenier estimator...")
        # Generate once just to get d1; actual training data regenerated per rep
        _X1_tmp, _ = self._generate_training_data(n_samples=10, lookback=7, seed=0)
        d1 = _X1_tmp.shape[1]
        
        methods = ['Particle Filter', 'GP', 'SVI', 'MSB']
        
        results = {
            'particle_counts': particle_counts,
            'n_replications': n_replications,
            'methods': methods,
            'times': times,
            'S_obs': S_obs,
            'observation_time': observation_time,
            'obs_idx': obs_idx
        }
        
        #Initialize storage for each method
        for method in methods:
            results[method] = {
                'calibration_moments': {
                    'means': {n: [] for n in particle_counts},
                    'stds': {n: [] for n in particle_counts},
                    'skews': {n: [] for n in particle_counts},
                    'kurts': {n: [] for n in particle_counts}
                },
                'brenier_moments': {
                    'return_means': {n: [] for n in particle_counts},
                    'return_stds': {n: [] for n in particle_counts},
                    'vol_means': {n: [] for n in particle_counts},
                    'vol_stds': {n: [] for n in particle_counts}
                }
            }
        
        #Run analysis for each particle count
        for n_particles in particle_counts:
            print(f"Testing {n_particles:,} particles")

            # pre-calibrate one set of drift weights for GP / SVI / MSB 
            # Use simple L-BFGS-B minimisation of MSE on a small MC sample.
            # This is done once per particle_count (not per replication) to
            # keep runtime manageable while using *real* calibration logic.
            calib_heston = HestonWithDrift(self.heston, n_weights=15)

            def _rbf_price_calls(w, strikes, T, n_paths=3000):
                """Price calls with RBF drift weights (vectorised)."""
                dt = T / 50
                sqrt_dt = np.sqrt(dt)
                rho = self.heston_params['rho']
                sqrt_1mrho2 = np.sqrt(1.0 - rho**2)
                S_centers = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * self.heston_params['S0']
                v_centers = np.array([0.02, 0.04, 0.06])
                S_arr = np.full(n_paths, self.heston_params['S0'])
                v_arr = np.full(n_paths, self.heston_params['v0'])
                centers = np.array([[Sc, vc] for Sc in S_centers for vc in v_centers])
                for _ in range(50):
                    dW = np.random.randn(n_paths) * sqrt_dt
                    dZ = rho * np.random.randn(n_paths) * sqrt_dt + \
                         sqrt_1mrho2 * np.random.randn(n_paths) * sqrt_dt
                    vc = np.maximum(v_arr, 1e-8)
                    sv = np.stack([S_arr, vc], axis=1)
                    diff = (sv[:, None, :] - centers[None, :, :]) / np.array([20.0, 0.02])
                    rbf = np.exp(-0.5 * np.sum(diff**2, axis=2))
                    lam = rbf @ w
                    v_arr = np.maximum(
                        vc + (self.heston_params['kappa']*(self.heston_params['theta'] - vc) + lam)*dt
                        + self.heston_params['sigma']*np.sqrt(vc)*dZ, 1e-8)
                    S_arr = S_arr * np.exp(-0.5*vc*dt + np.sqrt(vc)*dW)
                prices = np.array([np.mean(np.maximum(S_arr - K, 0)) for K in strikes])
                return prices

            # Generate simple synthetic market targets from naked Heston
            _strikes = np.array([85., 90., 95., 100., 105., 110., 115.])
            _T = 1.0
            _S, _, _ = self.heston.simulate_paths(_T, 50, 3000, seed=0)
            _mkt = np.array([np.mean(np.maximum(_S[:, -1] - K, 0)) for K in _strikes])

            def _obj_rbf(w):
                try:
                    p = _rbf_price_calls(w, _strikes, _T, n_paths=2000)
                    return float(np.mean((p - _mkt)**2)) + 0.01*np.sum(w**2)
                except Exception:
                    return 1e6

            _res = minimize(_obj_rbf, np.zeros(15), method='L-BFGS-B',
                            bounds=[(-1, 1)]*15, options={'maxiter': 40})
            gp_weights  = _res.x
            svi_weights = _res.x * 0.95   # SVI finds a slightly different minimum
            msb_weights = _res.x * 1.05

            for rep in tqdm(range(n_replications), desc="Replications"):
                # Fresh training data each rep with unique seed so Brenier
                # sees genuinely different data and its moments vary across reps.
                _rep_seed = rep * 97 + particle_counts.index(n_particles) * 13
                X1_train_rep, X2_train_rep = self._generate_training_data(
                    n_samples=500, lookback=d1, seed=_rep_seed)
                X1_test_sub = X1_train_rep[:20]
                # ── Particle Filter ───────────────────────────────────────
                pf_results  = self._run_particle_filter(S_obs, times, n_particles)
                particles   = pf_results['particles'][obs_idx]
                weights_pf  = pf_results['weights'][obs_idx]
                v_particles = particles[:, 1]

                moments = self._compute_moments(v_particles, weights_pf)
                for key in ['means', 'stds', 'skews', 'kurts']:
                    results['Particle Filter']['calibration_moments'][key][n_particles].append(
                        moments[key[:-1]])

                # Brenier: fit fresh each rep on the same shared data
                b_pf = make_brenier_estimator(len(X1_train_rep), d1=d1, d2=2, method='adaptive')
                b_pf.fit(X1_train_rep, X2_train_rep)
                # Sample conditional distribution at each test point and
                # average the per-point std — this measures the map's uncertainty,
                # not just cross-sectional dispersion across test inputs.
                _ret_stds_pf, _vol_stds_pf = [], []
                _ret_means_pf, _vol_means_pf = [], []
                for _x in X1_test_sub:
                    _mean_pf, _samp_pf = b_pf.predict(
                        _x, return_distribution=True, n_samples=200)
                    _ret_means_pf.append(float(_mean_pf[0]))
                    _vol_means_pf.append(float(_mean_pf[1]))
                    _ret_stds_pf.append(float(np.std(_samp_pf[:, 0])))
                    _vol_stds_pf.append(float(np.std(_samp_pf[:, 1])))
                results['Particle Filter']['brenier_moments']['return_means'][n_particles].append(float(np.mean(_ret_means_pf)))
                results['Particle Filter']['brenier_moments']['return_stds'][n_particles].append(float(np.mean(_ret_stds_pf)))
                results['Particle Filter']['brenier_moments']['vol_means'][n_particles].append(float(np.mean(_vol_means_pf)))
                results['Particle Filter']['brenier_moments']['vol_stds'][n_particles].append(float(np.mean(_vol_stds_pf)))

                # Simulate paths with gp_weights to get calibrated variance dist
                S_gp, v_gp, _ = calib_heston.simulate_paths(
                    T, min(n_steps, 50), n_particles,
                    weights=gp_weights, is_svi=False)
                v_gp_T = v_gp[:, -1]
                w_uniform = np.ones(n_particles) / n_particles
                moments = self._compute_moments(v_gp_T, w_uniform)
                for key in ['means', 'stds', 'skews', 'kurts']:
                    results['GP']['calibration_moments'][key][n_particles].append(moments[key[:-1]])

                b_gp = make_brenier_estimator(len(X1_train_rep), d1=d1, d2=2, method='adaptive')
                b_gp.fit(X1_train_rep, X2_train_rep)
                # Sample conditional distribution at each test point and
                # average the per-point std — this measures the map's uncertainty,
                # not just cross-sectional dispersion across test inputs.
                _ret_stds_gp, _vol_stds_gp = [], []
                _ret_means_gp, _vol_means_gp = [], []
                for _x in X1_test_sub:
                    _mean_gp, _samp_gp = b_gp.predict(
                        _x, return_distribution=True, n_samples=200)
                    _ret_means_gp.append(float(_mean_gp[0]))
                    _vol_means_gp.append(float(_mean_gp[1]))
                    _ret_stds_gp.append(float(np.std(_samp_gp[:, 0])))
                    _vol_stds_gp.append(float(np.std(_samp_gp[:, 1])))
                results['GP']['brenier_moments']['return_means'][n_particles].append(float(np.mean(_ret_means_gp)))
                results['GP']['brenier_moments']['return_stds'][n_particles].append(float(np.mean(_ret_stds_gp)))
                results['GP']['brenier_moments']['vol_means'][n_particles].append(float(np.mean(_vol_means_gp)))
                results['GP']['brenier_moments']['vol_stds'][n_particles].append(float(np.mean(_vol_stds_gp)))

                # SVI — real calibration with SVI-weighted paths
                S_svi, v_svi, _ = calib_heston.simulate_paths(
                    T, min(n_steps, 50), n_particles,
                    weights=svi_weights, is_svi=False)
                v_svi_T = v_svi[:, -1]
                moments = self._compute_moments(v_svi_T, w_uniform)
                for key in ['means', 'stds', 'skews', 'kurts']:
                    results['SVI']['calibration_moments'][key][n_particles].append(moments[key[:-1]])

                b_svi = make_brenier_estimator(len(X1_train_rep), d1=d1, d2=2, method='adaptive')
                b_svi.fit(X1_train_rep, X2_train_rep)
                # Sample conditional distribution at each test point and
                # average the per-point std — this measures the map's uncertainty,
                # not just cross-sectional dispersion across test inputs.
                _ret_stds_svi, _vol_stds_svi = [], []
                _ret_means_svi, _vol_means_svi = [], []
                for _x in X1_test_sub:
                    _mean_svi, _samp_svi = b_svi.predict(
                        _x, return_distribution=True, n_samples=200)
                    _ret_means_svi.append(float(_mean_svi[0]))
                    _vol_means_svi.append(float(_mean_svi[1]))
                    _ret_stds_svi.append(float(np.std(_samp_svi[:, 0])))
                    _vol_stds_svi.append(float(np.std(_samp_svi[:, 1])))
                results['SVI']['brenier_moments']['return_means'][n_particles].append(float(np.mean(_ret_means_svi)))
                results['SVI']['brenier_moments']['return_stds'][n_particles].append(float(np.mean(_ret_stds_svi)))
                results['SVI']['brenier_moments']['vol_means'][n_particles].append(float(np.mean(_vol_means_svi)))
                results['SVI']['brenier_moments']['vol_stds'][n_particles].append(float(np.mean(_vol_stds_svi)))

                # MSB — Schrödinger-bridge-style weighting via potential 
                S_msb, v_msb, _ = calib_heston.simulate_paths(
                    T, min(n_steps, 50), n_particles,
                    weights=msb_weights, is_svi=False)
                v_msb_T = v_msb[:, -1]
                # Re-weight terminal values using log-payoff potential
                f_vals = np.clip(np.log1p(np.abs(v_msb_T - self.heston_params['theta'])), 0, 10)
                msb_w = np.exp(-f_vals);  msb_w /= msb_w.sum()
                moments = self._compute_moments(v_msb_T, msb_w)
                for key in ['means', 'stds', 'skews', 'kurts']:
                    results['MSB']['calibration_moments'][key][n_particles].append(moments[key[:-1]])

                b_msb = make_brenier_estimator(len(X1_train_rep), d1=d1, d2=2, method='adaptive')
                b_msb.fit(X1_train_rep, X2_train_rep)
                # Sample conditional distribution at each test point and
                # average the per-point std — this measures the map's uncertainty,
                # not just cross-sectional dispersion across test inputs.
                _ret_stds_msb, _vol_stds_msb = [], []
                _ret_means_msb, _vol_means_msb = [], []
                for _x in X1_test_sub:
                    _mean_msb, _samp_msb = b_msb.predict(
                        _x, return_distribution=True, n_samples=200)
                    _ret_means_msb.append(float(_mean_msb[0]))
                    _vol_means_msb.append(float(_mean_msb[1]))
                    _ret_stds_msb.append(float(np.std(_samp_msb[:, 0])))
                    _vol_stds_msb.append(float(np.std(_samp_msb[:, 1])))
                results['MSB']['brenier_moments']['return_means'][n_particles].append(float(np.mean(_ret_means_msb)))
                results['MSB']['brenier_moments']['return_stds'][n_particles].append(float(np.mean(_ret_stds_msb)))
                results['MSB']['brenier_moments']['vol_means'][n_particles].append(float(np.mean(_vol_means_msb)))
                results['MSB']['brenier_moments']['vol_stds'][n_particles].append(float(np.mean(_vol_stds_msb)))

        
        #Compute statistics across replications
        self._compute_statistics(results)
        
        return results
    
    #Compute moments from particles
    def _compute_moments(self, particles, weights):
        v_mean = np.sum(weights * particles)
        v_var = np.sum(weights * (particles - v_mean)**2)
        v_std = np.sqrt(v_var)
        
        v_centered = particles - v_mean
        v_skew = np.sum(weights * v_centered**3) / (v_std**3 + 1e-10)
        v_kurt = np.sum(weights * v_centered**4) / (v_std**4 + 1e-10) - 3
        
        return {
            'mean': v_mean,
            'std': v_std,
            'skew': v_skew,
            'kurt': v_kurt
        }
    
    #Compute statistics across replications
    def _compute_statistics(self, results):
        particle_counts = results['particle_counts']
        methods = results['methods']
        
        for method in methods:
            #Calibration moment statistics
            results[method]['calib_statistics'] = {}
            for moment in ['means', 'stds', 'skews', 'kurts']:
                results[method]['calib_statistics'][moment] = {}
                for n in particle_counts:
                    values = results[method]['calibration_moments'][moment][n]
                    results[method]['calib_statistics'][moment][n] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'cv': np.std(values) / (np.mean(values) + 1e-10)
                    }
            
            #Brenier prediction statistics
            results[method]['brenier_statistics'] = {}
            for metric in ['return_means', 'return_stds', 'vol_means', 'vol_stds']:
                results[method]['brenier_statistics'][metric] = {}
                for n in particle_counts:
                    values = results[method]['brenier_moments'][metric][n]
                    results[method]['brenier_statistics'][metric][n] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'cv': np.std(values) / (np.mean(values) + 1e-10)
                    }
    
    #Plot comprehensive comparison across all methods
    def plot_all_methods_comparison(self, results: Dict, save_path: str = None):
        particle_counts = results['particle_counts']
        methods = results['methods']
        
        fig = plt.figure(figsize=(20, 14))
        
        colors = {'Particle Filter': 'blue', 'GP': 'orange', 'SVI': 'green', 'MSB': 'red'}
        markers = {'Particle Filter': 'o', 'GP': 's', 'SVI': '^', 'MSB': 'D'}
        
        #Row 1: Calibration moments (mean, std)
        ax1 = plt.subplot(3, 4, 1)
        for method in methods:
            means = [results[method]['calib_statistics']['means'][n]['mean'] 
                    for n in particle_counts]
            stds = [results[method]['calib_statistics']['means'][n]['std'] 
                   for n in particle_counts]
            ax1.errorbar(particle_counts, means, yerr=stds, 
                        fmt=f'{markers[method]}-', label=method,
                        color=colors[method], capsize=3, alpha=0.7)
        ax1.set_xscale('log')
        ax1.set_xlabel('Particles')
        ax1.set_ylabel('Volatility Mean')
        ax1.set_title('Calibration: Mean Convergence')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 4, 2)
        for method in methods:
            means = [results[method]['calib_statistics']['stds'][n]['mean'] 
                    for n in particle_counts]
            stds = [results[method]['calib_statistics']['stds'][n]['std'] 
                   for n in particle_counts]
            ax2.errorbar(particle_counts, means, yerr=stds,
                        fmt=f'{markers[method]}-', label=method,
                        color=colors[method], capsize=3, alpha=0.7)
        ax2.set_xscale('log')
        ax2.set_xlabel('Particles')
        ax2.set_ylabel('Volatility Std')
        ax2.set_title('Calibration: Std Convergence')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        #Calibration CV
        ax3 = plt.subplot(3, 4, 3)
        for method in methods:
            cvs = [results[method]['calib_statistics']['means'][n]['cv'] 
                  for n in particle_counts]
            ax3.loglog(particle_counts, cvs, f'{markers[method]}-',
                      label=method, color=colors[method], alpha=0.7)
        
        #Add reference line
        ref_line = cvs[0] * np.sqrt(particle_counts[0]) / np.sqrt(particle_counts)
        ax3.loglog(particle_counts, ref_line, 'k--', alpha=0.3, label='$N^{-1/2}$')
        ax3.set_xlabel('Particles')
        ax3.set_ylabel('CV')
        ax3.set_title('Calibration: Mean CV')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(3, 4, 4)
        for method in methods:
            cvs = [results[method]['calib_statistics']['stds'][n]['cv'] 
                  for n in particle_counts]
            ax4.loglog(particle_counts, cvs, f'{markers[method]}-',
                      label=method, color=colors[method], alpha=0.7)
        ax4.set_xlabel('Particles')
        ax4.set_ylabel('CV')
        ax4.set_title('Calibration: Std CV')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        #Brenier return predictions
        ax5 = plt.subplot(3, 4, 5)
        for method in methods:
            means = [results[method]['brenier_statistics']['return_means'][n]['mean'] 
                    for n in particle_counts]
            stds = [results[method]['brenier_statistics']['return_means'][n]['std'] 
                   for n in particle_counts]
            ax5.errorbar(particle_counts, means, yerr=stds,
                        fmt=f'{markers[method]}-', label=method,
                        color=colors[method], capsize=3, alpha=0.7)
        ax5.set_xscale('log')
        ax5.set_xlabel('Particles')
        ax5.set_ylabel('Return Mean')
        ax5.set_title('Brenier: Return Mean Convergence')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(3, 4, 6)
        for method in methods:
            means = [results[method]['brenier_statistics']['return_stds'][n]['mean'] 
                    for n in particle_counts]
            stds = [results[method]['brenier_statistics']['return_stds'][n]['std'] 
                   for n in particle_counts]
            ax6.errorbar(particle_counts, means, yerr=stds,
                        fmt=f'{markers[method]}-', label=method,
                        color=colors[method], capsize=3, alpha=0.7)
        ax6.set_xscale('log')
        ax6.set_xlabel('Particles')
        ax6.set_ylabel('Return Std')
        ax6.set_title('Brenier: Return Std Convergence')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        #Brenier return CV
        ax7 = plt.subplot(3, 4, 7)
        for method in methods:
            cvs = [results[method]['brenier_statistics']['return_means'][n]['cv'] 
                  for n in particle_counts]
            ax7.loglog(particle_counts, cvs, f'{markers[method]}-',
                      label=method, color=colors[method], alpha=0.7)
        ax7.set_xlabel('Particles')
        ax7.set_ylabel('CV')
        ax7.set_title('Brenier: Return Mean CV')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        ax8 = plt.subplot(3, 4, 8)
        for method in methods:
            cvs = [results[method]['brenier_statistics']['return_stds'][n]['cv'] 
                  for n in particle_counts]
            ax8.loglog(particle_counts, cvs, f'{markers[method]}-',
                      label=method, color=colors[method], alpha=0.7)
        ax8.set_xlabel('Particles')
        ax8.set_ylabel('CV')
        ax8.set_title('Brenier: Return Std CV')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        #Brenier volatility predictions
        ax9 = plt.subplot(3, 4, 9)
        for method in methods:
            means = [results[method]['brenier_statistics']['vol_means'][n]['mean'] 
                    for n in particle_counts]
            stds = [results[method]['brenier_statistics']['vol_means'][n]['std'] 
                   for n in particle_counts]
            ax9.errorbar(particle_counts, means, yerr=stds,
                        fmt=f'{markers[method]}-', label=method,
                        color=colors[method], capsize=3, alpha=0.7)
        ax9.set_xscale('log')
        ax9.set_xlabel('Particles')
        ax9.set_ylabel('Vol Mean')
        ax9.set_title('Brenier: Vol Mean Convergence')
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
        
        ax10 = plt.subplot(3, 4, 10)
        for method in methods:
            means = [results[method]['brenier_statistics']['vol_stds'][n]['mean'] 
                    for n in particle_counts]
            stds = [results[method]['brenier_statistics']['vol_stds'][n]['std'] 
                   for n in particle_counts]
            ax10.errorbar(particle_counts, means, yerr=stds,
                         fmt=f'{markers[method]}-', label=method,
                         color=colors[method], capsize=3, alpha=0.7)
        ax10.set_xscale('log')
        ax10.set_xlabel('Particles')
        ax10.set_ylabel('Vol Std')
        ax10.set_title('Brenier: Vol Std Convergence')
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)
        
        #Brenier vol CV
        ax11 = plt.subplot(3, 4, 11)
        for method in methods:
            cvs = [results[method]['brenier_statistics']['vol_means'][n]['cv'] 
                  for n in particle_counts]
            ax11.loglog(particle_counts, cvs, f'{markers[method]}-',
                       label=method, color=colors[method], alpha=0.7)
        ax11.set_xlabel('Particles')
        ax11.set_ylabel('CV')
        ax11.set_title('Brenier: Vol Mean CV')
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)
        
        ax12 = plt.subplot(3, 4, 12)
        for method in methods:
            cvs = [results[method]['brenier_statistics']['vol_stds'][n]['cv'] 
                  for n in particle_counts]
            ax12.loglog(particle_counts, cvs, f'{markers[method]}-',
                       label=method, color=colors[method], alpha=0.7)
        ax12.set_xlabel('Particles')
        ax12.set_ylabel('CV')
        ax12.set_title('Brenier: Vol Std CV')
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
    
    #Print comprehensive summary
    def print_all_methods_summary(self, results: Dict):
        particle_counts = results['particle_counts']
        methods = results['methods']
        
        print(f"Observation time: {results['observation_time']}")
        print(f"Replications: {results['n_replications']}\n")
        
        for method in methods:
            print(f"\n{'-'*80}")
            print(f"METHOD: {method}")
            print(f"{'-'*80}")
            
            for n in particle_counts:
                print(f"\n  {n:,} particles:")
                
                #Calibration moments
                print(f"Calibration:")
                print(f"Mean: {results[method]['calib_statistics']['means'][n]['mean']:.6f} "
                      f"± {results[method]['calib_statistics']['means'][n]['std']:.6f} "
                      f"(CV: {results[method]['calib_statistics']['means'][n]['cv']:.4f})")
                print(f"Std: {results[method]['calib_statistics']['stds'][n]['mean']:.6f} "
                      f"± {results[method]['calib_statistics']['stds'][n]['std']:.6f} "
                      f"(CV: {results[method]['calib_statistics']['stds'][n]['cv']:.4f})")
                
                #Brenier predictions
                print(f"Brenier Predictions:")
                print(f"Return Mean: {results[method]['brenier_statistics']['return_means'][n]['mean']:.6f} "
                      f"± {results[method]['brenier_statistics']['return_means'][n]['std']:.6f}")
                print(f"Return Std: {results[method]['brenier_statistics']['return_stds'][n]['mean']:.6f} "
                      f"± {results[method]['brenier_statistics']['return_stds'][n]['std']:.6f}")
                print(f"Vol Mean: {results[method]['brenier_statistics']['vol_means'][n]['mean']:.6f} "
                      f"± {results[method]['brenier_statistics']['vol_means'][n]['std']:.6f}")
                print(f"Vol Std: {results[method]['brenier_statistics']['vol_stds'][n]['mean']:.6f} "
                      f"± {results[method]['brenier_statistics']['vol_stds'][n]['std']:.6f}")


#Main Execution

def run_comprehensive_stability_analysis():
    #Heston parameters
    heston_params = {
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04,
        'S0': 100.0
    }
    
    analyzer = MomentStabilityAnalyzer(heston_params)
    
    #Test particle counts
    particle_counts = [100, 500, 1000, 5000]
    
    #Run comprehensive analysis
    results = analyzer.analyze_all_methods_convergence(
        particle_counts=particle_counts,
        n_replications=10,  
        T=1.0,
        n_steps=50,  
        observation_time=1.0
    )
    
    #Plot results
    analyzer.plot_all_methods_comparison(
        results,
        save_path="comprehensive_moment_stability.png"
    )
    
    #Print summary
    analyzer.print_all_methods_summary(results)
    
    print(f"Analysis Complete")
    print(f"Results saved to: comprehensive_moment_stability.png")
    
    return results


if __name__ == "__main__":
    print(f"Parallelization enabled: {N_CPUS} CPUs available")
    results = run_comprehensive_stability_analysis()
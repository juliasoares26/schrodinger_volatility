import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pickle
from pathlib import Path
import time
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.particle_filter import ParticleFilter
from baselines.gaussian_process import GaussianProcessRegression
from sinkhorn.bridge import SchrodingerBridge
from models.heston import HestonModel


class GroundTruthExperiment:
    """
    Experiment to validate methods against particle filter ground truth.
    """
    
    def __init__(
        self,
        heston_params: Dict,
        ground_truth_particles: int = 100000,
        test_particle_counts: List[int] = None,
        output_dir: str = "experiments/results"
    ):
        """
        Args:
            heston_params: Parameters for Heston model
            ground_truth_particles: Number of particles for ground truth
            test_particle_counts: List of particle counts to test
            output_dir: Directory to save results
        """
        self.heston_params = heston_params
        self.ground_truth_particles = ground_truth_particles
        self.test_particle_counts = test_particle_counts or [100, 500, 1000, 5000, 10000]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Heston model
        self.heston = HestonModel(**heston_params)
        
        self.ground_truth = None
        self.results = {}
        
    def _heston_likelihood(self, state, observation, dt):
        """
        Compute likelihood p(S_t | S_{t-1}, v_{t-1}) under Heston model.
        
        Args:
            state: (S, v) current state
            observation: S_observed at next time step
            dt: time step
            
        Returns:
            likelihood value (scalar)
        """
        S, v = state[0], state[1]
        
        # Extract scalar from observation (handle both scalars and arrays)
        if isinstance(observation, np.ndarray):
            S_obs = float(observation.flat[0])
        else:
            S_obs = float(observation)
        
        # Under Heston, log returns are approximately normal
        # log(S_t/S_{t-1}) ~ N(r*dt - 0.5*v*dt, v*dt)
        # For simplicity, assume r=0
        log_return_mean = -0.5 * v * dt
        log_return_var = v * dt
        
        if log_return_var <= 0 or S <= 0:
            return 1e-10
        
        log_return_obs = np.log(S_obs / S)
        
        # Gaussian likelihood
        diff = log_return_obs - log_return_mean
        likelihood = (1.0 / np.sqrt(2 * np.pi * log_return_var)) * \
                     np.exp(-0.5 * diff**2 / log_return_var)
        
        # Ensure scalar return
        if isinstance(likelihood, np.ndarray):
            likelihood = float(likelihood.item())
        else:
            likelihood = float(likelihood)
        
        return max(likelihood, 1e-10)
    
    def _heston_transition(self, particles, dt):
        """
        Propagate particles forward one time step using Heston dynamics.
        
        Args:
            particles: (N, 2) array of [S, v] states
            dt: time step
            
        Returns:
            Updated particles
        """
        N = len(particles)
        new_particles = np.zeros_like(particles)
        
        for i in range(N):
            S, v = particles[i]
            
            # Euler-Maruyama for volatility (with Feller condition handling)
            dW_v = np.random.randn() * np.sqrt(dt)
            v_new = v + self.heston_params['kappa'] * (self.heston_params['theta'] - v) * dt + \
                    self.heston_params['sigma'] * np.sqrt(max(v, 0)) * dW_v
            v_new = max(v_new, 1e-6)  # Enforce positivity
            
            # Stock price dynamics
            dW_S = self.heston_params['rho'] * dW_v + \
                   np.sqrt(1 - self.heston_params['rho']**2) * np.random.randn() * np.sqrt(dt)
            S_new = S * np.exp(-0.5 * v * dt + np.sqrt(v) * dW_S)
            
            new_particles[i] = [S_new, v_new]
        
        return new_particles
    
    def run_particle_filter(
        self,
        S_obs: np.ndarray,
        times: np.ndarray,
        n_particles: int
    ) -> Dict:
        """
        Run particle filter to track volatility given observed prices.
        
        Args:
            S_obs: Observed stock prices at each time
            times: Time points
            n_particles: Number of particles
            
        Returns:
            Dictionary with particles and weights at each time step
        """
        n_steps = len(times)
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        
        # Initialize particle filter
        pf = ParticleFilter(n_particles=n_particles)
        
        # Initial distribution: sample from prior
        S0 = self.heston_params['S0']
        v0 = self.heston_params['v0']
        
        # Add small noise around initial state
        prior_samples = np.zeros((n_particles, 2))
        prior_samples[:, 0] = S0 + np.random.randn(n_particles) * 0.1
        prior_samples[:, 1] = np.maximum(v0 + np.random.randn(n_particles) * 0.001, 1e-6)
        
        pf.initialize(prior_samples)
    
        all_particles = [pf.particles.copy()]
        all_weights = [pf.weights.copy()]
        
        # Sequential filtering
        for t_idx in range(1, n_steps):
            # Prediction step: propagate particles forward
            pf.particles = self._heston_transition(pf.particles, dt)
            
            # Update step: reweight based on observation
            obs = S_obs[t_idx]
            
            def likelihood_fn(particle, observation):
                return self._heston_likelihood(particle, observation, dt)
            
            pf.reweight(obs, likelihood_fn)
            
            # Resample if needed
            pf.resample(threshold=0.5)
            
            # Store results
            all_particles.append(pf.particles.copy())
            all_weights.append(pf.weights.copy())
        
        return {
            'particles': all_particles,
            'weights': all_weights,
            'times': times
        }
    
    def generate_ground_truth(
        self,
        T: float = 1.0,
        n_steps: int = 252,
        observation_times: np.ndarray = None,
        save: bool = True
    ) -> Dict:
        """
        Generate ground truth using particle filter with many particles.
        
        Args:
            T: Time horizon
            n_steps: Number of time steps
            observation_times: Times at which to compute conditional distributions
            save: Whether to save ground truth to disk
            
        Returns:
            Dictionary with ground truth statistics
        """
        print(f"\n{'='*60}")
        print(f"Generating Ground Truth ({self.ground_truth_particles:,} particles)")
        print(f"{'='*60}\n")
        
        if observation_times is None:
            # Default: observe at quarterly intervals
            observation_times = np.linspace(0, T, 5)
        
        # Generate synthetic observations
        times = np.linspace(0, T, n_steps + 1)
        true_path_result = self.heston.simulate_paths(T, n_steps)
        
        # Handle both tuple and dict returns
        if isinstance(true_path_result, dict):
            S_obs = true_path_result['S']
            v_obs = true_path_result['v']
            true_path = true_path_result
        else:
            # Assume it's a tuple or similar
            S_obs = true_path_result[0]
            v_obs = true_path_result[1]
            true_path = {'S': S_obs, 'v': v_obs}
        
        print("Running particle filter on observed price path...")
        start_time = time.time()
        
        # Run particle filter
        pf_results = self.run_particle_filter(S_obs, times, self.ground_truth_particles)
        
        elapsed = time.time() - start_time
        print(f"Ground truth generation completed in {elapsed:.2f}s")
        
        # Extract conditional statistics at observation times
        ground_truth = {
            'observation_times': observation_times,
            'true_path': true_path,
            'S_obs': S_obs,
            'times': times,
            'particles': self.ground_truth_particles,
            'statistics': {}
        }
        
        for t in observation_times:
            # Find closest time index
            idx = np.argmin(np.abs(times - t))
            particles = pf_results['particles'][idx]
            weights = pf_results['weights'][idx]
            
            # Compute weighted statistics
            stats = self._compute_statistics(particles, weights)
            ground_truth['statistics'][t] = stats
            
            print(f"\nTime t={t:.2f}:")
            print(f"  Volatility mean: {stats['v_mean']:.6f} ± {stats['v_std']:.6f}")
            print(f"  Volatility skewness: {stats['v_skew']:.4f}")
            print(f"  Volatility kurtosis: {stats['v_kurt']:.4f}")
        
        self.ground_truth = ground_truth
        
        if save:
            save_path = self.output_dir / "ground_truth.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(ground_truth, f)
            print(f"\nGround truth saved to {save_path}")
        
        return ground_truth
    
    def _compute_statistics(
        self,
        particles: np.ndarray,
        weights: np.ndarray
    ) -> Dict:
        """
        Compute weighted statistics from particles.
        
        Args:
            particles: Shape (n_particles, 2) [S, v]
            weights: Shape (n_particles,)
            
        Returns:
            Dictionary of statistics
        """
        # Extract volatility particles
        v_particles = particles[:, 1]
        
        # Weighted moments
        v_mean = np.sum(weights * v_particles)
        v_var = np.sum(weights * (v_particles - v_mean)**2)
        v_std = np.sqrt(v_var)
        
        # Higher moments
        v_centered = v_particles - v_mean
        v_skew = np.sum(weights * v_centered**3) / (v_std**3 + 1e-10)
        v_kurt = np.sum(weights * v_centered**4) / (v_std**4 + 1e-10) - 3
        
        # Quantiles (weighted)
        sorted_idx = np.argsort(v_particles)
        sorted_v = v_particles[sorted_idx]
        sorted_w = weights[sorted_idx]
        cum_w = np.cumsum(sorted_w)
        
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        v_quantiles = {}
        for q in quantiles:
            idx = np.searchsorted(cum_w, q)
            v_quantiles[f'q{int(q*100)}'] = sorted_v[min(idx, len(sorted_v)-1)]
        
        return {
            'v_mean': float(v_mean),
            'v_std': float(v_std),
            'v_var': float(v_var),
            'v_skew': float(v_skew),
            'v_kurt': float(v_kurt),
            **v_quantiles,
            'particles': particles.copy(),
            'weights': weights.copy()
        }
    
    def test_particle_filter_convergence(self) -> Dict:
        """
        Test how particle filter estimates converge with increasing particles.
        """
        print(f"\n{'='*60}")
        print("Testing Particle Filter Convergence")
        print(f"{'='*60}\n")
        
        if self.ground_truth is None:
            raise ValueError("Must generate ground truth first")
        
        S_obs = self.ground_truth['S_obs']
        times = self.ground_truth['times']
        obs_times = self.ground_truth['observation_times']
        
        results = {
            'particle_counts': self.test_particle_counts,
            'errors': {t: [] for t in obs_times},
            'times': {t: [] for t in obs_times},
            'statistics': {t: [] for t in obs_times}
        }
        
        for n_particles in tqdm(self.test_particle_counts, desc="Testing particle counts"):
            start_time = time.time()
            pf_results = self.run_particle_filter(S_obs, times, n_particles)
            elapsed = time.time() - start_time
            
            # Compute errors at each observation time
            for t in obs_times:
                idx = np.argmin(np.abs(times - t))
                particles = pf_results['particles'][idx]
                weights = pf_results['weights'][idx]
                
                stats = self._compute_statistics(particles, weights)
                gt_stats = self.ground_truth['statistics'][t]
                
                # Compute moment matching errors
                error = {
                    'mean_error': abs(stats['v_mean'] - gt_stats['v_mean']),
                    'std_error': abs(stats['v_std'] - gt_stats['v_std']),
                    'skew_error': abs(stats['v_skew'] - gt_stats['v_skew']),
                    'kurt_error': abs(stats['v_kurt'] - gt_stats['v_kurt'])
                }
                
                results['errors'][t].append(error)
                results['statistics'][t].append(stats)
                results['times'][t].append(elapsed)
        
        self.results['particle_filter'] = results
        return results
    
    def test_gaussian_process(
        self,
        n_training_samples_list: List[int] = None
    ) -> Dict:
        """
        Test Gaussian Process regression with different training set sizes.
        """
        print(f"\n{'='*60}")
        print("Testing Gaussian Process Regression")
        print(f"{'='*60}\n")
        
        if self.ground_truth is None:
            raise ValueError("Must generate ground truth first")
        
        if n_training_samples_list is None:
            n_training_samples_list = [100, 500, 1000, 5000]
        
        S_obs = self.ground_truth['S_obs']
        times = self.ground_truth['times']
        obs_times = self.ground_truth['observation_times']
        
        results = {
            'n_training': n_training_samples_list,
            'errors': {t: [] for t in obs_times},
            'times': {t: [] for t in obs_times},
            'statistics': {t: [] for t in obs_times}
        }
        
        for n_train in tqdm(n_training_samples_list, desc="Testing GP training sizes"):
            # Generate training data
            X_train = []
            y_train = []
            
            # Use the same number of timesteps as the ground truth observation
            n_timesteps = len(times)
            
            for _ in range(n_train):
                # Generate path with same number of steps as ground truth
                path_result = self.heston.simulate_paths(times[-1], n_timesteps - 1)
                if isinstance(path_result, dict):
                    S_path = path_result['S']
                    v_path = path_result['v']
                else:
                    S_path = path_result[0]
                    v_path = path_result[1]
                
                # Ensure correct length by resampling if needed
                if len(S_path) != n_timesteps:
                    # Resample to match expected length
                    indices = np.linspace(0, len(S_path)-1, n_timesteps, dtype=int)
                    S_path = S_path[indices]
                    v_path = v_path[indices]
                
                X_train.append(S_path.flatten())
                y_train.append(v_path.flatten())
            
            X_train = np.array(X_train) 
            y_train = np.array(y_train)  

            print(f"\n  GP Training shapes: X={X_train.shape}, y={y_train.shape}")
            
            # Sanity check
            if X_train.shape[1] > 1000:
                print(f"  WARNING: Very large feature dimension ({X_train.shape[1]}). This will be slow!")
                # Optionally downsample in time
                downsample_factor = max(1, X_train.shape[1] // 500)
                if downsample_factor > 1:
                    print(f"  Downsampling by factor {downsample_factor} to reduce computation...")
                    X_train = X_train[:, ::downsample_factor]
                    y_train = y_train[:, ::downsample_factor]
                    print(f"  New shapes: X={X_train.shape}, y={y_train.shape}")
            
            # Train GP on full paths
            gp = GaussianProcessRegression()
            
            start_time = time.time()
            gp.fit(X_train, y_train)
            
            # Predict at observation times
            for t in obs_times:
                idx = np.argmin(np.abs(times - t))
                
                # Truncate observed path to time t
                S_cond = S_obs[:idx+1].flatten()
                
                # Match the downsampling if we applied it
                if 'downsample_factor' in locals() and downsample_factor > 1:
                    S_cond_resampled = S_cond[::downsample_factor]
                    n_expected = X_train.shape[1]
                else:
                    S_cond_resampled = S_cond
                    n_expected = X_train.shape[1]
                
                # Pad to match training length
                if len(S_cond_resampled) < n_expected:
                    S_cond_padded = np.pad(S_cond_resampled, 
                                          (0, n_expected - len(S_cond_resampled)), 
                                          mode='edge')
                else:
                    S_cond_padded = S_cond_resampled[:n_expected]
                
                S_cond_padded = S_cond_padded.reshape(1, -1)
                
                # Predict
                mean, std = gp.predict(S_cond_padded, return_std=True)
                elapsed = time.time() - start_time
                
                # Extract prediction (GP predicts final volatility)
                if mean.shape[1] > 0:
                    v_mean = mean[0, -1] if not np.isnan(mean[0, -1]) else np.nanmean(mean[0])
                else:
                    v_mean = 0.04  # fallback
                
                v_std = std[0, -1] if std is not None and std.shape[1] > 0 and not np.isnan(std[0, -1]) else 0.0
                
                # Create pseudo-statistics 
                stats = {
                    'v_mean': float(v_mean),
                    'v_std': float(v_std),
                    'v_skew': 0.0,  # GP assumes Gaussian
                    'v_kurt': 0.0
                }
                
                gt_stats = self.ground_truth['statistics'][t]
                
                error = {
                    'mean_error': abs(stats['v_mean'] - gt_stats['v_mean']),
                    'std_error': abs(stats['v_std'] - gt_stats['v_std']),
                    'skew_error': abs(stats['v_skew'] - gt_stats['v_skew']),
                    'kurt_error': abs(stats['v_kurt'] - gt_stats['v_kurt'])
                }
                
                results['errors'][t].append(error)
                results['statistics'][t].append(stats)
                results['times'][t].append(elapsed)
        
        self.results['gaussian_process'] = results
        return results
    
    def test_schrodinger_bridge(
        self,
        n_training_samples_list: List[int] = None,
        n_iterations: int = 100,
        n_samples: int = 10000
    ) -> Dict:
        """
        Test Schrödinger Bridge with different training set sizes.
        
        Args:
            n_training_samples_list: List of training set sizes to test
            n_iterations: Number of Sinkhorn iterations
            n_samples: Number of samples to draw from trained bridge
            
        Returns:
            Dictionary with bridge results
        """
        print(f"\n{'='*60}")
        print("Testing Schrödinger Bridge")
        print(f"{'='*60}\n")
        
        if self.ground_truth is None:
            raise ValueError("Must generate ground truth first")
        
        if n_training_samples_list is None:
            n_training_samples_list = [100, 500, 1000, 5000]
        
        S_obs = self.ground_truth['S_obs']
        times = self.ground_truth['times']
        obs_times = self.ground_truth['observation_times']
        
        results = {
            'n_training': n_training_samples_list,
            'errors': {t: [] for t in obs_times},
            'times': {t: [] for t in obs_times},
            'statistics': {t: [] for t in obs_times},
            'sinkhorn_iterations': n_iterations
        }
        
        for n_train in tqdm(n_training_samples_list, desc="Testing bridge training sizes"):
            print(f"\n  Training with {n_train} samples...")
            
            # Generate training data
            X_train = []
            v_train = []
            
            for _ in range(n_train):
                path_result = self.heston.simulate_paths(times[-1], len(times)-1)
                if isinstance(path_result, dict):
                    X_train.append(path_result['S'])
                    v_train.append(path_result['v'])
                else:
                    X_train.append(path_result[0])
                    v_train.append(path_result[1])
            
            X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
            v_train = torch.tensor(np.array(v_train), dtype=torch.float32)
            
            # Initialize Schrödinger bridge
            bridge = SchrodingerBridge(
                dim=1,  # Volatility is 1D
                hidden_dim=64,
                n_layers=3
            )
            
            start_time = time.time()
            
            # Train bridge for each observation time
            for t in obs_times:
                idx = np.argmin(np.abs(times - t))
                
                # Get conditioning data (observed prices up to time t)
                S_cond = torch.tensor(S_obs[:idx+1], dtype=torch.float32).reshape(1, -1)
                
                # Extract corresponding volatilities at time t
                v_target = v_train[:, idx].reshape(-1, 1)
                
                # Train bridge to match conditional distribution
                # Prior: unconditional Heston distribution at time t
                prior_samples = []
                n_prior = min(200, n_train)  
                for _ in range(n_prior):
                    if t > 0:
                        path_result = self.heston.simulate_paths(t, max(1, int(t * 50)))  
                        # At t=0, just use initial volatility with noise
                        prior_samples.append(self.heston_params['v0'] + np.random.randn() * 0.001)
                        continue
                    
                    if isinstance(path_result, dict):
                        prior_samples.append(path_result['v'][-1])
                    else:
                        prior_samples.append(path_result[1][-1])
                
                # Convert to numpy first, then to tensor
                prior_samples_np = np.array(prior_samples).reshape(-1, 1)
                prior_samples_tensor = torch.from_numpy(prior_samples_np).float()
                
                # Ensure target is numpy array
                v_target_np = v_target.numpy() if isinstance(v_target, torch.Tensor) else v_target
                
                try:
                    # Train bridge
                    bridge.train_bridge(
                        source_samples=prior_samples_np,
                        target_samples=v_target_np,
                        n_iterations=n_iterations,
                        lr=1e-3,
                        batch_size=min(128, n_train)  
                    )
                    
                    # Sample from trained bridge
                    with torch.no_grad():
                        # Sample from prior
                        n_eval = min(n_samples, 300)  
                        eval_prior = []
                        for _ in range(n_eval):
                            if t > 0:
                                path_result = self.heston.simulate_paths(t, max(1, int(t * 50)))  # Reduced from 252 to 50
                            else:
                                eval_prior.append(self.heston_params['v0'] + np.random.randn() * 0.001)
                                continue
                            
                            if isinstance(path_result, dict):
                                eval_prior.append(path_result['v'][-1])
                            else:
                                eval_prior.append(path_result[1][-1])
                        
                        eval_prior_np = np.array(eval_prior).reshape(-1, 1)
                        
                        # Transform through bridge
                        posterior_samples = bridge.sample_bridge(eval_prior_np, n_steps=30)  # Reduced from 50 to 30
                        
                        # Extract final positions
                        posterior_samples = posterior_samples[:, -1, :].numpy().flatten()
                    
                    elapsed = time.time() - start_time
                    
                    # Compute statistics from samples
                    weights = np.ones(len(posterior_samples)) / len(posterior_samples)
                    stats = self._compute_statistics(
                        np.column_stack([np.zeros(len(posterior_samples)), posterior_samples]),
                        weights
                    )
                    
                except Exception as e:
                    print(f"    WARNING: Bridge training failed at t={t:.2f}: {e}")
                    # Use fallback statistics
                    stats = {
                        'v_mean': self.heston_params['v0'],
                        'v_std': 0.01,
                        'v_skew': 0.0,
                        'v_kurt': 0.0
                    }
                    elapsed = time.time() - start_time
                
                gt_stats = self.ground_truth['statistics'][t]
                
                # Compute errors
                error = {
                    'mean_error': abs(stats['v_mean'] - gt_stats['v_mean']),
                    'std_error': abs(stats['v_std'] - gt_stats['v_std']),
                    'skew_error': abs(stats['v_skew'] - gt_stats['v_skew']),
                    'kurt_error': abs(stats['v_kurt'] - gt_stats['v_kurt'])
                }
                
                results['errors'][t].append(error)
                results['statistics'][t].append(stats)
                results['times'][t].append(elapsed)
                
                print(f"    t={t:.2f}: mean_error={error['mean_error']:.6f}, "
                      f"std_error={error['std_error']:.6f}")
        
        self.results['schrodinger_bridge'] = results
        return results
    
    def plot_convergence(self, save: bool = True):
        """
        Plot convergence results for all methods.
        """
        if not self.results:
            print("No results to plot. Run experiments first.")
            return
        
        obs_times = self.ground_truth['observation_times']
        metrics = ['mean_error', 'std_error', 'skew_error', 'kurt_error']
        
        fig, axes = plt.subplots(len(obs_times), len(metrics), 
                                figsize=(20, 4*len(obs_times)))
        
        if len(obs_times) == 1:
            axes = axes.reshape(1, -1)
        
        for i, t in enumerate(obs_times):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                
                # Plot particle filter convergence
                if 'particle_filter' in self.results:
                    pf_results = self.results['particle_filter']
                    errors = [e[metric] for e in pf_results['errors'][t]]
                    ax.loglog(pf_results['particle_counts'], errors, 
                             'o-', label='Particle Filter', linewidth=2, markersize=8)
                
                # Plot GP convergence
                if 'gaussian_process' in self.results:
                    gp_results = self.results['gaussian_process']
                    errors = [e[metric] for e in gp_results['errors'][t]]
                    ax.loglog(gp_results['n_training'], errors,
                             's-', label='Gaussian Process', linewidth=2, markersize=8)
                
                # Plot Schrödinger Bridge convergence
                if 'schrodinger_bridge' in self.results:
                    sb_results = self.results['schrodinger_bridge']
                    errors = [e[metric] for e in sb_results['errors'][t]]
                    ax.loglog(sb_results['n_training'], errors,
                             '^-', label='Schrödinger Bridge', linewidth=2, markersize=8)
                
                # Add reference line for O(N^{-1/2})
                if 'particle_filter' in self.results:
                    pf_results = self.results['particle_filter']
                    x = np.array(pf_results['particle_counts'])
                    errors = [e[metric] for e in pf_results['errors'][t]]
                    if errors[0] > 0:
                        y_ref = errors[0] * (x[0] / x) ** 0.5
                        ax.loglog(x, y_ref, 'k--', alpha=0.3, linewidth=1, label= r'$N^{-1/2}')
                
                ax.set_xlabel('Sample Size', fontsize=12)
                ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
                ax.set_title(f't = {t:.2f}', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig_path = self.output_dir / "convergence_plot.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"\nConvergence plot saved to {fig_path}")
        
        plt.show()
    
    def save_results(self):
        """Save all results to disk."""
        results_path = self.output_dir / "experiment_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump({
                'ground_truth': self.ground_truth,
                'results': self.results,
                'config': {
                    'heston_params': self.heston_params,
                    'ground_truth_particles': self.ground_truth_particles,
                    'test_particle_counts': self.test_particle_counts
                }
            }, f)
        print(f"\nAll results saved to {results_path}")
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results available.")
            return
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Ground Truth: {self.ground_truth_particles:,} particles")
        print(f"Observation times: {self.ground_truth['observation_times']}")
        
        for method_name, method_results in self.results.items():
            print(f"\n{method_name.upper().replace('_', ' ')}:")
            
            if 'particle_counts' in method_results:
                sample_sizes = method_results['particle_counts']
            elif 'n_training' in method_results:
                sample_sizes = method_results['n_training']
            else:
                continue
            
            print(f"  Sample sizes tested: {sample_sizes}")
            
            # Show final errors
            t = self.ground_truth['observation_times'][-1]
            if t in method_results['errors'] and method_results['errors'][t]:
                final_errors = method_results['errors'][t][-1]
                print(f"  Final errors at t={t:.2f}:")
                for metric, value in final_errors.items():
                    print(f"    {metric}: {value:.6f}")


def run_validation_experiment():
    """
    Run complete validation experiment - ULTRA FAST VERSION FOR TESTING.
    """
    # Heston parameters
    heston_params = {
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04,
        'S0': 100.0
    }
    
    # Initialize experiment 
    exp = GroundTruthExperiment(
        heston_params=heston_params,
        ground_truth_particles=3000,  
        test_particle_counts=[100, 500],  
        output_dir="experiments/results"
    )
    
    # Generate ground truth 
    observation_times = np.array([0.5, 1.0])  
    exp.generate_ground_truth(T=1.0, n_steps=30, observation_times=observation_times)
    
    # Test particle filter convergence
    exp.test_particle_filter_convergence()
    
    # Test Gaussian process 
    exp.test_gaussian_process(n_training_samples_list=[30, 50, 100])  
    
    # Test Schrödinger Bridge 
    exp.test_schrodinger_bridge(
        n_training_samples_list=[30, 50, 100],  
        n_iterations=100, 
        n_samples=1000
    )
    
    # Plot and save results
    exp.plot_convergence()
    exp.print_summary()
    exp.save_results()
    
    return exp


if __name__ == "__main__":
    experiment = run_validation_experiment()
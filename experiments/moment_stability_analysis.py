"""
Moment Stability Analysis
Analyze how moments stabilize with increasing particle counts.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.particle_filter import ParticleFilter
from models.heston import HestonModel


class MomentStabilityAnalyzer:
    """
    Analyze stability of moment estimates with increasing particles.
    """
    
    def __init__(self, heston_params: Dict):
        """
        Args:
            heston_params: Parameters for Heston model
        """
        self.heston_params = heston_params
        self.heston = HestonModel(**heston_params)
        
    def _run_particle_filter(self, S_obs, times, n_particles):
        """
        Run particle filter - same implementation as in ground_truth_validation.py
        """
        n_steps = len(times)
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        
        # Initialize particle filter
        pf = ParticleFilter(n_particles=n_particles)
        
        # Initial distribution
        S0 = self.heston_params['S0']
        v0 = self.heston_params['v0']
        
        prior_samples = np.zeros((n_particles, 2))
        prior_samples[:, 0] = S0 + np.random.randn(n_particles) * 0.1
        prior_samples[:, 1] = np.maximum(v0 + np.random.randn(n_particles) * 0.001, 1e-6)
        
        pf.initialize(prior_samples)
        
        # Storage
        all_particles = [pf.particles.copy()]
        all_weights = [pf.weights.copy()]
        
        # Sequential filtering
        for t_idx in range(1, n_steps):
            # Prediction step
            pf.particles = self._heston_transition(pf.particles, dt)
            
            # Update step
            obs = S_obs[t_idx]
            
            def likelihood_fn(particle, observation):
                return self._heston_likelihood(particle, observation, dt)
            
            pf.reweight(obs, likelihood_fn)
            pf.resample(threshold=0.5)
            
            all_particles.append(pf.particles.copy())
            all_weights.append(pf.weights.copy())
        
        return {
            'particles': all_particles,
            'weights': all_weights,
            'times': times
        }
    
    def _heston_likelihood(self, state, observation, dt):
        """Compute likelihood p(S_t | S_{t-1}, v_{t-1})"""
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
        """Propagate particles forward using Heston dynamics"""
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
        
    def analyze_convergence_rate(
        self,
        particle_counts: List[int],
        n_replications: int = 20,
        T: float = 1.0,
        n_steps: int = 252,
        observation_time: float = None
    ) -> Dict:
        """
        Analyze convergence rate of moments with multiple replications.
        
        Args:
            particle_counts: List of particle counts to test
            n_replications: Number of replications per particle count
            T: Time horizon
            n_steps: Number of time steps
            observation_time: Time at which to analyze (default: T)
            
        Returns:
            Dictionary with convergence analysis results
        """
        if observation_time is None:
            observation_time = T
        
        print(f"\n{'='*60}")
        print(f"Moment Stability Analysis")
        print(f"{'='*60}")
        print(f"Replications per sample size: {n_replications}")
        print(f"Observation time: {observation_time}")
        print()
        
        # Generate one true path for all replications
        times = np.linspace(0, T, n_steps + 1)
        path_result = self.heston.simulate_paths(T, n_steps)
        
        # Handle both dict and tuple returns
        if isinstance(path_result, dict):
            S_obs = path_result['S']
        else:
            S_obs = path_result[0]
        
        obs_idx = np.argmin(np.abs(times - observation_time))
        
        results = {
            'particle_counts': particle_counts,
            'n_replications': n_replications,
            'means': {n: [] for n in particle_counts},
            'stds': {n: [] for n in particle_counts},
            'skews': {n: [] for n in particle_counts},
            'kurts': {n: [] for n in particle_counts},
            'times': times,
            'S_obs': S_obs,
            'observation_time': observation_time,
            'obs_idx': obs_idx
        }
        
        for n_particles in particle_counts:
            print(f"\nTesting {n_particles:,} particles...")
            
            for rep in tqdm(range(n_replications), desc="Replications"):
                # Run particle filter
                pf_results = self._run_particle_filter(S_obs, times, n_particles)
                
                # Extract particles at observation time
                particles = pf_results['particles'][obs_idx]
                weights = pf_results['weights'][obs_idx]
                
                v_particles = particles[:, 1]
                v_mean = np.sum(weights * v_particles)
                v_var = np.sum(weights * (v_particles - v_mean)**2)
                v_std = np.sqrt(v_var)
                
                v_centered = v_particles - v_mean
                v_skew = np.sum(weights * v_centered**3) / (v_std**3 + 1e-10)
                v_kurt = np.sum(weights * v_centered**4) / (v_std**4 + 1e-10) - 3
                
                results['means'][n_particles].append(v_mean)
                results['stds'][n_particles].append(v_std)
                results['skews'][n_particles].append(v_skew)
                results['kurts'][n_particles].append(v_kurt)
        
        # Compute statistics across replications
        results['mean_statistics'] = {}
        results['std_statistics'] = {}
        results['skew_statistics'] = {}
        results['kurt_statistics'] = {}
        
        for n_particles in particle_counts:
            results['mean_statistics'][n_particles] = {
                'mean': np.mean(results['means'][n_particles]),
                'std': np.std(results['means'][n_particles]),
                'cv': np.std(results['means'][n_particles]) / (np.mean(results['means'][n_particles]) + 1e-10)
            }
            
            results['std_statistics'][n_particles] = {
                'mean': np.mean(results['stds'][n_particles]),
                'std': np.std(results['stds'][n_particles]),
                'cv': np.std(results['stds'][n_particles]) / (np.mean(results['stds'][n_particles]) + 1e-10)
            }
            
            results['skew_statistics'][n_particles] = {
                'mean': np.mean(results['skews'][n_particles]),
                'std': np.std(results['skews'][n_particles])
            }
            
            results['kurt_statistics'][n_particles] = {
                'mean': np.mean(results['kurts'][n_particles]),
                'std': np.std(results['kurts'][n_particles])
            }
        
        return results
    
    def plot_convergence_analysis(
        self,
        results: Dict,
        save_path: str = None
    ):
        """
        Plot convergence analysis results.
        """
        particle_counts = results['particle_counts']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        moments = ['mean', 'std', 'skew', 'kurt']
        titles = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis']
        
        for idx, (moment, title) in enumerate(zip(moments, titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Extract statistics
            stat_dict = results[f'{moment}_statistics']
            means = [stat_dict[n]['mean'] for n in particle_counts]
            stds = [stat_dict[n]['std'] for n in particle_counts]
            
            # Plot mean estimate with error bars
            ax.errorbar(particle_counts, means, yerr=stds,
                       fmt='o-', capsize=5, capthick=2, linewidth=2,
                       markersize=8, label='Estimate ± Std')
            
            # Add shaded region
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.fill_between(particle_counts, 
                           means_arr - stds_arr, 
                           means_arr + stds_arr,
                           alpha=0.2)
            
            # Format
            ax.set_xscale('log')
            ax.set_xlabel('Number of Particles')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Convergence\n({results["n_replications"]} replications)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
    
    def plot_coefficient_of_variation(
        self,
        results: Dict,
        save_path: str = None
    ):
        """
        Plot coefficient of variation to assess relative stability.
        """
        particle_counts = results['particle_counts']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean CV
        mean_cvs = [results['mean_statistics'][n]['cv'] for n in particle_counts]
        axes[0].loglog(particle_counts, mean_cvs, 'o-', linewidth=2, markersize=8)
        
        # Add reference line (1/sqrt(N))
        if mean_cvs[0] > 0:
            ref_line = np.array(mean_cvs[0]) * np.sqrt(particle_counts[0]) / np.sqrt(particle_counts)
            axes[0].loglog(particle_counts, ref_line, 'k--', alpha=0.5, label='$N^{-1/2}$')
        
        axes[0].set_xlabel('Number of Particles')
        axes[0].set_ylabel('Coefficient of Variation')
        axes[0].set_title('Mean Estimate Relative Stability')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Std CV
        std_cvs = [results['std_statistics'][n]['cv'] for n in particle_counts]
        axes[1].loglog(particle_counts, std_cvs, 'o-', linewidth=2, markersize=8)
        
        # Add reference line
        if std_cvs[0] > 0:
            ref_line = np.array(std_cvs[0]) * np.sqrt(particle_counts[0]) / np.sqrt(particle_counts)
            axes[1].loglog(particle_counts, ref_line, 'k--', alpha=0.5, label='$N^{-1/2}$')
        
        axes[1].set_xlabel('Number of Particles')
        axes[1].set_ylabel('Coefficient of Variation')
        axes[1].set_title('Std Estimate Relative Stability')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
    
    def assess_sufficient_particles(
        self,
        results: Dict,
        cv_threshold: float = 0.01
    ) -> Dict:
        """
        Assess what number of particles is sufficient based on CV threshold.
        
        Args:
            results: Results from analyze_convergence_rate
            cv_threshold: Maximum acceptable coefficient of variation
            
        Returns:
            Dictionary with recommendations
        """
        particle_counts = results['particle_counts']
        
        recommendations = {}
        
        # Check mean CV
        mean_cvs = [results['mean_statistics'][n]['cv'] for n in particle_counts]
        for n, cv in zip(particle_counts, mean_cvs):
            if cv <= cv_threshold:
                recommendations['mean'] = n
                break
        else:
            recommendations['mean'] = None
        
        # Check std CV
        std_cvs = [results['std_statistics'][n]['cv'] for n in particle_counts]
        for n, cv in zip(particle_counts, std_cvs):
            if cv <= cv_threshold:
                recommendations['std'] = n
                break
        else:
            recommendations['std'] = None
        
        print(f"\n{'='*60}")
        print(f"Sufficiency Assessment (CV threshold: {cv_threshold})")
        print(f"{'='*60}\n")
        
        if recommendations['mean'] is not None:
            print(f"Mean estimate stable at: {recommendations['mean']:,} particles")
            print(f"  (CV = {results['mean_statistics'][recommendations['mean']]['cv']:.4f})")
        else:
            print(f"Mean estimate NOT stable within tested range")
            print(f"  (min CV = {min(mean_cvs):.4f} at {particle_counts[-1]:,} particles)")
        
        if recommendations['std'] is not None:
            print(f"Std estimate stable at: {recommendations['std']:,} particles")
            print(f"  (CV = {results['std_statistics'][recommendations['std']]['cv']:.4f})")
        else:
            print(f"Std estimate NOT stable within tested range")
            print(f"  (min CV = {min(std_cvs):.4f} at {particle_counts[-1]:,} particles)")
        
        # Overall recommendation
        if recommendations['mean'] and recommendations['std']:
            recommended = max(recommendations['mean'], recommendations['std'])
            print(f"\nRecommended minimum: {recommended:,} particles")
            recommendations['recommended'] = recommended
        else:
            print(f"\nRecommended: Test higher particle counts")
            recommendations['recommended'] = None
        
        return recommendations
    
    def print_summary(self, results: Dict):
        """
        Print summary statistics.
        """
        particle_counts = results['particle_counts']
        
        print(f"\n{'='*60}")
        print(f"MOMENT STABILITY SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Observation time: {results['observation_time']}")
        print(f"Replications: {results['n_replications']}\n")
        
        for n in particle_counts:
            print(f"\n{n:,} particles:")
            print(f"  Mean: {results['mean_statistics'][n]['mean']:.6f} "
                  f"± {results['mean_statistics'][n]['std']:.6f} "
                  f"(CV: {results['mean_statistics'][n]['cv']:.4f})")
            print(f"  Std:  {results['std_statistics'][n]['mean']:.6f} "
                  f"± {results['std_statistics'][n]['std']:.6f} "
                  f"(CV: {results['std_statistics'][n]['cv']:.4f})")
            print(f"  Skew: {results['skew_statistics'][n]['mean']:.4f} "
                  f"± {results['skew_statistics'][n]['std']:.4f}")
            print(f"  Kurt: {results['kurt_statistics'][n]['mean']:.4f} "
                  f"± {results['kurt_statistics'][n]['std']:.4f}")


def run_stability_analysis():
    """
    Run complete moment stability analysis.
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
    
    analyzer = MomentStabilityAnalyzer(heston_params)
    
    # Test particle counts
    particle_counts = [100, 500, 1000, 5000, 10000] 

    # Run analysis
    results = analyzer.analyze_convergence_rate(
        particle_counts=particle_counts,
        n_replications=20, 
        T=1.0,
        n_steps=100,  
        observation_time=1.0
    )
    
    # Plot results
    analyzer.plot_convergence_analysis(
        results,
        save_path="experiments/results/moment_convergence.png"
    )
    
    analyzer.plot_coefficient_of_variation(
        results,
        save_path="experiments/results/cv_analysis.png"
    )
    
    # Assess sufficiency
    recommendations = analyzer.assess_sufficient_particles(results, cv_threshold=0.02)
    
    # Print summary
    analyzer.print_summary(results)
    
    return results, recommendations


if __name__ == "__main__":
    results, recommendations = run_stability_analysis()
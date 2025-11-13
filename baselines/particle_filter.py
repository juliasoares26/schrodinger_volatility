"""
Baseline Method 3: Particle Filter for Conditional Sampling
Sequential Monte Carlo approach for dynamic inference problems
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class ParticleFilter:
    """
    Sequential Monte Carlo (SMC) for conditional sampling and Bayesian filtering.
    
    The particle filter recursively approximates posterior distributions as 
    observations arrive sequentially:
    
        t=0: Initialize from prior μ
        t=1: Observe y₁ → Update to p(x | y₁)
        t=2: Observe y₂ → Update to p(x | y₁, y₂)
        ...
    
    The posterior at time t is represented by weighted particles {(x_i^t, w_i^t)},
    which approximate the target distribution via importance sampling.
    
    Key Steps
    ---------
    1. **Reweight**: Update particle weights based on likelihood p(y_t | x_i)
    2. **Resample**: Eliminate low-weight particles when ESS drops (degeneracy control)
    3. **Move**: Apply transition kernel to maintain particle diversity
    
    Applications in Volatility Modeling
    ------------------------------------
    - Intraday recalibration as new option prices arrive
    - Real-time tracking of latent volatility dynamics
    - Sequential incorporation of market data
    
    Parameters
    ----------
    n_particles : int, default=1000
        Number of particles in the ensemble
        
    Attributes
    ----------
    particles : ndarray, shape (n_particles, dim)
        Current particle locations
    weights : ndarray, shape (n_particles,)
        Normalized importance weights
        
    Notes
    -----
    The effective sample size (ESS) quantifies weight degeneracy:
        ESS = (∑ w_i)² / ∑ w_i²
    
    ESS ≈ n_particles indicates uniform weights (good)
    ESS << n_particles indicates weight collapse (resampling needed)
    
    References
    ----------
    Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and 
        smoothing: Fifteen years later." Handbook of nonlinear filtering, 12(656-704), 3.
    """
    
    def __init__(self, n_particles=1000):
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
        
    def initialize(self, prior_samples):
        """
        Initialize particle ensemble from prior distribution.
        
        Parameters
        ----------
        prior_samples : ndarray, shape (n_particles, dim)
            Samples drawn from prior distribution μ
        """
        self.particles = prior_samples.copy()
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def effective_sample_size(self):
        """
        Compute effective sample size (ESS) as measure of weight degeneracy.
        
        Returns
        -------
        ess : float
            Effective sample size: ESS = (∑ w_i)² / ∑ w_i²
            
        Notes
        -----
        ESS ranges from 1 (complete collapse) to n_particles (uniform weights).
        Resampling is typically triggered when ESS falls below n_particles/2.
        """
        return (np.sum(self.weights)**2) / np.sum(self.weights**2)
    
    def reweight(self, observation, likelihood_fn):
        """
        Update particle weights via importance sampling using Bayes' rule.
        
        Computes w_i^(t) ∝ w_i^(t-1) · p(y_t | x_i) for each particle.
        
        Parameters
        ----------
        observation : array_like
            New observation y_t
        likelihood_fn : callable
            Function computing p(y | x) for given particle and observation
        """
        likelihoods = np.array([
            likelihood_fn(self.particles[i], observation)
            for i in range(self.n_particles)
        ])
        
        self.weights *= likelihoods
        self.weights /= np.sum(self.weights) + 1e-10
        
    def resample(self, threshold=0.5):
        """
        Resample particles when effective sample size drops below threshold.
        
        Uses systematic resampling, which has lower variance than multinomial
        resampling while maintaining unbiasedness.
        
        Parameters
        ----------
        threshold : float, default=0.5
            Resample when ESS < threshold × n_particles
            
        Returns
        -------
        resampled : bool
            True if resampling was performed
        """
        ess = self.effective_sample_size()
        
        if ess < threshold * self.n_particles:
            print(f"  Resampling (ESS={ess:.0f}/{self.n_particles})")
            
            indices = self._systematic_resample(self.weights)
            
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
            
            return True
        
        return False
    
    def _systematic_resample(self, weights):
        """
        Perform systematic resampling with deterministic stratified sampling.
        
        This method has lower variance than multinomial resampling while
        maintaining the correct marginal distribution.
        
        Parameters
        ----------
        weights : ndarray
            Normalized particle weights
            
        Returns
        -------
        indices : ndarray
            Resampled particle indices
        """
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform()) / n
        
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        
        return indices
    
    def move(self, transition_kernel):
        """
        Apply transition kernel to maintain particle diversity after resampling.
        
        Common choices include:
        - MCMC kernels (e.g., Metropolis-Hastings)
        - Random walk diffusion
        - Deterministic drift plus noise
        
        Parameters
        ----------
        transition_kernel : callable
            Function applying stochastic transition: x' = K(x)
        """
        self.particles = transition_kernel(self.particles)
    
    def estimate_posterior_mean(self):
        """
        Estimate posterior mean via weighted average of particles.
        
        Returns
        -------
        mean : ndarray
            Posterior mean estimate: E[X | y] ≈ ∑ w_i x_i
        """
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def estimate_posterior_cov(self):
        """
        Estimate posterior covariance via weighted sample covariance.
        
        Returns
        -------
        cov : ndarray, shape (dim, dim)
            Posterior covariance estimate
        """
        mean = self.estimate_posterior_mean()
        diff = self.particles - mean
        cov = np.average(
            np.einsum('ni,nj->nij', diff, diff),
            weights=self.weights,
            axis=0
        )
        return cov


if __name__ == "__main__":
    np.random.seed(42)
    
    # Configuration
    dim = 2
    n_particles = 1000
    
    # Prior: N(0, I)
    prior_samples = np.random.randn(n_particles, dim)
    
    # True target: N([3, 3], I) (unknown to filter)
    true_mean = np.array([3.0, 3.0])
    true_cov = np.eye(dim)
    
    # Noisy observations
    n_obs = 5
    true_samples = np.random.multivariate_normal(true_mean, true_cov, n_obs)
    observations = true_samples + np.random.randn(n_obs, dim) * 0.5
    
    print(f"Toy example: Sequential conditioning")
    print(f"  Prior: N(0, I)")
    print(f"  Target: N({true_mean}, I)")
    print(f"  Observations: {n_obs} noisy samples")
    print(f"  Particles: {n_particles}")
    
    # Initialize particle filter
    pf = ParticleFilter(n_particles=n_particles)
    pf.initialize(prior_samples)
    
    # Gaussian likelihood: p(y | x) = N(y | x, noise_cov)
    noise_cov = 0.5**2 * np.eye(dim)
    
    def likelihood_fn(particle, observation):
        """Gaussian likelihood function"""
        diff = observation - particle
        exponent = -0.5 * np.dot(diff, np.linalg.solve(noise_cov, diff))
        normalizer = 1.0 / np.sqrt((2*np.pi)**dim * np.linalg.det(noise_cov))
        return normalizer * np.exp(exponent)
    
    # Random walk transition kernel
    def transition_kernel(particles):
        noise = np.random.randn(*particles.shape) * 0.1
        return particles + noise
    
    # Track filtering history
    history = {
        'particles': [pf.particles.copy()],
        'weights': [pf.weights.copy()],
        'ess': [pf.effective_sample_size()],
        'mean_estimate': [pf.estimate_posterior_mean()]
    }
    
    # Sequential filtering updates
    print("\nSequential updates:")
    for i, obs in enumerate(observations):
        print(f"\n  Observation {i+1}/{n_obs}: {obs}")
        
        pf.reweight(obs, likelihood_fn)
        ess_before = pf.effective_sample_size()
        print(f"    ESS after reweight: {ess_before:.0f}/{n_particles}")
        
        resampled = pf.resample(threshold=0.5)
        
        if resampled:
            pf.move(transition_kernel)
        
        history['particles'].append(pf.particles.copy())
        history['weights'].append(pf.weights.copy())
        history['ess'].append(pf.effective_sample_size())
        history['mean_estimate'].append(pf.estimate_posterior_mean())
    
    # Final estimates
    final_mean = pf.estimate_posterior_mean()
    final_cov = pf.estimate_posterior_cov()
    
    print(f"\nFinal results:")
    print(f"  True mean: {true_mean}")
    print(f"  Estimated mean: {final_mean}")
    print(f"  Error: {np.linalg.norm(final_mean - true_mean):.4f}")
    print(f"  Final ESS: {pf.effective_sample_size():.0f}/{n_particles}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    
    # Evolution of particle distribution
    for i in range(min(4, len(history['particles']))):
        ax = plt.subplot(2, 4, i+1)
        
        idx = i * (len(history['particles']) // 4) if i < 3 else -1
        particles = history['particles'][idx]
        weights = history['weights'][idx]
        
        sizes = weights * 1000 / weights.max()
        ax.scatter(particles[:, 0], particles[:, 1], 
                  s=sizes, alpha=0.5, c='steelblue')
        
        ax.scatter(*true_mean, marker='*', s=500, 
                  color='red', edgecolors='black', 
                  linewidths=2, label='True', zorder=10)
        
        mean_est = history['mean_estimate'][idx]
        ax.scatter(*mean_est, marker='X', s=300,
                  color='orange', edgecolors='black',
                  linewidths=2, label='Estimate', zorder=10)
        
        if idx > 0:
            ax.scatter(observations[:idx, 0], observations[:idx, 1],
                      marker='o', s=100, color='green',
                      edgecolors='black', linewidths=1,
                      label='Observations', zorder=9)
        
        ax.set_xlim(-3, 6)
        ax.set_ylim(-3, 6)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'Step {idx}: ESS={history["ess"][idx]:.0f}')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Diagnostics
    ax = plt.subplot(2, 4, 5)
    ax.plot(history['ess'], 'o-', linewidth=2)
    ax.axhline(0.5*n_particles, color='red', linestyle='--', 
              label='Resample threshold')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('ESS')
    ax.set_title('Effective Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = plt.subplot(2, 4, 6)
    errors = [np.linalg.norm(m - true_mean) for m in history['mean_estimate']]
    ax.plot(errors, 'o-', linewidth=2, color='coral')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('L2 Error')
    ax.set_title('Estimation Error')
    ax.grid(True, alpha=0.3)
    
    ax = plt.subplot(2, 4, 7)
    final_weights_sorted = np.sort(pf.weights)[::-1]
    ax.bar(range(len(final_weights_sorted[:50])), 
           final_weights_sorted[:50])
    ax.set_xlabel('Particle Rank')
    ax.set_ylabel('Weight')
    ax.set_title('Final Weight Distribution (Top 50)')
    ax.grid(True, alpha=0.3)
    
    ax = plt.subplot(2, 4, 8)
    means_array = np.array(history['mean_estimate'])
    ax.plot(means_array[:, 0], 'o-', label='x₁', linewidth=2)
    ax.plot(means_array[:, 1], 's-', label='x₂', linewidth=2)
    ax.axhline(true_mean[0], color='blue', linestyle='--', alpha=0.5)
    ax.axhline(true_mean[1], color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Mean Estimate')
    ax.set_title('Convergence to True Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/synthetic/particle_filter_demo.png', dpi=150)
    plt.show()
    
    print("\nResults saved to: data/synthetic/particle_filter_demo.png")
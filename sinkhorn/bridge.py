import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class SinkhornBridge:
    """
    Schrödinger Bridge via Entropic Optimal Transport.
    
    Computes the optimal stochastic bridge between two probability distributions
    μ (source) and ν (target) by solving the regularized optimal transport problem:
    
        min_{π ∈ Π(μ,ν)} ∫∫ c(x,y) dπ(x,y) + ε KL(π | μ ⊗ ν)
    
    where c(x,y) = ||x-y||²/2 is the squared Euclidean cost and ε > 0 is the
    entropic regularization parameter.
    
    The solution is characterized by dual potentials (f, g) satisfying:
        π(x,y) ∝ exp(-(c(x,y) - f(x) - g(y))/ε)
    
    These potentials are computed via the Sinkhorn algorithm, which alternates
    between projections onto marginal constraints.
    
    Parameters
    ----------
    source_samples : array_like, shape (n_source, d)
        Samples from the source distribution μ
    target_samples : array_like, shape (n_target, d)
        Samples from the target distribution ν
    epsilon : float, default=0.1
        Entropic regularization strength. Smaller values approach unregularized
        optimal transport but require more iterations to converge
    device : str, default='cpu'
        PyTorch device for computation
        
    Attributes
    ----------
    f : torch.Tensor, shape (n_source,)
        Source potential (dual variable)
    g : torch.Tensor, shape (n_target,)
        Target potential (dual variable)
    """
    
    def __init__(self, 
                 source_samples,
                 target_samples,
                 epsilon=0.1,
                 device='cpu'):
        
        self.source = torch.tensor(source_samples, dtype=torch.float32).to(device)
        self.target = torch.tensor(target_samples, dtype=torch.float32).to(device)
        
        self.n_source = len(self.source)
        self.n_target = len(self.target)
        
        self.epsilon = epsilon
        self.device = device
        
        self.f = None
        self.g = None
        
    def compute_cost_matrix(self):
        """
        Compute pairwise squared Euclidean cost matrix.
        
        Returns
        -------
        C : torch.Tensor, shape (n_source, n_target)
            Cost matrix where C[i,j] = ||x_i - y_j||² / 2
        """
        diff = self.source.unsqueeze(1) - self.target.unsqueeze(0)
        C = 0.5 * torch.sum(diff ** 2, dim=-1)
        return C
    
    def sinkhorn_iterations(self, max_iter=1000, tol=1e-6):
        """
        Solve entropic optimal transport via Sinkhorn's fixed-point algorithm.
        
        The algorithm alternates between updates to the dual potentials:
            f^{k+1} = -ε log(∑_j exp(-(C_ij - g^k_j)/ε))
            g^{k+1} = -ε log(∑_i exp(-(C_ij - f^{k+1}_i)/ε))
        
        This is equivalent to alternating projections in log-domain (Sinkhorn-Knopp).
        
        Parameters
        ----------
        max_iter : int, default=1000
            Maximum number of Sinkhorn iterations
        tol : float, default=1e-6
            Convergence tolerance on change in scaling variables
            
        Returns
        -------
        f : torch.Tensor, shape (n_source,)
            Converged source potential
        g : torch.Tensor, shape (n_target,)
            Converged target potential
        """
        C = self.compute_cost_matrix()
        
        f = torch.zeros(self.n_source, device=self.device)
        g = torch.zeros(self.n_target, device=self.device)
        
        K = torch.exp(-C / self.epsilon)
        
        # Uniform marginals (can be generalized to weighted case)
        a = torch.ones(self.n_source, device=self.device) / self.n_source
        b = torch.ones(self.n_target, device=self.device) / self.n_target
        
        # Scaling variables in Sinkhorn-Knopp algorithm
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        for iteration in tqdm(range(max_iter), desc="Sinkhorn"):
            u_prev = u.clone()
            
            u = a / (K @ v + 1e-10)
            v = b / (K.t() @ u + 1e-10)
            
            error = torch.max(torch.abs(u - u_prev))
            if error < tol:
                print(f"Converged at iteration {iteration}")
                break
        
        # Convert scaling variables to potentials
        self.f = -self.epsilon * torch.log(u + 1e-10)
        self.g = -self.epsilon * torch.log(v + 1e-10)
        
        return self.f, self.g
    
    def compute_drift(self, x, t):
        """
        Compute drift coefficient of the Schrödinger bridge SDE at point (x, t).
        
        The bridge process satisfies the SDE:
            dX_t = b_t(X_t) dt + dW_t
        
        where the drift is given by:
            b_t(x) = ∇_x log E[exp(-f(X_T)/ε) | X_t = x]
        
        This requires computing conditional expectations, typically approximated
        via kernel methods or neural networks.
        
        Parameters
        ----------
        x : torch.Tensor
            Current position
        t : float
            Current time
            
        Returns
        -------
        drift : torch.Tensor
            Drift vector at (x, t)
            
        Notes
        -----
        Implementation requires approximation of conditional expectations.
        Common approaches include:
        - Kernel density estimation
        - Neural network function approximation
        - Particle-based methods
        """
        raise NotImplementedError("Drift computation requires conditional expectation approximation")
    
    def sample_bridge(self, n_steps=100, n_samples=1000):
        """
        Sample trajectories from the Schrödinger bridge via Euler-Maruyama discretization.
        
        Discretizes the bridge SDE:
            X_{t+dt} = X_t + b_t(X_t) dt + √dt * Z
        where Z ~ N(0, I).
        
        Parameters
        ----------
        n_steps : int, default=100
            Number of time discretization steps
        n_samples : int, default=1000
            Number of bridge trajectories to sample
            
        Returns
        -------
        trajectories : torch.Tensor, shape (n_samples, n_steps, d)
            Sampled bridge trajectories
            
        Notes
        -----
        Requires implementation of compute_drift method.
        """
        raise NotImplementedError("Sampling requires drift computation to be implemented")


if __name__ == "__main__":
    # Toy example: Gaussian to Gaussian transport
    np.random.seed(42)
    
    # Source: N(0, I)
    source = np.random.randn(1000, 2)
    
    # Target: N([3, 3], [[1, 0.5], [0.5, 1]])
    mean_target = np.array([3, 3])
    cov_target = np.array([[1, 0.5], [0.5, 1]])
    target = np.random.multivariate_normal(mean_target, cov_target, 1000)
    
    bridge = SinkhornBridge(source, target, epsilon=0.1)
    
    print("Computing Sinkhorn potentials...")
    f, g = bridge.sinkhorn_iterations(max_iter=500)
    
    print(f"Potentials computed successfully")
    print(f"f shape: {f.shape}")
    print(f"g shape: {g.shape}")
    
    # Visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(source[:, 0], source[:, 1], alpha=0.5, label='Source')
    axes[0].set_title('Source Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(target[:, 0], target[:, 1], alpha=0.5, color='orange', label='Target')
    axes[1].set_title('Target Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(source[:, 0], source[:, 1], alpha=0.3, label='Source')
    axes[2].scatter(target[:, 0], target[:, 1], alpha=0.3, color='orange', label='Target')
    axes[2].set_title('Both Distributions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sinkhorn_test.png')
    plt.show()
    
    print("\nVisualization saved to: sinkhorn_test.png")
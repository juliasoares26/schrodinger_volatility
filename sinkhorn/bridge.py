import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        
        for iteration in range(max_iter):
            u_prev = u.clone()
            
            u = a / (K @ v + 1e-10)
            v = b / (K.t() @ u + 1e-10)
            
            error = torch.max(torch.abs(u - u_prev))
            if error < tol:
                if iteration % 100 == 0 or iteration < 10:
                    print(f"Sinkhorn converged at iteration {iteration}")
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


class DriftNetwork(nn.Module):
    """
    Neural network to approximate the drift coefficient of the bridge SDE.
    
    Parameters
    ----------
    dim : int
        Dimension of the state space
    hidden_dim : int, default=64
        Hidden layer dimension
    n_layers : int, default=3
        Number of hidden layers
    """
    
    def __init__(self, dim, hidden_dim=64, n_layers=3):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(dim + 1, hidden_dim))  # +1 for time
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, t):
        """
        Compute drift at position x and time t.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, dim)
            Positions
        t : torch.Tensor, shape (batch_size, 1)
            Times
            
        Returns
        -------
        drift : torch.Tensor, shape (batch_size, dim)
            Drift coefficients
        """
        # Concatenate position and time
        xt = torch.cat([x, t], dim=-1)
        return self.network(xt)


class SchrodingerBridge:
    """
    Schrödinger Bridge with Neural Network Drift Approximation.
    
    This class implements a trainable Schrödinger bridge that:
    1. Uses Sinkhorn algorithm to compute optimal transport potentials
    2. Learns drift via neural network
    3. Enables sampling from the bridge distribution
    
    Parameters
    ----------
    dim : int
        Dimension of the state space
    hidden_dim : int, default=64
        Hidden dimension for drift network
    n_layers : int, default=3
        Number of layers in drift network
    device : str, default='cpu'
        PyTorch device
        
    Attributes
    ----------
    drift_net : DriftNetwork
        Neural network approximating drift
    """
    
    def __init__(self, dim, hidden_dim=64, n_layers=3, device='cpu'):
        self.dim = dim
        self.device = device
        
        # Initialize drift network
        self.drift_net = DriftNetwork(dim, hidden_dim, n_layers).to(device)
        
        # Sinkhorn bridge (will be initialized during training)
        self.sinkhorn = None
        
    def train_bridge(self, 
                    source_samples,
                    target_samples,
                    n_iterations=100,
                    lr=1e-3,
                    batch_size=256,
                    epsilon=0.1):
        """
        Train the bridge to transport source to target distribution.
        
        Parameters
        ----------
        source_samples : array_like, shape (n_source, dim)
            Samples from source distribution
        target_samples : array_like, shape (n_target, dim)
            Samples from target distribution
        n_iterations : int, default=100
            Number of training iterations
        lr : float, default=1e-3
            Learning rate
        batch_size : int, default=256
            Batch size for training
        epsilon : float, default=0.1
            Entropic regularization for Sinkhorn
            
        Returns
        -------
        losses : list
            Training losses
        """
        # First, compute Sinkhorn potentials
        print("Computing Sinkhorn potentials...")
        self.sinkhorn = SinkhornBridge(
            source_samples,
            target_samples,
            epsilon=epsilon,
            device=self.device
        )
        
        f, g = self.sinkhorn.sinkhorn_iterations(max_iter=500, tol=1e-6)
        
        # Convert to tensors
        source = torch.tensor(source_samples, dtype=torch.float32).to(self.device)
        target = torch.tensor(target_samples, dtype=torch.float32).to(self.device)
        
        # Train drift network
        print("Training drift network...")
        optimizer = optim.Adam(self.drift_net.parameters(), lr=lr)
        
        losses = []
        
        for iteration in range(n_iterations):
            # Sample batch
            idx_s = np.random.choice(len(source), min(batch_size, len(source)), replace=True)
            idx_t = np.random.choice(len(target), min(batch_size, len(target)), replace=True)
            
            x_source = source[idx_s]
            x_target = target[idx_t]
            
            # Sample random times
            t = torch.rand(len(x_source), 1).to(self.device)
            
            # Linear interpolation between source and target
            x_t = (1 - t) * x_source + t * x_target
            
            # Compute drift
            drift = self.drift_net(x_t, t)
            
            # Loss: match expected displacement
            expected_displacement = x_target - x_source
            loss = torch.mean((drift - expected_displacement) ** 2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration+1}/{n_iterations}, Loss: {loss.item():.6f}")
        
        return losses
    
    def sample_bridge(self, source_samples, n_steps=50):
        """
        Sample from the trained bridge starting from source samples.
        
        Uses Euler-Maruyama discretization of the bridge SDE:
            dX_t = b(X_t, t) dt + dW_t
        
        Parameters
        ----------
        source_samples : array_like, shape (n_samples, dim)
            Starting positions (from source distribution)
        n_steps : int, default=50
            Number of discretization steps
            
        Returns
        -------
        trajectories : torch.Tensor, shape (n_samples, n_steps+1, dim)
            Sampled trajectories
        """
        source = torch.tensor(source_samples, dtype=torch.float32).to(self.device)
        n_samples = len(source)
        
        # Initialize trajectories
        trajectories = torch.zeros(n_samples, n_steps + 1, self.dim).to(self.device)
        trajectories[:, 0, :] = source
        
        dt = 1.0 / n_steps
        
        self.drift_net.eval()
        with torch.no_grad():
            for step in range(n_steps):
                t_current = step * dt
                t_tensor = torch.full((n_samples, 1), t_current).to(self.device)
                
                x_current = trajectories[:, step, :]
                
                # Compute drift
                drift = self.drift_net(x_current, t_tensor)
                
                # Brownian increment
                dW = torch.randn_like(x_current) * np.sqrt(dt)
                
                # Euler-Maruyama step
                x_next = x_current + drift * dt + dW
                
                trajectories[:, step + 1, :] = x_next
        
        return trajectories


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
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class HestonModel:
    """
    Heston stochastic volatility model implementation.
    
    The Heston model describes the evolution of an asset price S_t and its variance v_t
    through the following SDEs:
        dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
        dv_t = kappa(theta - v_t) dt + sigma sqrt(v_t) dW2_t
    where dW1_t and dW2_t are correlated Brownian motions with correlation rho.
    
    Parameters
    ----------
    S0 : float
        Initial asset price
    v0 : float
        Initial variance (not volatility)
    kappa : float
        Mean reversion speed for variance
    theta : float
        Long-term mean variance
    sigma : float
        Volatility of volatility (vol-of-vol)
    rho : float
        Correlation between asset returns and variance
    r : float
        Risk-free rate
    """
    
    def __init__(self, 
                 S0=100, 
                 v0=0.04, 
                 kappa=2.0, 
                 theta=0.04, 
                 sigma=0.3,
                 rho=-0.7, 
                 r=0.0):
        
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r

    def simulate_paths(self, T=1.0, n_steps=252, n_paths=10000, seed=42):
        """
        Simulate asset price paths using the Euler-Maruyama discretization scheme.
        
        Parameters
        ----------
        T : float
            Time horizon in years
        n_steps : int
            Number of time steps
        n_paths : int
            Number of Monte Carlo paths
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        S : ndarray, shape (n_paths, n_steps+1)
            Simulated asset price paths
        v : ndarray, shape (n_paths, n_steps+1)
            Simulated variance paths
        t : ndarray, shape (n_steps+1,)
            Time grid
        """
        np.random.seed(seed)
        
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)
            
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            v_current = np.maximum(v[:, i], 0) 
            
            v[:, i+1] = v_current + self.kappa * (self.theta - v_current) * dt \
                        + self.sigma * np.sqrt(v_current * dt) * W2
            
            S[:, i+1] = S[:, i] * np.exp(
                (self.r - 0.5 * v_current) * dt + np.sqrt(v_current * dt) * W1
            )
        
        return S, v, t
    
    def option_prices_mc(self, strikes, T=1.0, n_paths=100000):
        """
        Price European call options via Monte Carlo simulation.
        
        Parameters
        ----------
        strikes : array_like
            Strike prices for call options
        T : float
            Time to maturity in years
        n_paths : int
            Number of Monte Carlo paths
            
        Returns
        -------
        call_prices : ndarray
            Call option prices for each strike
        """
        S, _, _ = self.simulate_paths(T=T, n_paths=n_paths)
        S_T = S[:, -1]
        
        call_prices = np.array([
            np.exp(-self.r * T) * np.mean(np.maximum(S_T - K, 0))
            for K in strikes
        ])
        
        return call_prices
    
    def implied_volatility_surface(self,
                                   strike_range=(0.8, 1.2),
                                   n_strikes=20,
                                   maturities=[0.25, 0.5, 1.0, 2.0]):
        """
        Generate implied volatility surface using Monte Carlo pricing and
        Newton-Raphson inversion.
        
        Parameters
        ----------
        strike_range : tuple
            Range of strikes as (min_ratio, max_ratio) relative to S0
        n_strikes : int
            Number of strike points to compute
        maturities : list
            List of option maturities in years
            
        Returns
        -------
        strikes_norm : ndarray
            Normalized strikes (K/S0)
        surface : dict
            Dictionary mapping maturity to implied volatility array
        """
        strikes = np.linspace(strike_range[0] * self.S0,
                             strike_range[1] * self.S0,
                             n_strikes)
        
        surface = {}
        
        for T in maturities:
            call_prices = self.option_prices_mc(strikes, T=T)
            
            impl_vols = np.array([
                self._implied_vol_newton(call_prices[i], strikes[i], T)
                for i in range(len(strikes))
            ])
            
            surface[T] = impl_vols
        
        return strikes / self.S0, surface
    
    def _implied_vol_newton(self, price, K, T, tol=1e-6, max_iter=100):
        """
        Compute implied volatility via Newton-Raphson method on Black-Scholes formula.
        
        Parameters
        ----------
        price : float
            Observed call option price
        K : float
            Strike price
        T : float
            Time to maturity
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum number of iterations
            
        Returns
        -------
        sigma : float
            Implied volatility
        """
        sigma = 0.3
        
        for _ in range(max_iter):
            d1 = (np.log(self.S0 / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            bs_price = self.S0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            vega = self.S0 * norm.pdf(d1) * np.sqrt(T)
            
            diff = bs_price - price
            
            if abs(diff) < tol:
                return sigma
            
            if vega > 1e-10:
                sigma = sigma - diff / vega
            
            sigma = np.clip(sigma, 0.01, 2.0)
        
        return sigma


if __name__ == "__main__":
    heston = HestonModel(
        S0=100,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7
    )
    
    S, v, t = heston.simulate_paths(T=1.0, n_steps=252, n_paths=5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for i in range(5):
        ax1.plot(t, S[i, :], alpha=0.7, label=f'Path {i+1}')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Stock Price')
    ax1.set_title('Heston Model: Sample Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i in range(5):
        ax2.plot(t, np.sqrt(v[i, :]), alpha=0.7)
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility')
    ax2.set_title('Volatility Paths')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heston_sample_paths.png', dpi=150)
    plt.show()
    
    print("Heston model implemented and tested successfully")
    print(f"Initial price: {heston.S0}")
    print(f"Initial volatility: {np.sqrt(heston.v0)*100:.1f}%")
    print(f"Long-term volatility: {np.sqrt(heston.theta)*100:.1f}%")
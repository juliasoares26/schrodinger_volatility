import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys

#Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from heston import HestonUnified


#Optimize Schrödinger potential with numerical stability
class SchrodingerPotentialOptimizer:
    
    def __init__(self, strikes, market_prices, heston):
        self.strikes = strikes
        self.market_prices = market_prices
        self.heston = heston
        self.n_strikes = len(strikes)
        self.weights = np.zeros(self.n_strikes)

    #f(S) = Σ ωα (S - Kα)+ with clipping
    def potential(self, S, weights):
        f = np.zeros_like(S)
        for i, K in enumerate(self.strikes):
            f += weights[i] * np.maximum(S - K, 0)
        return np.clip(f, -50, 50)
    
    #Objective with strong regularization
    def objective(self, weights):
        market_term = np.sum(weights * self.market_prices)
        
        n_mc = 2000
        S, _, _ = self.heston.simulate_paths(T=1.0, n_steps=30, n_paths=n_mc)
        S_T = S[:, -1]
        
        f_vals = self.potential(S_T, weights)
        u_value = -np.log(np.mean(np.exp(-f_vals)) + 1e-10)
        
        #Stronger L2 regularization
        regularization = 0.1 * np.sum(weights**2)
        
        return -(market_term - u_value) + regularization
    
    def optimize(self, max_iter=50):
        print("Optimizing Schrodinger Potential")
        
        #More restricted bounds
        bounds = [(-5, 5) for _ in range(self.n_strikes)]
        
        result = minimize(
            self.objective,
            self.weights,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        self.weights = result.x
        
        print(f"\nOptimization complete")
        print(f"Final objective: {-result.fun:.6f}")
        print(f"Iterations: {result.nit}")
        print(f"Weights: min={self.weights.min():.4f}, max={self.weights.max():.4f}")
        
        if np.abs(self.weights).max() > 3:
            print(f"Warning: Large weights detected!")
            print(f"Suggestion: Increase regularization or reduce bounds")
        
        return self.weights

#Network without BatchNorm for stability
class VolatilityDriftNetwork(nn.Module):
    
    def __init__(self, hidden_dim=64, n_layers=2, dropout=0.0):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.Tanh()
        )
        
        self.state_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.Tanh()
        )
        
        layers = []
        current_dim = hidden_dim // 4 + hidden_dim // 2
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, t, S, v):
        t_norm = t / 1.0
        S_norm = torch.log(S / 100.0 + 1e-8)
        v_norm = torch.sqrt(v + 1e-8) / 0.2
        
        t_emb = self.time_embed(t_norm.unsqueeze(-1))
        state_emb = self.state_embed(torch.cat([S_norm.unsqueeze(-1), 
                                                  v_norm.unsqueeze(-1)], dim=-1))
        
        x = torch.cat([t_emb, state_emb], dim=-1)
        drift = self.network(x).squeeze(-1)
        
        return torch.clamp(drift, -5, 5)


class MartingaleSchrodingerBridge:
    
    def __init__(self, heston, device='cpu'):
        self.heston = heston
        self.device = device
        self.drift_net = VolatilityDriftNetwork().to(device)
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train(
        self,
        strikes,
        market_prices,
        T=1.0,
        n_iterations=1000,
        batch_size=128,
        lr=5e-4,
        patience=100
    ):

        print("MSB Calibration")
        
        #Optimize potential
        potential_opt = SchrodingerPotentialOptimizer(
            strikes, market_prices, self.heston
        )
        weights = potential_opt.optimize(max_iter=50)
        
        if np.abs(weights).max() > 10:
            print("\n ERROR: Potential weights diverged!")
            print("Suggestion: Reduce n_paths or adjust regularization")
            return []
        
        #Generate training data
        print(f"\n Generating training data...")
        n_paths = 5000
        n_steps = 50
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        S_paths, v_paths, _ = self.heston.simulate_paths(
            T, n_steps, n_paths, seed=42
        )
        
        v_paths = np.maximum(v_paths, 1e-8)
        
        print(f"{n_paths} paths with {n_steps} steps")
        print(f"S range: [{S_paths.min():.2f}, {S_paths.max():.2f}]")
        print(f"v range: [{v_paths.min():.6f}, {v_paths.max():.6f}]")
        
        #Training
        optimizer = optim.AdamW(
            self.drift_net.parameters(),
            lr=lr * 0.1,
            weight_decay=1e-4
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50
        )
        
        self.drift_net.train()
        losses = []
        
        pbar = tqdm(range(n_iterations), desc="Training MSB")
        
        for iteration in pbar:
            path_idx = np.random.choice(n_paths, batch_size, replace=False)
            time_idx = np.random.choice(n_steps, batch_size, replace=True)
            
            t = torch.tensor(times[time_idx], dtype=torch.float32).to(self.device)
            S = torch.tensor(S_paths[path_idx, time_idx], dtype=torch.float32).to(self.device)
            v = torch.tensor(v_paths[path_idx, time_idx], dtype=torch.float32).to(self.device)
            S_T = torch.tensor(S_paths[path_idx, -1], dtype=torch.float32).to(self.device)
            
            v = torch.clamp(v, min=1e-8)
            S = torch.clamp(S, min=0.01)
            
            lambda_pred = self.drift_net(t, S, v)
            
            f_T = torch.zeros_like(S_T)
            for i, K in enumerate(strikes):
                f_T += weights[i] * torch.relu(S_T - K)
            f_T = torch.clamp(f_T, -50, 50)
            
            loss_potential = torch.mean(f_T)
            loss_drift = 0.5 * torch.mean(lambda_pred**2)
            loss_entropy = 0.01 * torch.mean(torch.abs(lambda_pred))
            
            loss = loss_potential + loss_drift + loss_entropy
            
            if torch.isnan(loss) or torch.isnan(lambda_pred).any():
                print(f"\n Error: NaN detected at iteration {iteration}")
                print(f"t: min={t.min():.4f}, max={t.max():.4f}")
                print(f"S: min={S.min():.4f}, max={S.max():.4f}")
                print(f"v: min={v.min():.6f}, max={v.max():.6f}")
                if not torch.isnan(lambda_pred).all():
                    print(f"lambda_pred: min={lambda_pred.min():.4f}, max={lambda_pred.max():.4f}")
                print(f"f_T: min={f_T.min():.4f}, max={f_T.max():.4f}")
                print("\n Possible cause: Invalid inputs or exploding gradients")
                break
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.drift_net.parameters(), 0.5)
            optimizer.step()
            scheduler.step(loss)
            
            losses.append(loss.item())
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                print(f"\n Early stopping at iteration {iteration}")
                break
            
            if iteration % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'potential': f'{loss_potential.item():.4f}',
                    'drift': f'{lambda_pred.abs().mean().item():.3f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        print(f"\n MSB training complete")
        if len(losses) > 0:
            print(f"Final loss = {losses[-1]:.6f}")
            print(f"Total iterations = {len(losses)}")
        else:
            print(f"Training failed immediately")
        return losses
    
    @torch.no_grad()
    def price_option(self, K, T, n_paths=5000):
        """Price option with validation"""
        self.drift_net.eval()
        
        n_steps = 50
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        S = torch.ones((n_paths, n_steps + 1), device=self.device) * self.heston.S0
        v = torch.ones((n_paths, n_steps + 1), device=self.device) * self.heston.v0
        
        for i in range(n_steps):
            t = torch.full((n_paths,), times[i], device=self.device)
            lambda_t = self.drift_net(t, S[:, i], v[:, i])
            
            dW = torch.randn(n_paths, device=self.device) * np.sqrt(dt)
            dZ_indep = torch.randn(n_paths, device=self.device) * np.sqrt(dt)
            dZ = self.heston.rho * dW + np.sqrt(1 - self.heston.rho**2) * dZ_indep
            
            v_current = torch.clamp(v[:, i], min=1e-8)
            drift_v = self.heston.kappa * (self.heston.theta - v_current) + \
                      (1 - self.heston.rho**2) * lambda_t
            dv = drift_v * dt + self.heston.sigma * torch.sqrt(v_current) * dZ
            v[:, i + 1] = torch.clamp(v_current + dv, min=1e-8)
            
            dS = S[:, i] * torch.sqrt(v_current) * dW
            S[:, i + 1] = torch.clamp(S[:, i] + dS, min=0.01)
        
        S_T = S[:, -1]
        payoff = torch.relu(S_T - K)
        
        price = payoff.mean().item()
        std_err = payoff.std().item() / np.sqrt(n_paths)
        
        if np.isnan(price) or price < 0:
            print(f"Invalid price for K={K}: {price}")
            return 0.0, 0.0
        
        return price, std_err

#Conditional Brenier Bridge

#Network for conditional transport map T_2(x2; x1)
#T(x) = [T_1(x1), T_2(x2; x1)]
#T_2(X2; x1) ~ μ_{2|1}(·|T_1(x1))
class ConditionalBrenierNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        
        #Conditioning encoder: x1 -> context
        self.condition_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        #Noise processor: x2 (noise) + context -> output
        self.map_network = []
        input_dim = 2 + hidden_dim // 2  #noise (2D) + context
        
        for _ in range(n_layers):
            self.map_network.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.map_network.append(nn.Linear(hidden_dim, 2))  #Output: [return, vol]
        self.map_network = nn.Sequential(*self.map_network)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    

    #Args:
    #x1_features: (batch, feature_dim) - past market features
    #x2_noise: (batch, 2) - noise from reference distribution

    #Returns:
    #x2_pred: (batch, 2) - predicted [future_return, future_vol]

    def forward(self, x1_features, x2_noise):
        context = self.condition_encoder(x1_features)
        combined = torch.cat([x2_noise, context], dim=-1)
        x2_pred = self.map_network(combined)
        return x2_pred


#Conditional Brenier Maps for Price Prediction.
#Rescaled cost c_t 
#Entropic OT 
#Conditional simulation 

class ConditionalBrenierBridge:
    def __init__(self, feature_dim, device='cpu'):
        self.device = device
        self.feature_dim = feature_dim
        self.transport_net = ConditionalBrenierNetwork(feature_dim).to(device)
        self.best_loss = float('inf')


#Compute rescaled cost matrix C_t.
#A_t = diag(1_{d1}, √t·1_{d2})
#c_t(x,y) = ½‖A_t(x-y)‖²
     
    def compute_rescaled_cost(self, X1, X2, t):
    
        d1 = X1.shape[1]
        d2 = X2.shape[1]
        
        #Rescaling: features at scale 1, targets at scale √t
        X1_scaled = X1  #No scaling for features
        X2_scaled = X2 * np.sqrt(t)  #Scale targets
        
        #Pairwise squared distances
        X1_sq = np.sum(X1_scaled**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2_scaled**2, axis=1, keepdims=True)
        cross = X1_scaled @ X2_scaled.T
        
        cost = (X1_sq + X2_sq.T - 2*cross) / 2.0
        return cost
    
     
    #Compute Sinkhorn weights for entropic OT.
    #π_ε(x,y) = exp((f_ε(x) + g_ε(y) - c_t(x,y))/ε)

    def sinkhorn_weights(self, X1, X2, t, epsilon, n_iter=100):
        C = self.compute_rescaled_cost(X1, X2, t)
        n, m = C.shape
        
        K = np.exp(-C / epsilon)
        u = np.ones(n) / n
        v = np.ones(m) / m
        
        for _ in range(n_iter):
            u = 1.0 / (K @ v + 1e-10)
            v = 1.0 / (K.T @ u + 1e-10)
        
        #Transport plan
        P = u[:, np.newaxis] * K * v[np.newaxis, :]
        return P
    
    def train(
        self,
        X1_train,
        X2_train,
        t=0.01,
        epsilon=0.002,
        n_iterations=2000,
        batch_size=256,
        lr=1e-3
    ):
        
    #Train conditional Brenier map
    #algorithm
    #Use rescaled cost with parameter t
    #Entropic regularization ε ∝ t²
    #Sample from reference ρ2 = N(0,I)
        
       
        print("Condition brenier bridge")
        print(f"Dataset: {len(X1_train)} samples")
        print(f"Features: {X1_train.shape[1]}")
        print(f"Targets: {X2_train.shape[1]}")
        print(f"Rescaling t: {t}")
        print(f"Entropic ε: {epsilon}")
        
        #Normalize data
        X1_mean = X1_train.mean(axis=0)
        X1_std = X1_train.std(axis=0) + 1e-8
        X1_normalized = (X1_train - X1_mean) / X1_std
        
        X2_mean = X2_train.mean(axis=0)
        X2_std = X2_train.std(axis=0) + 1e-8
        X2_normalized = (X2_train - X2_mean) / X2_std
        
        self.X1_mean = X1_mean
        self.X1_std = X1_std
        self.X2_mean = X2_mean
        self.X2_std = X2_std
        
        print(f"\n Data statistics:")
        print(f"X1 mean: {X1_mean[:3]} ...")
        print(f"X1 std:  {X1_std[:3]} ...")
        print(f"X2 mean: {X2_mean}")
        print(f"X2 std:  {X2_std}")
        
        #Convert to tensors
        X1_torch = torch.tensor(X1_normalized, dtype=torch.float32).to(self.device)
        X2_torch = torch.tensor(X2_normalized, dtype=torch.float32).to(self.device)
        
        #Optimizer
        optimizer = optim.Adam(self.transport_net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100
        )
        
        self.transport_net.train()
        losses = []
        
        pbar = tqdm(range(n_iterations), desc="Training Conditional Bridge")
        
        for iteration in pbar:
            #Sample batch
            idx = np.random.choice(len(X1_train), batch_size, replace=False)
            x1_batch = X1_torch[idx]
            x2_batch = X2_torch[idx]
            
            #Sample from reference distribution ρ2 = N(0,I)
            noise = torch.randn(batch_size, 2, device=self.device)
            
            #Predict transport
            x2_pred = self.transport_net(x1_batch, noise)
            
            #Loss: Wasserstein-2 approximation
            #L2 distance between predicted and target
            loss_transport = torch.mean((x2_pred - x2_batch)**2)
            
            #Regularization: encourage smooth transport
            loss_reg = 0.01 * torch.mean(x2_pred**2)
            
            loss = loss_transport + loss_reg
            
            if torch.isnan(loss):
                print(f"\n NaN detected at iteration {iteration}")
                print(f"x1_batch: min={x1_batch.min():.4f}, max={x1_batch.max():.4f}")
                print(f"x2_batch: min={x2_batch.min():.4f}, max={x2_batch.max():.4f}")
                print(f"x2_pred: min={x2_pred.min():.4f}, max={x2_pred.max():.4f}")
                break
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transport_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)
            
            losses.append(loss.item())
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
            
            if iteration % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'transport': f'{loss_transport.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        print(f"\nConditional Bridge training complete")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Total iterations: {len(losses)}")
        
        return losses
    
     #Sample from conditional distribution μ(x2|x1).
     #T_2(X2; x1) ~ μ_{2|1}(·|x1) for X2 ~ ρ2
     #Returns:
     #samples: (n_samples, 2) array of [future_return, future_vol]

    @torch.no_grad()
    def predict_conditional(self, x1_query, n_samples=1000):
        self.transport_net.eval()
        
        #Normalize query
        x1_norm = (x1_query - self.X1_mean) / self.X1_std
        x1_torch = torch.tensor(x1_norm, dtype=torch.float32).to(self.device)
        x1_batch = x1_torch.unsqueeze(0).repeat(n_samples, 1)
        
        #Sample from reference
        noise = torch.randn(n_samples, 2, device=self.device)
        
        #Push forward through transport map
        x2_pred_norm = self.transport_net(x1_batch, noise)
        
        #Denormalize
        x2_pred = x2_pred_norm.cpu().numpy() * self.X2_std + self.X2_mean
        
        return x2_pred
    
    #Predict conditional mean E[X2|X1=x1]
    def predict_mean(self, x1_query):
        
        samples = self.predict_conditional(x1_query, n_samples=5000)
        return samples.mean(axis=0)
    
    #Predict conditional quantile
    def predict_quantile(self, x1_query, quantile=0.05):
        samples = self.predict_conditional(x1_query, n_samples=5000)
        return np.quantile(samples, quantile, axis=0)

#Unified Workflow
def load_synthetic_data():

    possible_paths = [
        Path('data/unified_heston_prediction_data.npz'),
        Path('../data/unified_heston_prediction_data.npz'),
        Path('../../data/unified_heston_prediction_data.npz'),
    ]
    
    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break
    
    if data_file is None:
        print("\n Error: unified_heston_prediction_data.npz not found")
        print("Run generate_unified_data.py first")
        exit(1)
    
    data = np.load(data_file, allow_pickle=True)
    
    print(f"\n Data loaded from: {data_file}")
    print(f"Available datasets: {list(data.keys())}")
    
    return data


def run_option_pricing_calibration(data):
    
    print(f"\n{'='*70}")
    print("Option Pricing with MSB")
    print(f"{'='*70}")
    
    #Extract volatility surface data
    vol_surface = data['vol_surface']
    strikes_norm = data['strikes_norm']
    maturities = data['maturities']
    heston_params = data['heston_params'].item()
    
    #Create Heston model
    heston = HestonUnified(**heston_params)
    
    #Use 1Y maturity
    T_idx = np.argmin(np.abs(maturities - 1.0))
    T = maturities[T_idx]
    
    strikes = strikes_norm * heston.S0
    ivs_market = vol_surface[T_idx, :]
    
    #Filter structural breaks
    mask = np.ones(len(ivs_market), dtype=bool)
    for i in range(1, len(ivs_market)-1):
        if ivs_market[i] < 0.15 and ivs_market[i-1] > 0.18:
            print(f"Detected structural break at K={strikes[i]:.0f}")
            print(f"IV jumps from {ivs_market[i-1]*100:.1f}% → {ivs_market[i]*100:.1f}%")
            mask[i:] = False
            break
    
    strikes = strikes[mask]
    strikes_norm = strikes_norm[mask]
    ivs_market = ivs_market[mask]
    
    #Convert IV to prices
    def bs_price(S, K, T, sigma):
        d1 = (np.log(S/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*norm.cdf(d2)
    
    market_prices = np.array([
        bs_price(heston.S0, K, T, iv)
        for K, iv in zip(strikes, ivs_market)
    ])
    
    print(f"\n  Calibrating for T={T}Y")
    print(f"Strikes: {len(strikes)} (after filtering)")
    print(f"ATM IV: {ivs_market[len(ivs_market)//2]*100:.2f}%")
    print(f"IV range: {ivs_market.min()*100:.2f}% - {ivs_market.max()*100:.2f}%")
    
    #Debug: Show data structure
    print(f"\n  Data structure:")
    print(f"  Strike | IV (%) | Price")
    print(f"  " + "-"*40)
    for K, iv, price in zip(strikes[:5], ivs_market[:5], market_prices[:5]):
        print(f"  {K:6.1f} | {iv*100:6.2f} | {price:8.4f}")
    if len(strikes) > 5:
        print(f"({len(strikes)-5} more strikes)")
    
    #Train MSB
    msb = MartingaleSchrodingerBridge(heston)
    losses_msb = msb.train(
        strikes=strikes,
        market_prices=market_prices,
        T=T,
        n_iterations=500,
        batch_size=128
    )
    
    return msb, losses_msb, strikes, market_prices, ivs_market, T


def run_price_prediction_with_brenier(data):
    print("Price Prediction with Conditional Brenier")
    
    #Extract price prediction data
    X1_train = data['full_X1']
    X2_train = data['full_X2']
    
    print(f"\n  Training set: {len(X1_train)} samples")
    print(f"Features: {X1_train.shape[1]}")
    print(f"Targets: {X2_train.shape[1]} (return, vol, max_dd)")
    
    #Use only return and vol for now
    X2_train = X2_train[:, :2]
    
    #Select t and epsilon
    n_samples = len(X1_train)
    t = 0.1 * (n_samples ** (-1/3))  #t ∝ n^(-1/3)
    epsilon = t ** 2  #ε ∝ t²
    
    print(f"\n  Optimal parameters (n={n_samples}):")
    print(f"t = {t:.6f}")
    print(f"ε = {epsilon:.6f}")
    
    #Train Conditional Brenier Bridge
    cbb = ConditionalBrenierBridge(
        feature_dim=X1_train.shape[1],
        device='cpu'
    )
    
    losses_cbb = cbb.train(
        X1_train=X1_train,
        X2_train=X2_train,
        t=t,
        epsilon=epsilon,
        n_iterations=2000,
        batch_size=256
    )
    
    return cbb, losses_cbb, X1_train, X2_train


def validate_predictions(cbb, X1_test, X2_test, X1_train, data):
    
    print("Validation: Conditional Predictions")

    #Select test queries at different volatility levels from test set
    X1_raw = data['full_X1_raw']
    lookback = int(data['full_lookback'])
    vol_idx = lookback  #Realized vol feature
    
    
    #compute vol directly from test set
    test_vol_values = X1_test[:, vol_idx] if X1_test.shape[1] > vol_idx else X1_test[:, 0]
    
    #Low, medium, high vol scenarios 
    low_idx = np.argmin(test_vol_values)
    med_idx = np.argsort(test_vol_values)[len(test_vol_values)//2]
    high_idx = np.argmax(test_vol_values)
    
    scenarios = [
        ('Low Vol', low_idx, test_vol_values[low_idx]),
        ('Med Vol', med_idx, test_vol_values[med_idx]),
        ('High Vol', high_idx, test_vol_values[high_idx])
    ]
    
    
    for name, idx, vol in scenarios:
        x1_query = X1_test[idx]
        x2_true = X2_test[idx]
        
        #Predict mean
        x2_pred_mean = cbb.predict_mean(x1_query)
        
        #Predict quantiles
        x2_pred_q05 = cbb.predict_quantile(x1_query, 0.05)
        x2_pred_q95 = cbb.predict_quantile(x1_query, 0.95)
        
        print(f"\n  {name} (vol={vol:.4f}):")
        print(f"True: return={x2_true[0]:.4f}, vol={x2_true[1]:.4f}")
        print(f"Pred: return={x2_pred_mean[0]:.4f}, vol={x2_pred_mean[1]:.4f}")
        print(f"Error: return={abs(x2_true[0]-x2_pred_mean[0]):.4f}, "
              f"vol={abs(x2_true[1]-x2_pred_mean[1]):.4f}")
        print(f"90% CI: [{x2_pred_q05[0]:.4f}, {x2_pred_q95[0]:.4f}]")
    

def visualize_results(losses_msb, losses_cbb, cbb, X1_test, X2_test, data):
  
    print("Generating Visualizations")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    #MSB Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if len(losses_msb) > 0:
        ax1.plot(losses_msb, linewidth=2, color='#2E86AB')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('MSB Training Loss', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, 'MSB Training Failed', 
                ha='center', va='center', fontsize=12)
        ax1.set_title('MSB Training Loss', fontsize=12, fontweight='bold')
    
    #Conditional Brenier Training Loss
    ax2 = fig.add_subplot(gs[0, 1])
    if len(losses_cbb) > 0:
        ax2.plot(losses_cbb, linewidth=2, color='#A23B72')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Conditional Brenier Training Loss', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'Brenier Training Failed', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Conditional Brenier Training Loss', fontsize=12, fontweight='bold')
    
    #Conditional Distribution Samples
    ax3 = fig.add_subplot(gs[0, 2])
    if len(X1_test) > 0:
        #Sample from three different conditioning points
        idx_low = np.argmin(X1_test[:, 0])
        idx_med = len(X1_test) // 2
        idx_high = np.argmax(X1_test[:, 0])
        
        for idx, color, label in [(idx_low, '#06A77D', 'Low Vol'),
                                   (idx_med, '#F77F00', 'Med Vol'),
                                   (idx_high, '#D62828', 'High Vol')]:
            samples = cbb.predict_conditional(X1_test[idx], n_samples=500)
            ax3.scatter(samples[:, 0], samples[:, 1], 
                       alpha=0.3, s=10, color=color, label=label)
        
        ax3.set_xlabel('Future Return', fontsize=11)
        ax3.set_ylabel('Future Volatility', fontsize=11)
        ax3.set_title('Conditional Samples μ(·|x₁)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    #Predicted vs True Returns
    ax4 = fig.add_subplot(gs[1, 0])
    if len(X1_test) > 0:
        returns_pred = np.array([cbb.predict_mean(x1)[0] for x1 in X1_test[:100]])
        returns_true = X2_test[:100, 0]
        
        ax4.scatter(returns_true, returns_pred, alpha=0.5, s=30, color='#2E86AB')
        
        #Perfect prediction line
        lim = max(abs(returns_true).max(), abs(returns_pred).max())
        ax4.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
        
        #R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(returns_true, returns_pred)
        ax4.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('True Return', fontsize=11)
        ax4.set_ylabel('Predicted Return', fontsize=11)
        ax4.set_title('Return Prediction Accuracy', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    #Predicted vs True Volatility
    ax5 = fig.add_subplot(gs[1, 1])
    if len(X1_test) > 0:
        vols_pred = np.array([cbb.predict_mean(x1)[1] for x1 in X1_test[:100]])
        vols_true = X2_test[:100, 1]
        
        ax5.scatter(vols_true, vols_pred, alpha=0.5, s=30, color='#A23B72')
        
        #Perfect prediction line
        lim = max(vols_true.max(), vols_pred.max())
        ax5.plot([0, lim], [0, lim], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
        
        #R² score
        r2 = r2_score(vols_true, vols_pred)
        ax5.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=ax5.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax5.set_xlabel('True Volatility', fontsize=11)
        ax5.set_ylabel('Predicted Volatility', fontsize=11)
        ax5.set_title('Volatility Prediction Accuracy', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    #Prediction Error Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    if len(X1_test) > 0:
        returns_pred = np.array([cbb.predict_mean(x1)[0] for x1 in X1_test[:100]])
        errors = returns_pred - X2_test[:100, 0]
        
        ax6.hist(errors, bins=30, alpha=0.7, color='#06A77D', edgecolor='black')
        ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax6.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean = {errors.mean():.4f}')
        
        ax6.set_xlabel('Prediction Error', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Return Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
    
    #Uncertainty Quantification (Return)
    ax7 = fig.add_subplot(gs[2, 0])
    if len(X1_test) > 0:
        n_samples = min(50, len(X1_test))
        indices = np.random.choice(len(X1_test), n_samples, replace=False)
        
        returns_true = X2_test[indices, 0]
        returns_pred = []
        returns_q05 = []
        returns_q95 = []
        
        for idx in indices:
            pred_mean = cbb.predict_mean(X1_test[idx])
            pred_q05 = cbb.predict_quantile(X1_test[idx], 0.05)
            pred_q95 = cbb.predict_quantile(X1_test[idx], 0.95)
            
            returns_pred.append(pred_mean[0])
            returns_q05.append(pred_q05[0])
            returns_q95.append(pred_q95[0])
        
        returns_pred = np.array(returns_pred)
        returns_q05 = np.array(returns_q05)
        returns_q95 = np.array(returns_q95)
        
        x = np.arange(n_samples)
        ax7.fill_between(x, returns_q05, returns_q95, alpha=0.3, color='#2E86AB', 
                        label='90% CI')
        ax7.plot(x, returns_pred, 'o-', color='#2E86AB', label='Prediction', markersize=4)
        ax7.plot(x, returns_true, 's', color='#D62828', label='True', markersize=4, alpha=0.7)
        
        ax7.set_xlabel('Sample Index', fontsize=11)
        ax7.set_ylabel('Return', fontsize=11)
        ax7.set_title('Return Prediction with Uncertainty', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
    
    #Uncertainty Quantification (Volatility)
    ax8 = fig.add_subplot(gs[2, 1])
    if len(X1_test) > 0:
        vols_true = X2_test[indices, 1]
        vols_pred = []
        vols_q05 = []
        vols_q95 = []
        
        for idx in indices:
            pred_mean = cbb.predict_mean(X1_test[idx])
            pred_q05 = cbb.predict_quantile(X1_test[idx], 0.05)
            pred_q95 = cbb.predict_quantile(X1_test[idx], 0.95)
            
            vols_pred.append(pred_mean[1])
            vols_q05.append(pred_q05[1])
            vols_q95.append(pred_q95[1])
        
        vols_pred = np.array(vols_pred)
        vols_q05 = np.array(vols_q05)
        vols_q95 = np.array(vols_q95)
        
        ax8.fill_between(x, vols_q05, vols_q95, alpha=0.3, color='#A23B72', 
                        label='90% CI')
        ax8.plot(x, vols_pred, 'o-', color='#A23B72', label='Prediction', markersize=4)
        ax8.plot(x, vols_true, 's', color='#D62828', label='True', markersize=4, alpha=0.7)
        
        ax8.set_xlabel('Sample Index', fontsize=11)
        ax8.set_ylabel('Volatility', fontsize=11)
        ax8.set_title('Volatility Prediction with Uncertainty', fontsize=12, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)
    
    #Transport Map Visualization
    ax9 = fig.add_subplot(gs[2, 2])
    if len(X1_test) > 0:
        #Select one conditioning point
        idx_query = len(X1_test) // 2
        
        #Sample from reference distribution
        n_vis = 200
        noise_samples = np.random.randn(n_vis, 2)
        
        #Push through transport map
        x1_query = X1_test[idx_query]
        x1_norm = (x1_query - cbb.X1_mean) / cbb.X1_std
        x1_torch = torch.tensor(x1_norm, dtype=torch.float32).to(cbb.device)
        x1_batch = x1_torch.unsqueeze(0).repeat(n_vis, 1)
        noise_torch = torch.tensor(noise_samples, dtype=torch.float32).to(cbb.device)
        
        with torch.no_grad():
            transported = cbb.transport_net(x1_batch, noise_torch).cpu().numpy()
        
        #Denormalize
        transported = transported * cbb.X2_std + cbb.X2_mean
        
        ax9.scatter(noise_samples[:, 0], noise_samples[:, 1], 
                   alpha=0.4, s=20, color='#95A3B3', label='Reference ρ₂')
        ax9.scatter(transported[:, 0], transported[:, 1], 
                   alpha=0.4, s=20, color='#06A77D', label='Transported T(ρ₂)')
        
        ax9.set_xlabel('Dimension 1', fontsize=11)
        ax9.set_ylabel('Dimension 2', fontsize=11)
        ax9.set_title('Transport Map T: ρ₂ → μ(·|x₁)', fontsize=12, fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Unified Bridge Framework: MSB + Conditional Brenier', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    #Save figure
    output_path = Path('data') / 'unified_bridge_results.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_path}")
    
    plt.show()
    
    plt.tight_layout()
    output_path = Path('data') / 'unified_bridge_results.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    #Load data
    data = load_synthetic_data()
    
    #Option pricing calibration with MSB
    msb, losses_msb, strikes, market_prices, ivs_market, T = \
        run_option_pricing_calibration(data)
    
    #Validate MSB
    if len(losses_msb) > 0:
        print("MSB Validation: Option Prices")
        
        #Compute implied volatility from option price
        def black_scholes_iv(S, K, T, price, r=0.0):
            sigma = 0.3
            for _ in range(100):
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                price_est = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                vega = S*norm.pdf(d1)*np.sqrt(T)
                
                diff = price - price_est
                if abs(diff) < 1e-6:
                    break
                if vega < 1e-10:
                    break
                
                sigma += diff / vega
                sigma = max(0.01, min(2.0, sigma))
            
            return sigma
        
        print(f"\n  Strike | Market Price | Model Price | Error | Market IV | Model IV")
        print(f"  " + "-"*75)
        
        for K, C_mkt, IV_mkt in zip(strikes[:5], market_prices[:5], ivs_market[:5]):
            C_model, _ = msb.price_option(K, T)
            error = abs(C_model - C_mkt)
            
            try:
                IV_model = black_scholes_iv(msb.heston.S0, K, T, C_model)
            except:
                IV_model = 0.2
            
            print(f"  {K:6.0f} | {C_mkt:11.4f} | {C_model:11.4f} | {error:5.4f} | "
                  f"{IV_mkt*100:8.2f}% | {IV_model*100:8.2f}%")
        
        if len(strikes) > 5:
            print(f"({len(strikes)-5} more strikes)")
    else:
        print(f"\n MSB training failed - no validation performed")
    
    #Price prediction with Conditional Brenier
    cbb, losses_cbb, X1_train, X2_train = \
        run_price_prediction_with_brenier(data)
    
    #Validate Conditional Brenier
    n_test = min(500, len(X1_train))
    test_idx = np.random.choice(len(X1_train), n_test, replace=False)
    X1_test = X1_train[test_idx]
    X2_test = X2_train[test_idx]
    
    validate_predictions(cbb, X1_test, X2_test, X1_train, data)
    
    #Visualize combined results
    visualize_results(losses_msb, losses_cbb, cbb, X1_test, X2_test, data)
    
    #Final summary
    
    print("\n MSB: Option pricing calibrated")
    if len(losses_msb) > 0:
        print(f"Final loss: {losses_msb[-1]:.6f}")
        print(f"Iterations: {len(losses_msb)}")
    
    print("\n Conditional Brenier: Price prediction trained")
    if len(losses_cbb) > 0:
        print(f"Final loss: {losses_cbb[-1]:.6f}")
        print(f"Iterations: {len(losses_cbb)}")
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

N_CPUS = cpu_count()

# Heston pricer

class HestonPricer:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r=0.0):
        self.S0    = S0
        self.v0    = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho   = rho
        self.r     = r

    def simulate_paths(self, T, n_steps, n_paths=10000, seed=None,
                       return_vol=False):
        if seed is not None:
            np.random.seed(seed)

        dt         = T / n_steps
        sqrt_1mrho = np.sqrt(1.0 - self.rho ** 2)

        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # Pre-allocate antithetic noise (halves MC variance for free).
        # Ensure even count; if n_paths is odd, draw one extra path and trim.
        half    = (n_paths + 1) // 2
        Z1_all  = np.random.randn(n_steps, half)
        Z2_all  = np.random.randn(n_steps, half)
        Z1_full = np.concatenate([Z1_all, -Z1_all], axis=1)[:, :n_paths]
        Z2_full = np.concatenate([Z2_all, -Z2_all], axis=1)[:, :n_paths]

        for i in range(n_steps):
            W1 = Z1_full[i]
            W2 = self.rho * Z1_full[i] + sqrt_1mrho * Z2_full[i]

            vc = np.maximum(v[:, i], 1e-8)

            dv = (self.kappa * (self.theta - vc) * dt
                  + self.sigma * np.sqrt(vc * dt) * W2)
            v[:, i + 1] = np.maximum(vc + dv, 1e-8)

            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * vc) * dt + np.sqrt(vc * dt) * W1
            )

        if return_vol:
            return S, v
        return S

    def price_option(self, K, T, n_paths=50000, n_steps=None):
        if n_steps is None:
            n_steps = max(50, int(T * 252))
        S_paths = self.simulate_paths(T, n_steps, n_paths)
        payoff  = np.maximum(S_paths[:, -1] - K, 0)
        return np.exp(-self.r * T) * np.mean(payoff)

    def price_options_batch(self, strikes, T, n_paths=50000, n_steps=None):
        """Price all strikes in one simulation — avoids repeating the MC."""
        if n_steps is None:
            n_steps = max(50, int(T * 252))
        S_paths = self.simulate_paths(T, n_steps, n_paths)
        S_T     = S_paths[:, -1]
        payoffs = np.maximum(S_T[:, None] - np.asarray(strikes)[None, :], 0)
        return np.exp(-self.r * T) * payoffs.mean(axis=0)

# Black-Scholes helperr

def black_scholes_iv(S, K, T, price, r=0.0):
    """Implied vol via Newton-Raphson; returns 0.01 if price ≤ intrinsic."""
    intrinsic = max(S * np.exp(-0.0 * T) - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-10:
        return 0.01

    sigma = 0.3
    for _ in range(100):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)

        diff = price - bs_price
        if abs(diff) < 1e-8:
            break
        if vega < 1e-12:
            break

        sigma += diff / vega
        sigma = np.clip(sigma, 0.01, 5.0)

    return float(sigma)

# Volatility surface generation — one MC per maturity (batch strikesr

def _price_one_maturity(args):
    """Top-level worker (picklable) for multiprocessing."""
    params, strikes, T, n_paths = args
    pricer = HestonPricer(**params)
    return pricer.price_options_batch(strikes, T, n_paths=n_paths)


def generate_heston_surface(heston_params, strikes_norm, maturities,
                            n_paths=50000):
    """
    One simulation per maturity (not per strike).  All strikes share the
    same terminal S_T vector, so the cost is O(len(maturities)) instead of
    O(len(maturities) × len(strikes)).  Maturities are priced in parallel.
    """
    S0 = heston_params['S0']
    strikes = strikes_norm * S0

    args = [(heston_params, strikes, T, n_paths) for T in maturities]

    print(f"\n  Pricing surface: {len(maturities)} maturities × "
          f"{len(strikes)} strikes  ({N_CPUS} workers)")

    if N_CPUS > 1:
        with Pool(N_CPUS) as pool:
            all_prices = pool.map(_price_one_maturity, args)
    else:
        all_prices = [_price_one_maturity(a) for a in args]

    vol_surface = np.zeros((len(maturities), len(strikes_norm)))

    for i, (T, prices) in enumerate(zip(maturities, all_prices)):
        for j, (K, price) in enumerate(zip(strikes, prices)):
            try:
                iv = black_scholes_iv(S0, K, T, price)
                vol_surface[i, j] = iv if iv > 0.01 else (
                    vol_surface[i, j - 1] if j > 0 else 0.2
                )
            except Exception:
                vol_surface[i, j] = vol_surface[i, j - 1] if j > 0 else 0.2

        print(f"  T={T:.2f}Y  ATM vol = {vol_surface[i, len(strikes) // 2] * 100:.2f}%")

    return vol_surface

# IV cache for 'full' features 

def _build_iv_cache(pricer, prices_path, feature_ends, n_paths=20000):
    maturities_iv = [1 / 12, 3 / 12]
    iv_cache = {}  # idx -> (iv_1m, iv_3m)

    for T in maturities_iv:
        # One MC from S0=100
        n_steps = max(20, int(T * 252))
        S_paths = pricer.simulate_paths(T, n_steps, n_paths=n_paths)
        S_T_norm = S_paths[:, -1] / pricer.S0   # terminal ratio

        for idx in feature_ends:
            S_curr = prices_path[idx]
            S_T = S_T_norm * S_curr       # scale to current price
            payoff = np.maximum(S_T - S_curr, 0)   # ATM call
            call_price = np.exp(-pricer.r * T) * payoff.mean()
            iv = black_scholes_iv(S_curr, S_curr, T, call_price)
            if idx not in iv_cache:
                iv_cache[idx] = {}
            iv_cache[idx][T] = iv

    return iv_cache

# Vectorised rolling statistics helperr

def _rolling_vol(returns, window):
    """Annualised rolling volatility, shape (n_samples,)"""
    n = len(returns)
    out = np.empty(n - window + 1)
    for i in range(len(out)):
        out[i] = np.std(returns[i: i + window], ddof=1) * np.sqrt(252)
    return out


def _autocorr_lag1(returns):
    """Lag-1 autocorrelation; returns 0 if std is zero"""
    if len(returns) < 2:
        return 0.0
    x = returns[:-1] - returns[:-1].mean()
    y = returns[1:]  - returns[1:].mean()
    denom = np.std(returns[:-1], ddof=1) * np.std(returns[1:], ddof=1)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / (len(x) * denom))

# Main data generator

class ConditionalPriceGenerator:
    """
    Uses Heston dynamics to create the joint distribution μ(x1, x2)
    """

    def __init__(self, heston_params, seed=42):
        self.heston_params = heston_params
        self.pricer = HestonPricer(**heston_params)
        self.rng = np.random.RandomState(seed)

    def generate_price_prediction_data(self, n_samples=3000, lookback=20,
                                       horizon=5, features='full'):
        burn_in = 50
        total_steps = burn_in + lookback + n_samples * horizon
        dt = 1 / 252

        S_paths, v_paths = self.pricer.simulate_paths(
            T = total_steps * dt,
            n_steps= total_steps,
            n_paths= 1,
            seed = self.rng.randint(10000),
            return_vol=True,
        )

        prices = S_paths[0]
        vols  = v_paths[0]  # instantaneous variance
        log_ret = np.diff(np.log(prices))  # length = total_steps

        # Starting indices for each sample (non-overlapping, stride = horizon)
        starts = burn_in + np.arange(n_samples) * horizon

        # Pre-compute IV cache 
        feature_ends = starts + lookback      # index where feature window ends
        iv_cache = {}
        if features == 'full':
            print("  Building IV cache (one MC per maturity)…")
            iv_cache = _build_iv_cache(
                self.pricer, prices, feature_ends, n_paths=30000
            )

        # Extract samples 
        X1_list   = []
        X2_list   = []
        metadata  = []

        print(f"\n  Extracting {n_samples} non-overlapping samples  "
              f"(stride={horizon})")

        for i in range(n_samples):
            s_idx   = int(starts[i])
            f_end   = s_idx + lookback
            t_end   = f_end + horizon

            past_ret = log_ret[s_idx: f_end]          # length = lookback
            fut_ret  = log_ret[f_end: t_end]           # length = horizon

            # Features 
            realized_vol = np.std(past_ret, ddof=1) * np.sqrt(252)
            current_vol = float(np.sqrt(np.maximum(vols[f_end], 1e-8)))
            momentum_5d  = float(np.mean(past_ret[-5:]))
            momentum_20d = float(np.mean(past_ret))
            trend = float(past_ret[-1] - past_ret[0])
            autocorr = _autocorr_lag1(past_ret)

            # vol-of-vol: std of 5-day rolling vols over the lookback window
            if lookback >= 10:
                rolling = _rolling_vol(past_ret, 5)   # (lookback-4,) vols
                vol_of_vol = float(np.std(rolling, ddof=1)) if len(rolling) > 1 else 0.0
            else:
                vol_of_vol = 0.0

            if features == 'full':
                iv_1m  = iv_cache.get(f_end, {}).get(1 / 12, current_vol)
                iv_3m  = iv_cache.get(f_end, {}).get(3 / 12, current_vol)
                iv_slope = iv_3m - iv_1m
                features_array = np.concatenate([
                    past_ret,
                    [realized_vol, current_vol, momentum_5d, momentum_20d,
                     trend, autocorr, vol_of_vol, iv_1m, iv_3m, iv_slope]
                ])
            else:
                features_array = np.concatenate([
                    past_ret,
                    [realized_vol, current_vol, momentum_5d, momentum_20d,
                     trend, autocorr, vol_of_vol]
                ])

            # Targets 
            cumulative_return = float(np.sum(fut_ret))

            # future_vol from mean instantaneous variance 
            future_var_mean = float(np.mean(vols[f_end: t_end]))
            future_vol = float(np.sqrt(np.maximum(future_var_mean, 1e-8)))

            cum_ret     = np.cumsum(fut_ret)
            running_max = np.maximum.accumulate(cum_ret)
            max_drawdown = float(np.max(running_max - cum_ret)) if len(cum_ret) > 0 else 0.0

            X1_list.append(features_array)
            X2_list.append([cumulative_return, future_vol, max_drawdown])
            metadata.append({
                'start_price': float(prices[f_end]),
                'final_price': float(prices[t_end]),
                'current_vol': current_vol,
                'start_idx': s_idx,
            })

        X1 = np.array(X1_list, dtype=np.float64)
        X2 = np.array(X2_list, dtype=np.float64)

        # Normalize X1 (robust: clip extreme std values)
        X1_mean = X1.mean(axis=0)
        X1_std  = np.maximum(X1.std(axis=0, ddof=1), 1e-8)
        X1_norm = (X1 - X1_mean) / X1_std

        print(f"Done.Feature dim={X1.shape[1]}, target dim={X2.shape[1]}")
        print(f"Return: mean={X2[:, 0].mean():.4f}  std={X2[:, 0].std():.4f}")
        print(f"Fut. vol: mean={X2[:, 1].mean():.4f}  std={X2[:, 1].std():.4f}")
        print(f"Drawdown: mean={X2[:, 2].mean():.4f}  max={X2[:, 2].max():.4f}")

        return {
            'X1': X1_norm,
            'X2': X2,
            'X1_raw': X1,
            'X1_mean': X1_mean,
            'X1_std': X1_std,
            'metadata': metadata,
            'feature_dim': int(X1.shape[1]),
            'target_dim': int(X2.shape[1]),
            'lookback': lookback,
            'horizon': horizon,
            'feature_type': features,
        }

    def generate_multimodal_regime_data(self, n_samples=2000):
        regime_probs = [0.5, 0.3, 0.2]
        regime_params = []

        for regime in range(3):
            p = self.heston_params.copy()
            if regime == 0:    # Bull
                p['v0']      = p['theta'] * 0.7
                p['theta']   = p['theta'] * 0.8
                drift_adjust = 0.10
            elif regime == 1:  # Neutral
                drift_adjust = 0.03
            else:              # Bear
                p['v0']      = p['theta'] * 1.5
                p['theta']   = p['theta'] * 1.2
                drift_adjust = -0.05
            regime_params.append((p, drift_adjust))

        # Build one pricer per regime
        pricers = [HestonPricer(**rp[0]) for rp in regime_params]

        X1_list, X2_list, regime_list = [], [], []

        print(f"\n  Generating {n_samples} regime-switching samples…")

        for _ in range(n_samples):
            regime = int(self.rng.choice(3, p=regime_probs))
            pricer, drift_adjust = pricers[regime], regime_params[regime][1]

            T_sim = 1 / 12        # ~21 trading days
            n_steps = 21
            S_path, v_path = pricer.simulate_paths(
                T=T_sim, n_steps=n_steps, n_paths=1,
                seed=self.rng.randint(10000),
                return_vol=True,
            )

            ret = np.diff(np.log(S_path[0]))  # length = 21
            mid = n_steps // 2                  # split point = 10

            past_ret = ret[:mid]
            future_ret = ret[mid:]

            past_vol = np.std(past_ret, ddof=1) * np.sqrt(252) if len(past_ret) > 1 else 0.0
            current_vol = float(np.sqrt(np.maximum(v_path[0, mid], 1e-8)))
            momentum = float(np.mean(past_ret))

            future_return = float(np.sum(future_ret)) + drift_adjust / 12

            # future_vol from mean instantaneous variance (not 11 returns)
            future_var = float(np.mean(v_path[0, mid:]))
            future_vol = float(np.sqrt(np.maximum(future_var, 1e-8)))

            X1_list.append([past_vol, current_vol, momentum])
            X2_list.append([future_return, future_vol])
            regime_list.append(regime)

        regime_arr = np.array(regime_list)
        print(f"  Regime distribution: {np.bincount(regime_arr)}")

        return {
            'X1': np.array(X1_list,  dtype=np.float64),
            'X2': np.array(X2_list,  dtype=np.float64),
            'regimes': regime_arr,
            'feature_dim': 3,
            'target_dim': 2,
        }

# OT estimator 

class OptimalTransportEstimator:
    def __init__(self, t=0.01):
        self.t = t
        self.X1_train = None
        self.X2_train = None

    def fit(self, X1, X2):
        self.X1_train = X1
        self.X2_train = X2

    def predict_conditional(self, x1_query, n_samples=100):
        distances = np.sum((self.X1_train - x1_query) ** 2, axis=1)
        weights = np.exp(-distances / (2 * self.t ** 2))
        weights  /= weights.sum()
        indices = np.random.choice(
            len(self.X2_train), size=n_samples, p=weights, replace=True
        )
        return self.X2_train[indices], weights

    def predict_mean(self, x1_query):
        samples, _ = self.predict_conditional(x1_query, n_samples=1000)
        return samples.mean(axis=0)

    def predict_quantile(self, x1_query, quantile=0.05):
        samples, _ = self.predict_conditional(x1_query, n_samples=1000)
        return np.quantile(samples, quantile, axis=0)

# Visualisations 

def visualize_heston_surface(vol_surface, strikes_norm, maturities, save_path):
    fig = plt.figure(figsize=(16, 5))

    ax1 = plt.subplot(131)
    for i, T in enumerate(maturities):
        ax1.plot(strikes_norm, vol_surface[i, :] * 100,
                 'o-', label=f'T={T}Y', linewidth=2, markersize=6)
    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax1.set_xlabel('Strike / Spot', fontsize=12)
    ax1.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax1.set_title('Volatility Smiles\n(Heston Model)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(132)
    K_mesh, T_mesh = np.meshgrid(strikes_norm, maturities)
    pcm = ax2.pcolormesh(K_mesh, T_mesh, vol_surface * 100, cmap='viridis', shading='auto')
    plt.colorbar(pcm, ax=ax2, label='Implied Vol (%)')
    ax2.set_xlabel('Strike / Spot', fontsize=12)
    ax2.set_ylabel('Maturity (years)', fontsize=12)
    ax2.set_title('Volatility Surface', fontsize=14, fontweight='bold')

    ax3 = plt.subplot(133)
    atm_idx = len(strikes_norm) // 2
    ax3.plot(maturities, vol_surface[:, atm_idx] * 100,
             'o-', linewidth=2, markersize=8, color='darkblue')
    ax3.set_xlabel('Maturity (years)', fontsize=12)
    ax3.set_ylabel('ATM Implied Volatility (%)', fontsize=12)
    ax3.set_title('ATM Term Structure', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_prediction_data(data, save_path):
    X1 = data['X1_raw']
    X2 = data['X2']
    lookback = data['lookback']

    fig = plt.figure(figsize=(16, 10))

    realized_vol_idx = lookback
    momentum_idx = lookback + 2
    current_vol_idx  = lookback + 1

    ax1 = plt.subplot(2, 3, 1)
    sc  = ax1.scatter(X1[:, realized_vol_idx], X2[:, 0],
                      c=X2[:, 1], cmap='coolwarm', alpha=0.5, s=15)
    ax1.set_xlabel('Past Realized Volatility', fontsize=10)
    ax1.set_ylabel('Future Return', fontsize=10)
    ax1.set_title('Return vs Historical Vol', fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax1, label='Future Vol')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    sc2 = ax2.scatter(X1[:, momentum_idx], X2[:, 0],
                      c=X2[:, 2], cmap='plasma', alpha=0.5, s=15)
    ax2.set_xlabel('Momentum (5d)', fontsize=10)
    ax2.set_ylabel('Future Return', fontsize=10)
    ax2.set_title('Return vs Momentum', fontsize=11, fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='Max DD')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(X2[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Future Return', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Future Return Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(X1[:, current_vol_idx], X2[:, 1], alpha=0.4, s=15)
    lim = max(X1[:, current_vol_idx].max(), X2[:, 1].max())
    ax4.plot([0, lim], [0, lim], 'r--', linewidth=2, alpha=0.7, label='45° line')
    ax4.set_xlabel('Current Volatility', fontsize=10)
    ax4.set_ylabel('Future Realized Volatility', fontsize=10)
    ax4.set_title('Volatility Persistence', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    sc5 = ax5.scatter(X2[:, 0], X2[:, 2], c=X2[:, 1], cmap='YlOrRd', alpha=0.5, s=15)
    ax5.set_xlabel('Future Return', fontsize=10)
    ax5.set_ylabel('Max Drawdown', fontsize=10)
    ax5.set_title('Return vs Risk', fontsize=11, fontweight='bold')
    plt.colorbar(sc5, ax=ax5, label='Future Vol')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    h = ax6.hist2d(X1[:, realized_vol_idx], X2[:, 0], bins=30, cmap='Blues')
    ax6.set_xlabel('Past Realized Vol', fontsize=10)
    ax6.set_ylabel('Future Return', fontsize=10)
    ax6.set_title('Joint Distribution μ(x₁, x₂)', fontsize=11, fontweight='bold')
    plt.colorbar(h[3], ax=ax6, label='Density')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_conditional_predictions(data, estimator, save_path):
    X1 = data['X1']
    X2 = data['X2']
    X1_raw  = data['X1_raw']
    lookback = data['lookback']

    estimator.fit(X1, X2)

    vol_idx = lookback
    vol_values = X1_raw[:, vol_idx]
    low_idx = int(np.argmin(vol_values))
    high_idx = int(np.argmax(vol_values))
    med_idx = int(np.argsort(vol_values)[len(vol_values) // 2])

    query_indices = [low_idx, med_idx, high_idx]
    query_labels  = ['Low Vol', 'Medium Vol', 'High Vol']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, idx, label in zip(axes, query_indices, query_labels):
        query = X1[idx]
        samples, _ = estimator.predict_conditional(query, n_samples=1000)
        ax.hist2d(samples[:, 0], samples[:, 1], bins=30, cmap='viridis')
        ax.set_xlabel('Future Return', fontsize=11)
        ax.set_ylabel('Future Volatility', fontsize=11)
        ax.set_title(f'μ(x₂|x₁) — {label}\nVol={vol_values[idx]:.3f}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        mean_pred = samples.mean(axis=0)
        ax.plot(mean_pred[0], mean_pred[1], 'r*', markersize=15,
                markeredgecolor='white', markeredgewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# Main

def main():
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir.parent / 'data'
    data_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n  Data directory: {data_dir}")
    print(f"  Workers available: {N_CPUS}")

    heston_params = {
        'S0': 100.0,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04,
        'r': 0.0,
    }

    print("\n  Heston parameters:")
    for k, v in heston_params.items():
        print(f"    {k}: {v}")

    # Volatility surface 
    print("\n  Generating volatility surface…")
    strikes_norm = np.linspace(0.8, 1.2, 21)
    maturities   = np.array([0.25, 0.5, 0.75, 1.0])

    vol_surface = generate_heston_surface(
        heston_params, strikes_norm, maturities, n_paths=50000
    )

    print(f"\n  Surface statistics:")
    print(f"Min vol : {vol_surface.min() * 100:.2f}%")
    print(f"Max vol : {vol_surface.max() * 100:.2f}%")
    print(f"Mean vol: {vol_surface.mean() * 100:.2f}%")
    print(f"ATM 1Y  : {vol_surface[-1, 10] * 100:.2f}%")

    # Price prediction datasets 
    print("\n  Generating price prediction datasets…")
    generator = ConditionalPriceGenerator(heston_params, seed=42)

    print("\n [Dataset 1: Full features with IV surface]")
    data_full = generator.generate_price_prediction_data(
        n_samples=3000, lookback=20, horizon=5, features='full'
    )

    print("\n  [Dataset 2: Basic features]")
    data_basic = generator.generate_price_prediction_data(
        n_samples=3000, lookback=20, horizon=5, features='basic'
    )

    print("\n  [Dataset 3: Regime switching]")
    data_regime = generator.generate_multimodal_regime_data(n_samples=2000)

    # Save 
    output_file = data_dir / 'unified_heston_prediction_data.npz'

    np.savez(
        str(output_file),
        vol_surface = vol_surface,
        strikes_norm = strikes_norm,
        maturities = maturities,
        full_X1 = data_full['X1'],
        full_X2 = data_full['X2'],
        full_X1_raw = data_full['X1_raw'],
        full_X1_mean = data_full['X1_mean'],
        full_X1_std = data_full['X1_std'],
        full_lookback = data_full['lookback'],
        full_horizon = data_full['horizon'],
        basic_X1 = data_basic['X1'],
        basic_X2 = data_basic['X2'],
        basic_X1_raw = data_basic['X1_raw'],
        basic_X1_mean = data_basic['X1_mean'],
        basic_X1_std = data_basic['X1_std'],
        regime_X1 = data_regime['X1'],
        regime_X2 = data_regime['X2'],
        regime_labels = data_regime['regimes'],
        heston_params = heston_params,
        method = 'unified_heston_conditional_brenier',
        description = 'Unified Heston vol surface and conditional price prediction',
    )

    if output_file.exists():
        print(f"\n  Saved: {output_file}  ({output_file.stat().st_size:,} bytes)")

    # Visualisations 
    visualize_heston_surface(
        vol_surface, strikes_norm, maturities,
        data_dir / 'heston_surface.png'
    )
    visualize_prediction_data(data_full, data_dir / 'price_prediction_data.png')

    estimator = OptimalTransportEstimator(t=0.5)
    visualize_conditional_predictions(
        data_full, estimator, data_dir / 'conditional_distributions.png'
    )

    print("\n  Generation complete.")
    print(f"  Files in {data_dir}:")
    for f in sorted(data_dir.glob('*')):
        print(f"    {f.name}")


if __name__ == '__main__':
    main()
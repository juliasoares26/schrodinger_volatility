from __future__ import annotations

import argparse
import sys
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Path setup — works whether the file lives in experiments/ or the root

_THIS_FILE  = Path(__file__).resolve()
_SCRIPT_DIR = _THIS_FILE.parent                    # experiments/  (or root)
_PROJECT_ROOT = (
    _SCRIPT_DIR.parent
    if _SCRIPT_DIR.name == "experiments"
    else _SCRIPT_DIR
)

# Canonical output location  →  data/synthetic/
_DATA_DIR = _PROJECT_ROOT / "data" / "synthetic"

# Make models importable whether the script lives in experiments/ or root
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "models"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_CPUS = cpu_count()

# Heston Monte Carlo pricer (self-contained — no external model dependency

class _HestonPricer:
    """Lightweight Heston pricer used only during data generation."""

    def __init__(self, S0, v0, kappa, theta, sigma, rho, r=0.0):
        self.S0  = S0
        self.v0  = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho  = rho
        self.r = r

    def simulate_paths(self, T, n_steps, n_paths=10_000, seed=None,
                       return_vol=False):
        if seed is not None:
            np.random.seed(seed)

        dt  = T / n_steps
        sqrt_1mrho2 = np.sqrt(max(1.0 - self.rho ** 2, 0.0))

        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        half  = (n_paths + 1) // 2
        Z1_all = np.random.randn(n_steps, half)
        Z2_all = np.random.randn(n_steps, half)
        Z1 = np.concatenate([Z1_all,  -Z1_all],  axis=1)[:, :n_paths]
        Z2 = np.concatenate([Z2_all,  -Z2_all],  axis=1)[:, :n_paths]

        for i in range(n_steps):
            W1 = Z1[i]
            W2 = self.rho * Z1[i] + sqrt_1mrho2 * Z2[i]
            vc = np.maximum(v[:, i], 1e-8)
            v[:, i + 1] = np.maximum(
                vc + self.kappa * (self.theta - vc) * dt
                   + self.sigma * np.sqrt(vc * dt) * W2,
                1e-8,
            )
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * vc) * dt + np.sqrt(vc * dt) * W1
            )

        if return_vol:
            return S, v
        return S

    def price_options_batch(self, strikes, T, n_paths=50_000, n_steps=None):
        if n_steps is None:
            n_steps = max(50, int(T * 252))
        S_paths = self.simulate_paths(T, n_steps, n_paths)
        S_T     = S_paths[:, -1]
        payoffs = np.maximum(S_T[:, None] - np.asarray(strikes)[None, :], 0)
        return np.exp(-self.r * T) * payoffs.mean(axis=0)

# Black-Scholes helper

def _bs_iv(S, K, T, price, r=0.0):
    """Implied vol via Newton-Raphson; returns 0.01 if price ≤ intrinsic"""
    intrinsic = max(S * np.exp(-0.0 * T) - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-10:
        return 0.01
    sigma = 0.3
    for _ in range(100):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        diff  = price - bs
        if abs(diff) < 1e-8 or vega < 1e-12:
            break
        sigma += diff / vega
        sigma  = np.clip(sigma, 0.01, 5.0)
    return float(sigma)

# Volatility surface one MC per maturity (parallel)

def _price_one_maturity(args):
    """Top-level worker (picklable) for multiprocessing."""
    params, strikes, T, n_paths = args
    pricer = _HestonPricer(**params)
    return pricer.price_options_batch(strikes, T, n_paths=n_paths)


def generate_heston_surface(heston_params, strikes_norm, maturities,
                             n_paths=50_000):
    S0 = heston_params["S0"]
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
                iv = _bs_iv(S0, K, T, price)
                vol_surface[i, j] = iv if iv > 0.01 else (
                    vol_surface[i, j - 1] if j > 0 else 0.2
                )
            except Exception:
                vol_surface[i, j] = vol_surface[i, j - 1] if j > 0 else 0.2
        print(f"  T={T:.2f}Y  ATM vol = "
              f"{vol_surface[i, len(strikes) // 2] * 100:.2f}%")

    return vol_surface

# IV cache for 'full' feature

def _rolling_vol(returns, window):
    out = np.empty(len(returns) - window + 1)
    for i in range(len(out)):
        out[i] = np.std(returns[i: i + window], ddof=1) * np.sqrt(252)
    return out


def _autocorr_lag1(returns):
    if len(returns) < 2:
        return 0.0
    x = returns[:-1] - returns[:-1].mean()
    y = returns[1:]  - returns[1:].mean()
    denom = np.std(returns[:-1], ddof=1) * np.std(returns[1:], ddof=1)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / (len(x) * denom))


def _build_iv_cache(pricer, prices_path, feature_ends, n_paths=20_000):
    maturities_iv = [1 / 12, 3 / 12]
    iv_cache: dict = {}
    for T in maturities_iv:
        n_steps  = max(20, int(T * 252))
        S_paths  = pricer.simulate_paths(T, n_steps, n_paths=n_paths)
        S_T_norm = S_paths[:, -1] / pricer.S0
        for idx in feature_ends:
            S_curr = prices_path[idx]
            S_T    = S_T_norm * S_curr
            payoff = np.maximum(S_T - S_curr, 0)
            call_p = np.exp(-pricer.r * T) * payoff.mean()
            iv     = _bs_iv(S_curr, S_curr, T, call_p)
            if idx not in iv_cache:
                iv_cache[idx] = {}
            iv_cache[idx][T] = iv
    return iv_cache

# Main dataset generato

class ConditionalPriceGenerator:
    """
    Generates X1 (past features) / X2 (future targets) from Heston dynamics

    Uses non-overlapping windows (stride = horizon) to avoid data leakage
    """

    def __init__(self, heston_params, seed=42):
        self.heston_params = heston_params
        self.pricer = _HestonPricer(**heston_params)
        self.rng = np.random.RandomState(seed)

    def generate_price_prediction_data(self, n_samples=3000, lookback=20,
                                       horizon=5, features="full"):
        burn_in = 50
        total_steps = burn_in + lookback + n_samples * horizon
        dt = 1 / 252

        S_paths, v_paths = self.pricer.simulate_paths(
            T = total_steps * dt,
            n_steps = total_steps,
            n_paths = 1,
            seed = self.rng.randint(10_000),
            return_vol=True,
        )
        prices = S_paths[0]
        vols = v_paths[0]
        log_ret = np.diff(np.log(prices))

        starts = burn_in + np.arange(n_samples) * horizon
        feature_ends = starts + lookback
        iv_cache: dict = {}

        if features == "full":
            print("  Building IV cache (one MC per maturity)…")
            iv_cache = _build_iv_cache(self.pricer, prices, feature_ends,
                                       n_paths=30_000)

        X1_list: list = []
        X2_list: list = []

        print(f"\n  Extracting {n_samples} non-overlapping samples "
              f"(stride={horizon})")

        for i in range(n_samples):
            s_idx = int(starts[i])
            f_end = s_idx + lookback
            t_end = f_end + horizon

            past_ret = log_ret[s_idx: f_end]
            fut_ret  = log_ret[f_end: t_end]

            realized_vol = np.std(past_ret, ddof=1) * np.sqrt(252)
            current_vol  = float(np.sqrt(max(vols[f_end], 1e-8)))
            mom_5d  = float(np.mean(past_ret[-5:]))
            mom_20d = float(np.mean(past_ret))
            trend = float(past_ret[-1] - past_ret[0])
            autocorr = _autocorr_lag1(past_ret)

            if lookback >= 10:
                rolling = _rolling_vol(past_ret, 5)
                vol_of_vol = float(np.std(rolling, ddof=1)) if len(rolling) > 1 else 0.0
            else:
                vol_of_vol = 0.0

            if features == "full":
                iv_1m = iv_cache.get(f_end, {}).get(1 / 12, current_vol)
                iv_3m = iv_cache.get(f_end, {}).get(3 / 12, current_vol)
                iv_slope = iv_3m - iv_1m
                feat_arr = np.concatenate([
                    past_ret,
                    [realized_vol, current_vol, mom_5d, mom_20d,
                     trend, autocorr, vol_of_vol, iv_1m, iv_3m, iv_slope],
                ])
            else:
                feat_arr = np.concatenate([
                    past_ret,
                    [realized_vol, current_vol, mom_5d, mom_20d,
                     trend, autocorr, vol_of_vol],
                ])

            cum_ret = float(np.sum(fut_ret))
            future_var = float(np.mean(vols[f_end: t_end]))
            future_vol = float(np.sqrt(max(future_var, 1e-8)))
            cr = np.cumsum(fut_ret)
            running_max = np.maximum.accumulate(cr)
            max_dd = float(np.max(running_max - cr)) if len(cr) > 0 else 0.0

            X1_list.append(feat_arr)
            X2_list.append([cum_ret, future_vol, max_dd])

        X1 = np.array(X1_list, dtype=np.float64)
        X2 = np.array(X2_list, dtype=np.float64)
        X1_mean = X1.mean(axis=0)
        X1_std  = np.maximum(X1.std(axis=0, ddof=1), 1e-8)
        X1_norm = (X1 - X1_mean) / X1_std

        print(f"  Done.  dim={X1.shape[1]} features, "
              f"{X2.shape[1]} targets")
        print(f"  Return:   mean={X2[:, 0].mean():.4f}  "
              f"std={X2[:, 0].std():.4f}")
        print(f"  Fut. vol: mean={X2[:, 1].mean():.4f}  "
              f"std={X2[:, 1].std():.4f}")
        print(f"  Drawdown: mean={X2[:, 2].mean():.4f}  "
              f"max={X2[:, 2].max():.4f}")

        return {
            "X1": X1_norm,
            "X2": X2,
            "X1_raw": X1,
            "X1_mean": X1_mean,
            "X1_std": X1_std,
            "feature_dim": int(X1.shape[1]),
            "target_dim":  int(X2.shape[1]),
            "lookback": lookback,
            "horizon": horizon,
            "feature_type": features,
        }

    def generate_multimodal_regime_data(self, n_samples=2000):
        """Regime-switching dataset (Bull / Neutral / Bear)"""
        regime_probs  = [0.5, 0.3, 0.2]
        regime_params = []
        for regime in range(3):
            p = self.heston_params.copy()
            if regime == 0:
                p["v0"]    = p["theta"] * 0.7
                p["theta"] = p["theta"] * 0.8
                drift_adj  = 0.10
            elif regime == 1:
                drift_adj = 0.03
            else:
                p["v0"]    = p["theta"] * 1.5
                p["theta"] = p["theta"] * 1.2
                drift_adj  = -0.05
            regime_params.append((p, drift_adj))

        pricers = [_HestonPricer(**rp[0]) for rp in regime_params]

        X1_list, X2_list, regime_list = [], [], []
        print(f"\n  Generating {n_samples} regime-switching samples…")

        for _ in range(n_samples):
            regime = int(self.rng.choice(3, p=regime_probs))
            pricer = pricers[regime]
            _, drift_adj = regime_params[regime]

            T_sim = 1 / 12
            n_steps = 21
            S_path, v_path = pricer.simulate_paths(
                T=T_sim, n_steps=n_steps, n_paths=1,
                seed=self.rng.randint(10_000), return_vol=True,
            )
            ret = np.diff(np.log(S_path[0]))
            mid = n_steps // 2

            past_ret = ret[:mid]
            future_ret = ret[mid:]
            past_vol = (np.std(past_ret, ddof=1) * np.sqrt(252)
                            if len(past_ret) > 1 else 0.0)
            current_vol = float(np.sqrt(max(v_path[0, mid], 1e-8)))
            momentum = float(np.mean(past_ret))
            future_ret_c = float(np.sum(future_ret)) + drift_adj / 12
            future_var = float(np.mean(v_path[0, mid:]))
            future_vol = float(np.sqrt(max(future_var, 1e-8)))

            X1_list.append([past_vol, current_vol, momentum])
            X2_list.append([future_ret_c, future_vol])
            regime_list.append(regime)

        regime_arr = np.array(regime_list)
        print(f"  Regime distribution: {np.bincount(regime_arr)}")
        return {
            "X1": np.array(X1_list,  dtype=np.float64),
            "X2": np.array(X2_list,  dtype=np.float64),
            "regimes": regime_arr,
            "feature_dim": 3,
            "target_dim": 2,
        }

# OT estimator

class _OTEstimator:
    def __init__(self, t=0.01):
        self.t = t
        self.X1_train = None
        self.X2_train = None

    def fit(self, X1, X2):
        self.X1_train = X1
        self.X2_train = X2

    def predict_conditional(self, x1_query, n_samples=100):
        distances = np.sum((self.X1_train - x1_query) ** 2, axis=1)
        weights   = np.exp(-distances / (2 * self.t ** 2))
        weights  /= weights.sum()
        indices   = np.random.choice(len(self.X2_train), size=n_samples,
                                     p=weights, replace=True)
        return self.X2_train[indices], weights

# Visualisations 

def _plot_surface(vol_surface, strikes_norm, maturities, save_path):
    fig = plt.figure(figsize=(16, 5))

    ax1 = plt.subplot(131)
    for i, T in enumerate(maturities):
        ax1.plot(strikes_norm, vol_surface[i, :] * 100, "o-",
                 label=f"T={T}Y", linewidth=2, markersize=6)
    ax1.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Strike / Spot", fontsize=12)
    ax1.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax1.set_title("Volatility Smiles (Heston)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(132)
    K_mesh, T_mesh = np.meshgrid(strikes_norm, maturities)
    pcm = ax2.pcolormesh(K_mesh, T_mesh, vol_surface * 100,
                         cmap="viridis", shading="auto")
    plt.colorbar(pcm, ax=ax2, label="IV (%)")
    ax2.set_xlabel("Strike / Spot", fontsize=12)
    ax2.set_ylabel("Maturity (years)", fontsize=12)
    ax2.set_title("Volatility Surface", fontsize=14, fontweight="bold")

    ax3 = plt.subplot(133)
    atm_idx = len(strikes_norm) // 2
    ax3.plot(maturities, vol_surface[:, atm_idx] * 100, "o-",
             linewidth=2, markersize=8, color="darkblue")
    ax3.set_xlabel("Maturity (years)", fontsize=12)
    ax3.set_ylabel("ATM Implied Volatility (%)", fontsize=12)
    ax3.set_title("ATM Term Structure", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_prediction_data(data, save_path):
    X1 = data["X1_raw"]
    X2 = data["X2"]
    lookback = data["lookback"]
    rv_idx = lookback
    mom_idx = lookback + 2
    cv_idx = lookback + 1

    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 3, 1)
    sc = ax1.scatter(X1[:, rv_idx], X2[:, 0],
                      c=X2[:, 1], cmap="coolwarm", alpha=0.5, s=15)
    ax1.set_xlabel("Past Realized Volatility", fontsize=10)
    ax1.set_ylabel("Future Return", fontsize=10)
    ax1.set_title("Return vs Historical Vol", fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax1, label="Future Vol")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    sc2 = ax2.scatter(X1[:, mom_idx], X2[:, 0],
                      c=X2[:, 2], cmap="plasma", alpha=0.5, s=15)
    ax2.set_xlabel("Momentum (5d)", fontsize=10)
    ax2.set_ylabel("Future Return", fontsize=10)
    ax2.set_title("Return vs Momentum", fontsize=11, fontweight="bold")
    plt.colorbar(sc2, ax=ax2, label="Max DD")
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(X2[:, 0], bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax3.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax3.set_xlabel("Future Return", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title("Future Return Distribution", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(X1[:, cv_idx], X2[:, 1], alpha=0.4, s=15)
    lim = max(X1[:, cv_idx].max(), X2[:, 1].max())
    ax4.plot([0, lim], [0, lim], "r--", linewidth=2, alpha=0.7)
    ax4.set_xlabel("Current Volatility", fontsize=10)
    ax4.set_ylabel("Future Realized Volatility", fontsize=10)
    ax4.set_title("Volatility Persistence", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    sc5 = ax5.scatter(X2[:, 0], X2[:, 2],
                      c=X2[:, 1], cmap="YlOrRd", alpha=0.5, s=15)
    ax5.set_xlabel("Future Return", fontsize=10)
    ax5.set_ylabel("Max Drawdown", fontsize=10)
    ax5.set_title("Return vs Risk", fontsize=11, fontweight="bold")
    plt.colorbar(sc5, ax=ax5, label="Future Vol")
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    h = ax6.hist2d(X1[:, rv_idx], X2[:, 0], bins=30, cmap="Blues")
    ax6.set_xlabel("Past Realized Vol", fontsize=10)
    ax6.set_ylabel("Future Return", fontsize=10)
    ax6.set_title("Joint Distribution μ(x₁, x₂)", fontsize=11, fontweight="bold")
    plt.colorbar(h[3], ax=ax6, label="Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path.name}")


def _plot_conditional_predictions(data, estimator, save_path):
    X1 = data["X1"]
    X2 = data["X2"]
    X1_raw = data["X1_raw"]
    lookback = data["lookback"]

    estimator.fit(X1, X2)

    vol_idx = lookback
    vol_values = X1_raw[:, vol_idx]
    low_idx = int(np.argmin(vol_values))
    high_idx = int(np.argmax(vol_values))
    med_idx = int(np.argsort(vol_values)[len(vol_values) // 2])

    query_indices = [low_idx, med_idx, high_idx]
    query_labels  = ["Low Vol", "Medium Vol", "High Vol"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, idx, label in zip(axes, query_indices, query_labels):
        query   = X1[idx]
        samples, _ = estimator.predict_conditional(query, n_samples=1000)
        ax.hist2d(samples[:, 0], samples[:, 1], bins=30, cmap="viridis")
        ax.set_xlabel("Future Return", fontsize=11)
        ax.set_ylabel("Future Volatility", fontsize=11)
        ax.set_title(f"μ(x₂|x₁) — {label}\nVol={vol_values[idx]:.3f}",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        mean_pred = samples.mean(axis=0)
        ax.plot(mean_pred[0], mean_pred[1], "r*", markersize=15,
                markeredgecolor="white", markeredgewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")

# Mai

def main(
    n_full: int  = 3000,
    n_basic: int  = 3000,
    n_regime: int  = 2000,
    lookback:int  = 20,
    horizon: int  = 5,
    n_paths_surface: int = 50_000,
    run_plots: bool = True,
    seed: int  = 42,
) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  generate_unified_data.py")
    print(f"  Output directory : {_DATA_DIR}")
    print(f"  Workers available: {N_CPUS}")
    print("=" * 70)

    # Heston parameters 
    heston_params = {
        "S0":    100.0,
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho":   -0.7,
        "v0":    0.04,
        "r":     0.0,
    }
    print("\n  Heston parameters:")
    for k, v in heston_params.items():
        print(f"    {k}: {v}")

    # Volatility surface 
    print("\n  [Step 1/4] Generating volatility surface…")
    strikes_norm = np.linspace(0.8, 1.2, 21)
    maturities   = np.array([0.25, 0.5, 0.75, 1.0])

    vol_surface = generate_heston_surface(
        heston_params, strikes_norm, maturities, n_paths=n_paths_surface
    )

    atm_idx = len(strikes_norm) // 2
    print(f"\n  Surface statistics:")
    print(f"Min vol: {vol_surface.min() * 100:.2f}%")
    print(f"Max vol: {vol_surface.max() * 100:.2f}%")
    print(f"Mean vol: {vol_surface.mean() * 100:.2f}%")
    print(f"ATM 1Y: {vol_surface[-1, atm_idx] * 100:.2f}%")

    # Prediction datasets─
    generator = ConditionalPriceGenerator(heston_params, seed=seed)

    print("\n  [Step 2/4] Full-feature dataset…")
    data_full = generator.generate_price_prediction_data(
        n_samples=n_full, lookback=lookback, horizon=horizon, features="full"
    )

    print("\n  [Step 3/4] Basic-feature dataset…")
    data_basic = generator.generate_price_prediction_data(
        n_samples=n_basic, lookback=lookback, horizon=horizon, features="basic"
    )

    print("\n  [Step 4/4] Regime-switching dataset…")
    data_regime = generator.generate_multimodal_regime_data(n_samples=n_regime)

    # Save .npz
    output_file = _DATA_DIR / "unified_heston_prediction_data.npz"

    np.savez(
        str(output_file),
        # surface
        vol_surface = vol_surface,
        strikes_norm = strikes_norm,
        maturities = maturities,
        # full-feature dataset
        full_X1 = data_full["X1"],
        full_X2 = data_full["X2"],
        full_X1_raw  = data_full["X1_raw"],
        full_X1_mean = data_full["X1_mean"],
        full_X1_std  = data_full["X1_std"],
        full_lookback = data_full["lookback"],
        full_horizon = data_full["horizon"],
        # basic-feature dataset
        basic_X1 = data_basic["X1"],
        basic_X2 = data_basic["X2"],
        basic_X1_raw  = data_basic["X1_raw"],
        basic_X1_mean = data_basic["X1_mean"],
        basic_X1_std  = data_basic["X1_std"],
        # regime-switching dataset
        regime_X1 = data_regime["X1"],
        regime_X2 = data_regime["X2"],
        regime_labels = data_regime["regimes"],
        # metadata
        heston_params = heston_params,
        method = "unified_heston_conditional_brenier",
        description   = ("Unified Heston vol surface and conditional price "
                         "prediction — generated by experiments/"
                         "generate_unified_data.py"),
    )

    size_mb = output_file.stat().st_size / 1e6
    print(f"\n  Saved: {output_file}")
    print(f"  Size : {size_mb:.2f} MB")

    # Plots
    if run_plots:
        print("\n  Generating visualisations…")
        _plot_surface(
            vol_surface, strikes_norm, maturities,
            _DATA_DIR / "heston_surface.png",
        )
        _plot_prediction_data(data_full, _DATA_DIR / "price_prediction_data.png")

        estimator = _OTEstimator(t=0.5)
        _plot_conditional_predictions(
            data_full, estimator,
            _DATA_DIR / "conditional_distributions.png",
        )

    # Summary
    print("\n  Generation complete.")
    print(f"  Files in {_DATA_DIR}:")
    for f in sorted(_DATA_DIR.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size / 1e3:.1f} KB)")

    return output_file

# CL

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate unified_heston_prediction_data.npz "
                    "in data/synthetic/"
    )
    p.add_argument("--n_full",   type=int, default=3000,
                   help="Samples for full-feature dataset (default: 3000)")
    p.add_argument("--n_basic",  type=int, default=3000,
                   help="Samples for basic-feature dataset (default: 3000)")
    p.add_argument("--n_regime", type=int, default=2000,
                   help="Samples for regime-switching dataset (default: 2000)")
    p.add_argument("--lookback", type=int, default=20,
                   help="Lookback window in trading days (default: 20)")
    p.add_argument("--horizon",  type=int, default=5,
                   help="Forecast horizon in trading days (default: 5)")
    p.add_argument("--n_paths_surface", type=int, default=50_000,
                   help="MC paths for vol surface pricing (default: 50000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip plot generation")
    return p.parse_args()


if __name__ == "__main__":
    import argparse          
    args = _parse_args()
    main(
        n_full = args.n_full,
        n_basic = args.n_basic,
        n_regime = args.n_regime,
        lookback = args.lookback,
        horizon = args.horizon,
        n_paths_surface = args.n_paths_surface,
        run_plots = not args.no_plots,
        seed = args.seed,
    )
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp as sp_logsumexp
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, Optional
from functools import lru_cache
import os, sys

try:
    from joblib import Parallel, delayed as jdelayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# optional Numba JIT 
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator
    

@njit(cache=True)
def _simulate_kernel(S0, v0, kappa, theta, sigma, rho, r,
                     dt, n_steps, n_paths, Z1, Z2):
    rho2 = np.sqrt(1.0 - rho * rho)
    S = np.full(n_paths, float(S0))
    v = np.full(n_paths, float(v0))
    for i in range(n_steps):
        w1 = Z1[i]
        w2 = rho * Z1[i] + rho2 * Z2[i]
        for j in range(n_paths):
            v_j  = max(v[j], 0.0)          # full truncation prevents sqrt(neg)
            sv   = np.sqrt(v_j * dt)
            v[j] = max(v_j + kappa * (theta - v_j) * dt + sigma * sv * w2[j], 0.0)
            S[j] = S[j] * np.exp((r - 0.5 * v_j) * dt + sv * w1[j])
    return S


@njit(cache=True)
def _simulate_kernel_full(S0, v0, kappa, theta, sigma, rho, r,
                           dt, n_steps, n_paths, Z1, Z2):
    rho2 = np.sqrt(1.0 - rho * rho)
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    for i in range(n_steps):
        w1 = Z1[i]
        w2 = rho * Z1[i] + rho2 * Z2[i]
        for j in range(n_paths):
            v_j = v[j, i] if v[j, i] > 0.0 else 0.0
            sv   = np.sqrt(v_j * dt)
            v[j, i + 1] = v_j + kappa * (theta - v_j) * dt + sigma * sv * w2[j]
            S[j, i + 1] = S[j, i] * np.exp((r - 0.5 * v_j) * dt + sv * w1[j])
    return S, v


# Optimised Metrics

def compute_rescaled_cost_matrix(X, Y, t, d1, d2):
    d = X.shape[1]
    scale = np.ones(d)
    scale[d1:] = np.sqrt(t)
    X_scaled = X * scale
    Y_scaled = Y * scale
    X_sq = np.einsum('ij,ij->i', X_scaled, X_scaled)[:, None]
    Y_sq = np.einsum('ij,ij->i', Y_scaled, Y_scaled)[None, :]
    C = 0.5 * (X_sq + Y_sq - 2.0 * X_scaled @ Y_scaled.T)
    np.maximum(C, 0.0, out=C)
    return C, np.diag(scale)


@njit(cache=True, fastmath=True)
def _sinkhorn_numba_kernel(M, log_n, log_m, epsilon, max_iter, tol):
    n = M.shape[0]
    m = M.shape[1]
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(max_iter):
        for i in range(n):
            mx = M[i, 0] + g[0]
            for j in range(1, m):
                v = M[i, j] + g[j]
                if v > mx:
                    mx = v
            s = 0.0
            for j in range(m):
                s += np.exp(M[i, j] + g[j] - mx)
            f[i] = -(epsilon * (np.log(s) + mx)) - epsilon * log_n
        g_old = g.copy()
        for j in range(m):
            mx = M[0, j] + f[0]
            for i in range(1, n):
                v = M[i, j] + f[i]
                if v > mx:
                    mx = v
            s = 0.0
            for i in range(n):
                s += np.exp(M[i, j] + f[i] - mx)
            g[j] = -(epsilon * (np.log(s) + mx)) - epsilon * log_m
        delta = 0.0
        for j in range(m):
            d = g[j] - g_old[j]
            if d < 0:
                d = -d
            if d > delta:
                delta = d
        if delta < tol:
            break
    return g


def sinkhorn_dual_potentials(C, epsilon, max_iter=1000, tol=1e-6):
    n, m = C.shape
    log_n = np.log(n)
    log_m = np.log(m)
    M = (-C / epsilon).astype(np.float64, copy=False)
    if _HAS_NUMBA:
        return _sinkhorn_numba_kernel(M, log_n, log_m, epsilon, max_iter, tol)
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(max_iter):
        f_new = -log_n * epsilon - sp_logsumexp(M + g[np.newaxis, :], axis=1) * epsilon
        g_new = -log_m * epsilon - sp_logsumexp(M + f_new[:, np.newaxis], axis=0) * epsilon
        delta = np.max(np.abs(g_new - g))
        f, g = f_new, g_new
        if delta < tol:
            break
    return g


def _logsumexp(a, axis):
    return sp_logsumexp(a, axis=axis)

# Heston Model

class HestonUnified:
    def __init__(self, S0=100, v0=0.04, kappa=2.0, theta=0.04,
                 sigma=0.3, rho=-0.7, r=0.0, mode='financial'):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.mode  = mode
        self.potentials = {}
        self.drift_corrections = {}
        self.entropic_maps = {}
        self.rescaling_params = {}
        self._drift_table: Dict = {}

    def _draw_normals(self, n_steps, n_paths, seed=None):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_steps, n_paths)), rng.standard_normal((n_steps, n_paths))

    # Naked simulation 

    def simulate_paths(self, T=1.0, n_steps=252, n_paths=10000,
                       seed=42, full_paths=True):
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        Z1, Z2 = self._draw_normals(n_steps, n_paths, seed)
        if full_paths:
            S, v = _simulate_kernel_full(self.S0, self.v0, self.kappa, self.theta,
                                         self.sigma, self.rho, self.r,
                                         dt, n_steps, n_paths, Z1, Z2)
            return S, v, t_grid
        else:
            S_T = _simulate_kernel(self.S0, self.v0, self.kappa, self.theta,
                                   self.sigma, self.rho, self.r,
                                   dt, n_steps, n_paths, Z1, Z2)
            return S_T, None, t_grid

    # Schrödinger calibration 

    def calibrate_schrodinger_potential(self,
                                        market_prices: np.ndarray,
                                        strikes: np.ndarray,
                                        T: float,
                                        max_iter: int = 200,
                                        tol: float = 1e-8,
                                        l2_reg: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        N  = len(strikes)
        df = np.exp(-self.r * T)

        # Large MC sample for accurate gradient
        n_mc = 100_000
        S_T, _, _ = self.simulate_paths(T=T, n_paths=n_mc, seed=42,
                                         full_paths=False)

        # φ matrix: (N, n_mc) — discounted call payoffs
        phi   = df * np.maximum(S_T[None, :] - strikes[:, None], 0)
        C_mkt = market_prices.copy()

        # Objective + analytic gradient (with L2 regularisation)
        def _obj(omega, phi_mat, n):
            score = omega @ phi_mat          # (n,)
            s_max = score.max()
            e     = np.exp(score - s_max)
            Z     = e.mean()
            logZ  = np.log(Z) + s_max
            E_phi = (phi_mat @ e) / (Z * n)  # (N,)
            reg   = 0.5 * l2_reg * float(omega @ omega)
            obj   = -float(omega @ C_mkt) + logZ + reg
            grad  = -C_mkt + E_phi + l2_reg * omega
            return obj, grad

        # Screen phase: small MC, 3 restarts
        n_sc = 10_000
        S_sc, _, _ = self.simulate_paths(T=T, n_paths=n_sc, seed=7,
                                          full_paths=False)
        phi_sc = df * np.maximum(S_sc[None, :] - strikes[:, None], 0)

        rng = np.random.default_rng(0)
        inits = [np.zeros(N)] + [rng.standard_normal(N) * 0.01 for _ in range(2)]

        def _run(w0, phi_mat, n, max_it, ftol, gtol):
            return minimize(lambda w: _obj(w, phi_mat, n),
                            w0, method='L-BFGS-B', jac=True,
                            options={'maxiter': max_it, 'ftol': ftol, 'gtol': gtol})

        if _HAS_JOBLIB:
            screen = Parallel(n_jobs=-1, prefer='threads')(
                jdelayed(_run)(w0, phi_sc, n_sc,
                               max(20, max_iter // 5), tol * 100, tol)
                for w0 in inits)
        else:
            screen = [_run(w0, phi_sc, n_sc,
                           max(20, max_iter // 5), tol * 100, tol)
                      for w0 in inits]

        best = min(screen, key=lambda r: r.fun)
        result = _run(best.x, phi, n_mc, max_iter, tol, tol * 0.1)
        omega_star = result.x

        # Diagnostics 
        score_diag = omega_star @ phi
        s_max = score_diag.max()
        w_diag = np.exp(score_diag - s_max)
        w_diag /= w_diag.sum()
        ess = 1.0 / float(np.sum(w_diag ** 2))
        ess_pct = 100.0 * ess / n_mc

        # Potential on strike grid (for display / drift table) 
        # f*(K_i) evaluated at representative S_T = K_i nodes
        phi_at_K = df * np.maximum(strikes[:, None] - strikes[None, :], 0)
        potential = -(phi_at_K @ omega_star)
        potential -= potential.mean()

        self.potentials[T] = {
            'strikes': strikes,
            'values': potential,
            'omega': omega_star,
            'df': df,
            'market_prices': market_prices,
            'S_T': S_T,    # seed=42 paths used during calibration
            'phi': phi,    # (N, n_mc) precomputed; reuse in pricing
        }
        self._drift_table.pop(T, None)

        print(f"  ESS = {ess:.0f} / {n_mc}  ({ess_pct:.1f}%)")
        return potential, omega_star

    # ── 3. Drift correction ───────────────────────────────────────────────

    def _build_drift_table(self, T, n_t=30, n_v=30, n_mc=500):
        t_vals = np.linspace(0.0, T * 0.99, n_t)
        v_vals = np.linspace(1e-4, 4 * self.theta, n_v)
        potential = self.potentials[T]
        omega_star = potential['omega']
        K_calib = potential['strikes']
        df_c = potential['df']

        def _lam(ti, vi):
            dt = T - ti
            if dt <= 0:
                return 0.0
            eps_v = max(1e-4, 0.01 * vi)
            n_steps = max(5, int(dt * 50))
            dt_s = dt / n_steps
            rng = np.random.default_rng(int(abs(ti * 1e4 + vi * 1e6)) % (2**31))
            Z1 = rng.standard_normal((n_steps, n_mc))
            Z2 = rng.standard_normal((n_steps, n_mc))

            def _E(v_init):
                S_T = _simulate_kernel(self.S0, v_init, self.kappa, self.theta,
                                         self.sigma, self.rho, self.r,
                                         dt_s, n_steps, n_mc, Z1, Z2)
                phi = df_c * np.maximum(S_T[None, :] - K_calib[:, None], 0)
                score = omega_star @ phi
                score = np.clip(score, -50.0, 50.0)
                s_max = score.max()
                return float(np.exp(s_max) * np.exp(score - s_max).mean())

            Ev  = _E(vi)
            Evp = _E(vi + eps_v)
            if Ev > 1e-10 and Evp > 1e-10:
                d_log = (np.log(Evp) - np.log(Ev)) / eps_v
            else:
                d_log = 0.0
            return (1 - self.rho**2) * self.sigma**2 * d_log

        if _HAS_JOBLIB:
            lam_flat = Parallel(n_jobs=-1, prefer='processes')(
                jdelayed(_lam)(ti, vi) for ti in t_vals for vi in v_vals)
        else:
            lam_flat = [_lam(ti, vi) for ti in t_vals for vi in v_vals]

        self._drift_table[T] = (t_vals, v_vals,
                                 np.array(lam_flat).reshape(n_t, n_v))

    def compute_drift_correction(self, t, S, v, T):
        if T not in self.potentials:
            return 0.0
        if T not in self._drift_table:
            self._build_drift_table(T)
        t_vals, v_vals, lam_table = self._drift_table[T]
        ti = np.clip(t, t_vals[0], t_vals[-1])
        vi = np.clip(v, v_vals[0], v_vals[-1])
        it = min(np.searchsorted(t_vals, ti, side='right') - 1, len(t_vals) - 2)
        iv = min(np.searchsorted(v_vals, vi, side='right') - 1, len(v_vals) - 2)
        wt = (ti - t_vals[it]) / (t_vals[it + 1] - t_vals[it] + 1e-30)
        wv = (vi - v_vals[iv]) / (v_vals[iv + 1] - v_vals[iv] + 1e-30)
        return float(lam_table[it,   iv  ] * (1-wt)*(1-wv) +
                     lam_table[it+1, iv  ] *    wt *(1-wv) +
                     lam_table[it,   iv+1] * (1-wt)*   wv  +
                     lam_table[it+1, iv+1] *    wt *   wv)

    # Calibrated path simulation

    def simulate_calibrated_paths(self, T=1.0, n_steps=252, n_paths=10000):
        if T not in self.potentials:
            raise ValueError(f"Not calibrated for T={T}.")
        if T not in self._drift_table:
            print("  Building drift lookup table (one-time cost)…")
            self._build_drift_table(T)
        t_vals, v_vals, lam_table = self._drift_table[T]
        dt     = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        rng    = np.random.default_rng()
        Z1_all = rng.standard_normal((n_steps, n_paths))
        Z2_all = rng.standard_normal((n_steps, n_paths))
        rho2   = np.sqrt(1.0 - self.rho**2)
        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        for i in range(n_steps):
            W1 = Z1_all[i]
            W2 = self.rho * Z1_all[i] + rho2 * Z2_all[i]
            v_curr = np.maximum(v[:, i], 0.0)
            ti = np.clip(t_grid[i], t_vals[0], t_vals[-1])
            it = min(np.searchsorted(t_vals, ti, side='right') - 1, len(t_vals) - 2)
            wt = (ti - t_vals[it]) / (t_vals[it + 1] - t_vals[it] + 1e-30)
            vi_arr = np.clip(v_curr, v_vals[0], v_vals[-1])
            iv_arr = np.clip(np.searchsorted(v_vals, vi_arr, side='right') - 1,
                             0, len(v_vals) - 2)
            wv_arr = (vi_arr - v_vals[iv_arr]) / (
                      v_vals[iv_arr + 1] - v_vals[iv_arr] + 1e-30)
            lam = (lam_table[it,   iv_arr  ] * (1-wt)*(1-wv_arr) +
                   lam_table[it+1, iv_arr  ] *    wt *(1-wv_arr) +
                   lam_table[it,   iv_arr+1] * (1-wt)*   wv_arr  +
                   lam_table[it+1, iv_arr+1] *    wt *   wv_arr)
            sv = np.sqrt(v_curr * dt)
            v[:, i+1] = v_curr + (self.kappa*(self.theta - v_curr) + lam)*dt + self.sigma*sv*W2
            S[:, i+1] = S[:, i] * np.exp((self.r - 0.5*v_curr)*dt + sv*W1)
        return S, v, t_grid

    # Conditional Entropic Brenier Map 

    def fit_conditional_brenier_map(self, X_samples, Y_samples, d1, d2,
                                    t=None, epsilon=None, method='theory'):
        n = X_samples.shape[0]
        if t is None:
            t = 0.1 * (n ** (-1/3)) if method == 'theory' else 0.06
        if epsilon is None:
            epsilon = t**2 if method == 'theory' else t / 5
        C, A_t = compute_rescaled_cost_matrix(X_samples, Y_samples, t, d1, d2)
        dual_g = sinkhorn_dual_potentials(C, epsilon, max_iter=1000, tol=1e-6)
        Y_scaled = Y_samples @ A_t.T
        d_full = Y_samples.shape[1]

        def T_epsilon_t(x):
            x = np.atleast_2d(x)
            n_query = x.shape[0]
            if x.shape[1] < d_full:
                x = np.concatenate([x, np.zeros((n_query, d_full - x.shape[1]))], axis=1)
            x_scaled = x @ A_t.T
            x_sq = np.einsum('qd,qd->q', x_scaled, x_scaled)[:, None]
            Y_sq = np.einsum('id,id->i', Y_scaled, Y_scaled)[None, :]
            dist = 0.5 * (x_sq + Y_sq - 2.0 * x_scaled @ Y_scaled.T)
            log_w = (dual_g[None, :] - dist) / epsilon
            log_w -= log_w.max(axis=1, keepdims=True)
            w = np.exp(log_w)
            w /= w.sum(axis=1, keepdims=True)
            return (w @ Y_samples).squeeze()

        self.entropic_maps[t]    = T_epsilon_t
        self.rescaling_params[t] = {'t': t, 'epsilon': epsilon, 'A_t': A_t}
        return T_epsilon_t

    # Conditional sampling

    def conditional_sample(self, x_condition, map_func, n_samples=1000):
        noise   = np.random.randn(n_samples, x_condition.shape[0])
        x_batch = np.tile(x_condition, (n_samples, 1))
        return map_func(np.concatenate([x_batch, noise], axis=1))

    # Option pricing 
    def option_prices_mc(self, strikes: np.ndarray, T: float = 1.0,
                         n_paths: int = 100_000, calibrated: bool = False,
                         n_steps: int = 50) -> np.ndarray:

        # Always simulate under P₀ same paths for both branches
        S_T, _, _ = self.simulate_paths(T=T, n_steps=n_steps,
                                         n_paths=n_paths, full_paths=False)
        df      = np.exp(-self.r * T)
        payoffs = df * np.maximum(S_T[None, :] - strikes[:, None], 0)

        if calibrated and T in self.potentials:
            pot = self.potentials[T]
            omega_star = pot['omega']
            K_calib = pot['strikes']
            df_c = pot['df']

            # Reuse the exact same paths and phi from calibration (seed=42)
            # Using a fresh MC draw would introduce path-sampling noise that,
            # amplified by large omega_star, collapses the IS weights (ESS→0)
            S_T_cal = pot['S_T']     # (n_mc,) — calibration paths
            phi_p = pot['phi']     # (N, n_mc) — identical to calibration batch
            n_cal = S_T_cal.shape[0]
            payoffs = df_c * np.maximum(S_T_cal[None, :] - strikes[:, None], 0)

            score = omega_star @ phi_p
            s_max = score.max()
            w = np.exp(score - s_max)
            w /= w.sum()

            ess = 1.0 / float(np.sum(w ** 2))
            ess_pct = 100.0 * ess / n_cal
            print(f"  Pricing ESS = {ess:.0f} / {n_cal}  ({ess_pct:.1f}%)")

            return (payoffs * w[None, :]).sum(axis=1)

        return payoffs.mean(axis=1)

if __name__ == "__main__":
    import time

    heston = HestonUnified(S0=100, v0=0.04, kappa=2.0, theta=0.04,
                           sigma=0.3, rho=-0.7, mode='financial')

    print("\nModel Parameters:")
    print(f"  S₀={heston.S0}, v₀={heston.v0}, κ={heston.kappa}, "
          f"θ={heston.theta}, σ={heston.sigma}, ρ={heston.rho}")

    # Martingale Schrödinger Bridge 
    strikes = np.linspace(80, 120, 10)
    T = 1.0

    t0 = time.perf_counter()
    market_prices = heston.option_prices_mc(strikes, T, n_paths=100_000)
    print(f"\nMarket prices MC: {time.perf_counter()-t0:.2f}s")
    print(f"Market prices:    {np.round(market_prices, 4)}")

    # l2_reg sweep: start at 1e-2 and tighten until quality is good enough 
    # Thresholds: MAE < 0.05  (≈1% of ATM price),  Max Err < 0.20
    MAE_THRESH     = 0.05
    MAXERR_THRESH  = 0.20
    l2_candidates  = [1e-2, 5e-3, 2e-3, 1e-3]
    best_result    = None   # (potential, omega_star, calibrated_prices, l2)

    for l2 in l2_candidates:
        print(f"\n  Trying l2_reg = {l2:.0e} …")
        t0 = time.perf_counter()
        potential, omega_star = heston.calibrate_schrodinger_potential(
            market_prices, strikes, T, l2_reg=l2)
        print(f"  Calibration:    {time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        calibrated_prices = heston.option_prices_mc(strikes, T, n_paths=100_000,
                                                     calibrated=True)
        print(f"Calibrated MC:  {time.perf_counter()-t0:.2f}s")

        mae = float(np.mean(np.abs(calibrated_prices - market_prices)))
        max_err = float(np.max(np.abs(calibrated_prices - market_prices)))
        print(f"  MAE={mae:.6f}  Max Err={max_err:.6f}", end="")

        best_result = (potential, omega_star, calibrated_prices, l2)
        if mae < MAE_THRESH and max_err < MAXERR_THRESH:
            print("good enough — stopping.")
            break
        else:
            print("tightening …")

    potential, omega_star, calibrated_prices, best_l2 = best_result
    print(f"\nω* range: [{omega_star.min():.4f}, {omega_star.max():.4f}]  (l2_reg={best_l2:.0e})")
    print(f"Potential range: [{potential.min():.4f}, {potential.max():.4f}]")

    print(f"\nCalibration Quality:")
    print(f"MAE = {np.mean(np.abs(calibrated_prices - market_prices)):.6f}")
    print(f"Max Err = {np.max(np.abs(calibrated_prices - market_prices)):.6f}")

    # Conditional Brenier Map 
    n_samples = 5000
    X1 = np.random.uniform(-3, 3, (n_samples, 1))
    X2 = np.tanh(X1) + np.random.gamma(1, 0.3, (n_samples, 1))
    X_joint = np.hstack([X1, X2])

    t0 = time.perf_counter()
    T_hat = heston.fit_conditional_brenier_map(
        X_joint, X_joint, d1=1, d2=1, method='practical')
    print(f"\nBrenier map fit:  {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    x_test = np.array([0.5])
    cond_samples = heston.conditional_sample(x_test, T_hat, n_samples=1000)
    print(f"Conditional 1000: {time.perf_counter()-t0:.3f}s")
    print(f"  Mean={np.mean(cond_samples):.4f}  Std={np.std(cond_samples):.4f}")

    print("\nDone!")

# Calibration interface: used by msb.py and other wrappers

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CalibrationResult:
    """Holds the outcome of a single calibration run"""
    method_name: str
    params: np.ndarray
    model_prices: np.ndarray
    n_iterations: int = 0
    loss_history: Optional[np.ndarray] = None
    success: bool = True
    message: str = ""

    # filled by compute_errors()
    mae: float = 0.0
    rmse: float = 0.0
    max_error: float = 0.0

    def compute_errors(self, market_prices: np.ndarray) -> None:
        errs = np.abs(self.model_prices - market_prices)
        self.mae = float(errs.mean())
        self.rmse = float(np.sqrt((errs ** 2).mean()))
        self.max_error = float(errs.max())


class CalibrationMethod(ABC):
    """Abstract base for all calibration methods (GP, MSB, ParticleFilter, SVI …)"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def calibrate(
        self,
        strikes:       np.ndarray,
        market_prices: np.ndarray,
        T:             float,
        **kwargs,
    ) -> CalibrationResult: ...

    @abstractmethod
    def price_calls(
        self,
        strikes: np.ndarray,
        T:       float,
        **kwargs,
    ) -> np.ndarray: ...

    def reset(self) -> None:
        """Optional: re-initialise state for a fresh run"""
        pass

class NaiveBaseline(CalibrationMethod):
    def __init__(self, model):
        self._model = model

    @property
    def name(self) -> str:
        return "NaiveBaseline"

    def calibrate(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        T: float,
        **kwargs,
    ) -> CalibrationResult:
        prices = self.price_calls(strikes, T, **kwargs)
        result = CalibrationResult(
            method_name=self.name,
            params=np.array([]),
            model_prices=prices,
        )
        result.compute_errors(market_prices)
        return result

    def price_calls(
        self,
        strikes: np.ndarray,
        T: float,
        **kwargs,
    ) -> np.ndarray:
        return self._model.option_prices_mc(strikes, T=T, **kwargs)
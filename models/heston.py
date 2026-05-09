import numpy as np
import os
import warnings
from pathlib import Path
from typing import Optional
import multiprocessing

try:
    from numba import njit, prange, float64, complex128
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*a, **kw): return lambda f: f
    def prange(n): return range(n)

N_CORES = multiprocessing.cpu_count()
N_PHYSICAL = max(1, N_CORES // 2)


@njit(cache=True, fastmath=True)
def _norm_cdf_nb(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    p = 1.0 - (1.0 / (2.506628274631001) * np.exp(-0.5 * x * x)) * poly
    return p if x >= 0.0 else 1.0 - p


@njit(cache=True, fastmath=True)
def _bs_call_nb(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if sigma <= 0.0 or T <= 0.0:
        return max(S - K * np.exp(-r * T), 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf_nb(d1) - K * np.exp(-r * T) * _norm_cdf_nb(d2)


@njit(cache=True, fastmath=True)
def _iv_from_price_nb(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    v0_guess: float,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-5,
    maxiter: int = 60,
) -> float:
    floor_price = max(S - K * np.exp(-r * T), 0.0) + 1e-8
    price = max(price, floor_price)

    f_lo = _bs_call_nb(S, K, T, r, lo) - price
    f_hi = _bs_call_nb(S, K, T, r, hi) - price

    if f_lo * f_hi > 0.0:
        return v0_guess

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        f_mid = _bs_call_nb(S, K, T, r, mid) - price
        if abs(hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


@njit(cache=True, fastmath=True, parallel=True)
def _iv_surface_from_calls_nb(
    call_prices: np.ndarray,
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    v0_guess: float,
) -> np.ndarray:
    P = len(maturities)
    M = len(strikes)
    iv = np.zeros((P, M), dtype=np.float32)
    for p in prange(P):
        T = maturities[p]
        if T <= 0.0:
            continue
        for m in range(M):
            iv[p, m] = _iv_from_price_nb(
                call_prices[p, m], S0, strikes[m], T, r, v0_guess,
            )
    return iv


def _euler_maruyama_variance(
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sqrt_dt = np.sqrt(dt)
    dW = rng.standard_normal((horizon, n_paths)) * sqrt_dt

    v = np.full(n_paths, max(v0, 1e-4), dtype=np.float64)
    for h in range(horizon):
        vc = np.maximum(v, 0.0)
        v = vc + kappa * (theta - vc) * dt + sigma * np.sqrt(vc) * dW[h]
        np.maximum(v, 1e-6, out=v)

    return v


@njit(cache=True, fastmath=True, parallel=True)
def _euler_maruyama_variance_nb(
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    horizon: int,
    n_paths: int,
    rng_seed: int,
) -> np.ndarray:
    v_final = np.empty(n_paths, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for i in prange(n_paths):
        rng_state = np.uint64(rng_seed + i * 6364136223846793005 + 1)
        v = max(v0, 1e-4)

        for _ in range(horizon):
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u1 = (float(rng_state) / 1.8446744e+19) + 0.5
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u2 = (float(rng_state) / 1.8446744e+19) + 0.5
            u1 = max(u1, 1e-15)
            u1 = min(u1, 1.0 - 1e-15)
            u2 = max(u2, 1e-15)
            u2 = min(u2, 1.0 - 1e-15)
            z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * 3.141592653589793 * u2)

            vc = max(v, 0.0)
            dw = z * sqrt_dt
            v = vc + kappa * (theta - vc) * dt + sigma * np.sqrt(vc) * dw
            v = max(v, 1e-6)

        v_final[i] = v

    return v_final


def _heston_char_fn(
    u: np.ndarray,
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> np.ndarray:
    i = 1j
    xi = kappa - rho * sigma * i * u
    d = np.sqrt(xi**2 + sigma**2 * u * (u + i))
    d = np.where(d.real < 0, -d, d)

    g = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T)

    C = (r * i * u * T
         + (kappa * theta / sigma**2)
         * ((xi - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = (xi - d) / sigma**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    log_F = np.log(S0) + r * T
    return np.exp(C + D * v0 + i * u * log_F)


def heston_iv_surface(
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    N_fft: int = 128,
    alpha: float = 1.5,
) -> np.ndarray:
    M_strikes = len(strikes)
    P_mats = len(maturities)
    N = N_fft

    eta = 0.25
    lam = 2.0 * np.pi / (N * eta)
    b = N * lam / 2.0

    v_grid = np.arange(N, dtype=np.float64) * eta
    k_grid = -b + lam * np.arange(N, dtype=np.float64)

    w = np.ones(N, dtype=np.float64)
    w[0] = 1.0 / 3.0
    w[-1] = 1.0 / 3.0
    w[1:-1:2] = 4.0 / 3.0
    w[2:-2:2] = 2.0 / 3.0

    log_K = np.log(strikes / S0)
    u_complex = (v_grid - (alpha + 1.0) * 1j)
    denom_base = (alpha**2 + alpha - v_grid**2
                  + 1j * (2 * alpha + 1) * v_grid)
    phase = np.exp(1j * v_grid * b) * w * eta

    T_col = maturities[:, np.newaxis]
    u_col = u_complex[np.newaxis, :]

    xi = kappa - rho * sigma * 1j * u_col
    d_sq = xi**2 + sigma**2 * u_col * (u_col + 1j)
    d = np.sqrt(d_sq)
    neg = d.real < 0
    d = np.where(neg, -d, d)

    g = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T_col)

    log_num = np.log((1.0 - g * exp_dT) / (1.0 - g) + 1e-300)
    C = (r * 1j * u_col * T_col
         + (kappa * theta / sigma**2) * ((xi - d) * T_col - 2.0 * log_num))
    D = (xi - d) / sigma**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT + 1e-300)

    log_F = np.log(S0) + r * T_col
    phi = np.exp(C + D * v0 + 1j * u_col * log_F)

    psi = np.exp(-r * T_col) * phi / (denom_base + 1e-300)
    x_batch = psi * phase[np.newaxis, :]
    fft_vals = (np.exp(-alpha * k_grid) / np.pi
                * np.fft.fft(x_batch, axis=1).real)

    call_prices = np.zeros((P_mats, M_strikes), dtype=np.float64)
    for p in range(P_mats):
        if maturities[p] <= 0:
            continue
        call_prices[p] = np.interp(log_K, k_grid, fft_vals[p])

    iv_surface = _iv_surface_from_calls_nb(
        call_prices,
        S0,
        strikes.astype(np.float64),
        maturities.astype(np.float64),
        r,
        float(np.sqrt(max(v0, 1e-6))),
    )

    return iv_surface


def heston_iv_surface_batch(
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    v0_arr: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    N_fft: int = 64,
    alpha: float = 1.5,
) -> np.ndarray:
    B = len(v0_arr)
    M = len(strikes)
    P = len(maturities)
    N = N_fft

    eta = 0.25
    lam = 2.0 * np.pi / (N * eta)
    b = N * lam / 2.0

    v_grid = np.arange(N, dtype=np.float64) * eta
    k_grid = -b + lam * np.arange(N, dtype=np.float64)

    w = np.ones(N, dtype=np.float64)
    w[0] = 1.0 / 3.0
    w[-1] = 1.0 / 3.0
    w[1:-1:2] = 4.0 / 3.0
    w[2:-2:2] = 2.0 / 3.0

    log_K = np.log(strikes / S0)
    T_col = maturities[:, np.newaxis]
    u_col = (v_grid - (alpha + 1.0) * 1j)[np.newaxis, :]

    xi = kappa - rho * sigma * 1j * u_col
    d_sq = xi**2 + sigma**2 * u_col * (u_col + 1j)
    d = np.sqrt(d_sq)
    d = np.where(d.real < 0, -d, d)

    g = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T_col)

    log_num = np.log((1.0 - g * exp_dT) / (1.0 - g) + 1e-300)
    C = (r * 1j * u_col * T_col
         + (kappa * theta / sigma**2) * ((xi - d) * T_col - 2.0 * log_num))
    D = (xi - d) / sigma**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT + 1e-300)

    log_F = np.log(S0) + r * T_col
    phase = np.exp(1j * v_grid * b) * w * eta
    denom = alpha**2 + alpha - v_grid**2 + 1j*(2*alpha+1)*v_grid

    v0_col = v0_arr[:, np.newaxis, np.newaxis].astype(np.complex128)
    phi_B = np.exp(C[np.newaxis] + D[np.newaxis] * v0_col
                   + 1j * u_col[np.newaxis] * log_F[np.newaxis])

    exp_rT = np.exp(-r * T_col)[np.newaxis]
    psi_B = exp_rT * phi_B / (denom + 1e-300)
    x_B = psi_B * phase

    flat = x_B.reshape(B * P, N)
    fft_flat = np.fft.fft(flat, axis=1).real
    fft_B = (np.exp(-alpha * k_grid) / np.pi
             * fft_flat).reshape(B, P, N)

    call_B = np.zeros((B, P, M), dtype=np.float64)
    for bi in range(B):
        for p in range(P):
            if maturities[p] <= 0:
                continue
            call_B[bi, p] = np.interp(log_K, k_grid, fft_B[bi, p])

    iv_B = np.zeros((B, P, M), dtype=np.float32)
    for bi in range(B):
        iv_B[bi] = _iv_surface_from_calls_nb(
            call_B[bi],
            S0,
            strikes.astype(np.float64),
            maturities.astype(np.float64),
            r,
            float(np.sqrt(max(v0_arr[bi], 1e-6))),
        )

    return iv_B


class HestonCalibrator:
    _PARAM_BOUNDS = [(0.1, 8.0), (0.001, 0.25), (0.01, 2.0), (-0.99, -0.05)]
    _PARAM_NAMES = ["kappa", "theta", "sigma", "rho"]
    N_REGIMES = 4

    def __init__(
        self,
        surface_model,
        K: int,
        pc_stds_k: np.ndarray,
        moneyness_grid: np.ndarray,
        maturity_grid: np.ndarray,
        r: float = 0.0,
        window: int = 60,
        n_mc: int = 500,
        seed: int = 42,
    ):
        self.surface_model = surface_model
        self.K = K
        self.pc_stds_k = pc_stds_k
        self.moneyness_grid = moneyness_grid
        self.maturity_grid = maturity_grid
        self.r = r
        self.window = window
        self.n_mc = n_mc
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        self._params_cache: dict[int, np.ndarray] = {}
        self._last_params = np.array([2.0, 0.04, 0.3, -0.7])

        self._surfaces_real: Optional[np.ndarray] = None
        self._X_source: Optional[np.ndarray] = None
        self._X_target: Optional[np.ndarray] = None
        self._split: Optional[int] = None
        self._global_params: Optional[np.ndarray] = None

        self._regime_params: Optional[np.ndarray] = None
        self._regime_edges: Optional[np.ndarray] = None

    def _scores_to_surface(self, scores_k: np.ndarray) -> np.ndarray:
        shape = scores_k.shape[:-1]
        flat = scores_k.reshape(-1, self.K).astype(np.float32)
        flat_r = flat * self.pc_stds_k
        K_full = self.surface_model.pca.n_components
        pad = np.zeros((len(flat_r), K_full - self.K), dtype=np.float32)
        full = np.concatenate([flat_r, pad], axis=1)
        surfs = self.surface_model.pca.inverse_transform(full)
        P, M = len(self.maturity_grid), len(self.moneyness_grid)
        return surfs.reshape(*shape, P, M) if shape else surfs[0].reshape(P, M)

    def _surface_to_scores(self, surface_pm: np.ndarray) -> np.ndarray:
        single = surface_pm.ndim == 2
        if single:
            surface_pm = surface_pm[np.newaxis]
        flat = surface_pm.reshape(len(surface_pm), -1).astype(np.float32)
        scores_full = self.surface_model.pca.transform(flat)[:, :self.K]
        scores_w = scores_full / self.pc_stds_k
        return scores_w[0] if single else scores_w

    _INIT_GRID = np.array([
        [2.0, 0.04, 0.3, -0.7],
        [4.0, 0.05, 0.5, -0.8],
        [1.0, 0.03, 0.2, -0.5],
        [6.0, 0.02, 0.8, -0.9],
        [3.0, 0.08, 0.6, -0.6],
        [1.5, 0.04, 0.4, -0.85],
    ], dtype=np.float64)

    def _calibrate_window(
        self,
        surfaces_window: np.ndarray,
        S0: float = 1.0,
        max_iter: int = 200,
        popsize: int = 8,
        seed: int = 0,
        regime_vol: float = 0.20,
    ) -> np.ndarray:
        from scipy.optimize import differential_evolution

        target_iv = surfaces_window.mean(axis=0)
        S0_norm = 1.0
        strikes = self.moneyness_grid * S0_norm
        atm_idx = len(self.moneyness_grid) // 2
        v0_est = float(max(target_iv[0, atm_idx] ** 2, 0.001))
        v0_arr = np.array([v0_est])

        def _loss(params):
            kappa, theta, sigma, rho = params
            feller_pen = max(0.0, sigma**2 - 2.0 * kappa * theta) * 10.0
            try:
                iv_B = heston_iv_surface_batch(
                    S0_norm, strikes, self.maturity_grid, self.r,
                    v0_arr, kappa, theta, sigma, rho, N_fft=64,
                )
                return float(np.mean((iv_B[0] - target_iv) ** 2)) + feller_pen
            except Exception:
                return 1e6

        theta_prior = float(np.clip(regime_vol ** 2, 0.001, 0.24))
        regime_prior = np.array([3.0, theta_prior, 0.4, -0.7])

        x0_candidates = np.vstack([
            regime_prior,
            self._INIT_GRID,
            np.clip(self._last_params, [b[0] for b in self._PARAM_BOUNDS],
                                       [b[1] for b in self._PARAM_BOUNDS]),
        ])

        best_params = None
        best_loss = np.inf

        for i, x0 in enumerate(x0_candidates):
            try:
                result = differential_evolution(
                    _loss,
                    bounds=self._PARAM_BOUNDS,
                    maxiter=max_iter,
                    popsize=popsize,
                    seed=seed + i * 7,
                    tol=1e-5,
                    workers=1,
                    init="latinhypercube",
                    x0=x0,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x.copy()
            except Exception:
                continue

        if best_params is None:
            best_params = self._last_params.copy()

        self._last_params = best_params.copy()
        return best_params

    def _simulate_future_surfaces(
        self,
        v0: float,
        params: np.ndarray,
        horizon: int = 5,
        n_paths: int = 500,
    ) -> np.ndarray:
        kappa, theta, sigma, rho = params
        rho_c = np.sqrt(max(1.0 - rho**2, 0.0))
        dt = 1.0 / 252.0
        sqdt = np.sqrt(dt)

        Z = self.rng.standard_normal((horizon, n_paths, 2))
        dW_v = Z[:, :, 0] * sqdt
        dW_S = (rho * Z[:, :, 0] + rho_c * Z[:, :, 1]) * sqdt

        v = np.full(n_paths, max(v0, 1e-4), dtype=np.float64)
        logS = np.zeros(n_paths, dtype=np.float64)

        for h in range(horizon):
            vc = np.maximum(v, 0.0)
            sv = np.sqrt(vc)
            logS += (self.r - 0.5 * vc) * dt + sv * dW_S[h]
            v = vc + kappa * (theta - vc) * dt + sigma * sv * dW_v[h]
            np.maximum(v, 1e-6, out=v)

        S_final = np.exp(logS)
        v_final = v

        S0_norm = 1.0
        P, M = len(self.maturity_grid), len(self.moneyness_grid)
        surfs = np.zeros((n_paths, P, M), dtype=np.float32)

        def _one_surface(i: int, fallback: np.ndarray) -> np.ndarray:
            Si = float(S_final[i])
            v0_i = float(v_final[i])
            strikes_i = self.moneyness_grid * S0_norm
            try:
                iv = heston_iv_surface(
                    Si, strikes_i, self.maturity_grid, self.r,
                    v0_i, kappa, theta, sigma, rho,
                    N_fft=64,
                )
                return iv
            except Exception:
                return fallback

        if n_paths >= 32 and N_PHYSICAL > 1:
            from joblib import Parallel, delayed
            fallback = np.zeros((P, M), dtype=np.float32)
            try:
                fallback = _one_surface(0, np.zeros((P, M), dtype=np.float32))
            except Exception:
                pass
            results = Parallel(n_jobs=N_PHYSICAL, prefer="threads")(
                delayed(_one_surface)(i, fallback) for i in range(n_paths)
            )
            for i, s in enumerate(results):
                surfs[i] = s
        else:
            prev = np.zeros((P, M), dtype=np.float32)
            for i in range(n_paths):
                s = _one_surface(i, prev)
                surfs[i] = s
                prev = s

        return surfs

    def _get_regime_params(self, x: np.ndarray) -> np.ndarray:
        if self._regime_params is None or self._regime_edges is None:
            return self._global_params

        pc1 = float(x[0])
        edges = self._regime_edges
        r = int(np.searchsorted(edges[1:-1], pc1))
        r = max(0, min(r, self.N_REGIMES - 1))
        return self._regime_params[r]

    def fit(
        self,
        X_src_train: np.ndarray,
        X_tgt_train: np.ndarray,
        horizon: int = 5,
        max_iter: int = 150,
        verbose: bool = True,
    ) -> "HestonCalibrator":
        self._X_source = X_src_train
        self._X_target = X_tgt_train
        self._horizon = horizon
        self._dates: Optional[np.ndarray] = None

        surfs_src = self._scores_to_surface(X_src_train)
        pc1_train = X_src_train[:, 0]

        if verbose:
            print(f"  Heston regime calibration ({len(X_src_train)} surfaces, "
                  f"{self.N_REGIMES} regimes)...")
            print(f"  Moneyness: {self.moneyness_grid[[0, -1]]}  "
                  f"Maturities: {self.maturity_grid}")
            print(f"  Numba: {_NUMBA}  Cores: {N_PHYSICAL}")

        quantile_edges = np.linspace(0, 100, self.N_REGIMES + 1)
        edges = np.percentile(pc1_train, quantile_edges)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self._regime_edges = edges

        regime_params = np.zeros((self.N_REGIMES, 4))
        self._last_params = np.array([2.0, 0.04, 0.3, -0.7])

        for r in range(self.N_REGIMES):
            mask = (pc1_train >= edges[r]) & (pc1_train < edges[r + 1])
            n_r = mask.sum()

            if n_r < 10:
                regime_params[r] = regime_params[r - 1] if r > 0 else self._last_params
                if verbose:
                    print(f"  Regime {r+1}: n={n_r} (too small, copying previous)")
                continue

            surfs_r = surfs_src[mask]
            regime_vol = float(surfs_r[:, 0, len(self.moneyness_grid) // 2].mean())

            if verbose:
                pc1_lo = edges[r] if np.isfinite(edges[r]) else pc1_train[mask].min()
                pc1_hi = edges[r+1] if np.isfinite(edges[r+1]) else pc1_train[mask].max()
                print(f"  Regime {r+1}/{self.N_REGIMES}: n={n_r}  "
                      f"PC1=[{pc1_lo:.2f}, {pc1_hi:.2f}]  ", end="", flush=True)

            params_r = self._calibrate_window(
                surfs_r, max_iter=max_iter, seed=r, regime_vol=regime_vol,
            )
            regime_params[r] = params_r

            if verbose:
                kappa, theta, sigma, rho = params_r
                print(f"kappa={kappa:.3f}  theta={theta:.4f}  sigma={sigma:.3f}  rho={rho:.3f}")

        self._regime_params = regime_params
        self._global_params = regime_params[self.N_REGIMES // 2].copy()

        if verbose:
            print(f"\n  Parameters by regime:")
            print(f"  {'Regime':8s}  {'kappa':>7}  {'theta':>7}  {'sigma':>7}  {'rho':>7}  "
                  f"{'vol_lt':>7}  {'half-life':>10}")
            for r in range(self.N_REGIMES):
                kappa, theta, sigma, rho = regime_params[r]
                feller_ok = "ok" if 2*kappa*theta > sigma**2 else "warn"
                hl = np.log(2) / kappa * 252 if kappa > 0 else float("inf")
                print(f"  R{r+1:d}      {feller_ok}  "
                      f"{kappa:7.3f}  {theta:7.4f}  {sigma:7.3f}  {rho:7.3f}  "
                      f"{np.sqrt(theta)*100:6.2f}%  {hl:8.1f}d")

        return self

    def fit_with_dates(
        self,
        X_src_train: np.ndarray,
        X_tgt_train: np.ndarray,
        dates: Optional[np.ndarray] = None,
        horizon: int = 5,
        max_iter: int = 150,
        verbose: bool = True,
    ) -> "HestonCalibrator":
        self.fit(X_src_train, X_tgt_train, horizon=horizon,
                 max_iter=max_iter, verbose=verbose)
        self._dates = np.asarray(dates) if dates is not None else None
        return self

    def predict(self, x1_query: np.ndarray) -> np.ndarray:
        x1 = np.atleast_2d(x1_query).astype(np.float32)
        N = len(x1)

        def _predict_one(xi):
            return self.sample(xi, n=self.n_mc).mean(axis=0)

        if N == 1:
            out = _predict_one(x1[0])[np.newaxis]
        elif N_PHYSICAL > 1 and N >= 4:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=N_PHYSICAL, prefer="threads")(
                delayed(_predict_one)(xi) for xi in x1
            )
            out = np.stack(results, axis=0)
        else:
            out = np.stack([_predict_one(xi) for xi in x1], axis=0)

        return out.squeeze() if x1_query.ndim == 1 else out

    def sample(self, x: np.ndarray, n: int = 500) -> np.ndarray:
        if self._global_params is None:
            raise RuntimeError("Call fit() before sample().")

        x = np.atleast_1d(x).astype(np.float32)
        if x.ndim > 1:
            x = x[0]

        surf_now = self._scores_to_surface(x)
        atm_idx = len(self.moneyness_grid) // 2
        v0_est = max(float(surf_now[0, atm_idx] ** 2), 1e-4)

        params = self._get_regime_params(x)

        surfs_future = self._simulate_future_surfaces(
            v0=v0_est, params=params,
            horizon=getattr(self, "_horizon", 5), n_paths=n,
        )
        scores_future = self._surface_to_scores(surfs_future)
        return scores_future.astype(np.float32)

    def sample_batch(self, X: np.ndarray, n: int = 200) -> np.ndarray:
        if N_PHYSICAL > 1 and len(X) >= 4:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=N_PHYSICAL, prefer="threads")(
                delayed(self.sample)(xi, n) for xi in X
            )
            return np.stack(results, axis=0)

        return np.stack([self.sample(xi, n=n) for xi in X], axis=0)

    def save(self, path: str):
        import pickle
        state = {
            "K": self.K,
            "pc_stds_k": self.pc_stds_k,
            "moneyness_grid": self.moneyness_grid,
            "maturity_grid": self.maturity_grid,
            "r": self.r,
            "window": self.window,
            "n_mc": self.n_mc,
            "_global_params": self._global_params,
            "_last_params": self._last_params,
            "_params_cache": self._params_cache,
            "_horizon": getattr(self, "_horizon", 5),
            "_regime_params": self._regime_params,
            "_regime_edges": self._regime_edges,
            "_dates": getattr(self, "_dates", None),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=5)
        print(f"Heston saved to: {path}")

    @classmethod
    def load(cls, path: str, surface_model) -> "HestonCalibrator":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.surface_model = surface_model
        for k, v in state.items():
            setattr(obj, k, v)
        obj.rng = np.random.default_rng(42)
        obj._seed = 42
        obj._X_source = None
        obj._X_target = None
        if not hasattr(obj, "_regime_params"):
            obj._regime_params = None
        if not hasattr(obj, "_regime_edges"):
            obj._regime_edges = None
        if not hasattr(obj, "_dates"):
            obj._dates = None
        print(f"Heston loaded from: {path}")
        return obj


def load_surface_heston_data(
    npz_path: str,
    surface_model_path: str,
    horizon: int = 5,
    filter_covid: bool = True,
    covid_start: str = "2020-02-15",
    covid_end: str = "2021-01-01",
) -> dict:
    import importlib.util
    from datetime import datetime

    _sm_dir = Path(__file__).resolve().parent
    _sm_file = _sm_dir / "surface_model.py"
    if not _sm_file.exists():
        _sm_file = Path("surface_model.py")
    if not _sm_file.exists():
        raise FileNotFoundError("surface_model.py not found.")

    spec = importlib.util.spec_from_file_location("surface_model", str(_sm_file))
    sm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm_mod)

    sm = sm_mod.SurfaceModel()
    if Path(surface_model_path).exists():
        sm.load(surface_model_path)
        print("  SurfaceModel loaded.")
    else:
        print("  Checkpoint not found — training SurfaceModel...")
        surf_hist_path = str(Path(npz_path).parent / "surface_history.npz")
        sm.fit(npz_path, surface_history_path=surf_hist_path)
        Path(surface_model_path).parent.mkdir(parents=True, exist_ok=True)
        sm.save(surface_model_path)

    scores_full = sm.pc_history.astype(np.float32)
    var_expl = sm.pca.explained_variance_ratio_
    pc_stds = scores_full.std(axis=0)

    _dates_full: Optional[np.ndarray] = None
    for _attr in ("dates", "date_index", "index", "dates_"):
        if hasattr(sm, _attr):
            _d = getattr(sm, _attr)
            if _d is not None and len(_d) == len(scores_full):
                _dates_full = np.asarray(_d)
                break

    VAR_EXPL_MIN = 0.001
    n_components_avail = len(pc_stds)
    mask_std = pc_stds > 0.01
    mask_var = var_expl[:n_components_avail] > VAR_EXPL_MIN
    mask_ok = mask_std & mask_var
    n_natural = int(mask_ok.sum())
    n_floor_ok = int(mask_var[:min(5, n_components_avail)].sum())
    meaningful = min(max(n_natural, n_floor_ok), n_components_avail)

    scores = scores_full[:, :meaningful].copy()
    N, K = scores.shape
    print(f"  PCs retained: {K} of {n_components_avail}"
          f"  ({var_expl[:K].sum()*100:.2f}% variance)")

    pc_stds_k = np.maximum(scores.std(axis=0), 1e-6)
    scores = scores / pc_stds_k
    print(f"  Whitening: pc_stds = {np.round(pc_stds_k, 4).tolist()}")

    dates_source: Optional[np.ndarray] = None
    if _dates_full is not None:
        dates_source = _dates_full[:N - horizon].copy()

    X_source = scores[:N - horizon].copy()
    X_target = scores[horizon:].copy()

    if filter_covid and _dates_full is not None:
        try:
            t0_covid = np.datetime64(covid_start, "D")
            t1_covid = np.datetime64(covid_end, "D")
            dates_cmp = dates_source.astype("datetime64[D]")
            mask_ok = ~((dates_cmp >= t0_covid) & (dates_cmp <= t1_covid))
            n_removed = int((~mask_ok).sum())
            if n_removed > 0:
                X_source = X_source[mask_ok]
                X_target = X_target[mask_ok]
                dates_source = dates_source[mask_ok]
                print(f"  [filter_covid] {n_removed} days removed "
                      f"({covid_start} -> {covid_end})")
        except Exception as _e:
            warnings.warn(f"[load_surface_heston_data] filter_covid failed: {_e}")

    print(f"  N={len(X_source):,}  K={K}  horizon={horizon}d")

    try:
        hist = np.load(str(Path(npz_path).parent / "surface_history.npz"),
                       allow_pickle=False)
        moneyness_grid = hist.get("moneyness_grid", np.linspace(0.80, 1.20, 21))
        maturity_grid = hist.get("maturity_grid", np.array([0.25, 0.50, 0.75, 1.00]))
    except Exception:
        moneyness_grid = np.linspace(0.80, 1.20, 21)
        maturity_grid = np.array([0.25, 0.50, 0.75, 1.00])

    return {
        "X_source": X_source,
        "X_target": X_target,
        "surface_model": sm,
        "K": K,
        "pc_stds_k": pc_stds_k,
        "horizon": horizon,
        "moneyness_grid": moneyness_grid,
        "maturity_grid": maturity_grid,
        "dates": dates_source,
    }


def main():
    import time

    print("\n" + "=" * 62)
    print("  Heston Calibrator — Vol Surface (PCA)")
    print(f"  CPU cores: {N_CORES}  |  Numba: {_NUMBA}")
    print("=" * 62)

    NPZ_PATH = r"C:\volatility-options\data\live_spx_data_extended.npz"
    SM_PATH = r"C:\volatility-options\data\surface_model.npz"
    HESTON_PATH = r"C:\volatility-options\data\heston_surface.pkl"

    print("\n-- Loading data --")
    bundle = load_surface_heston_data(NPZ_PATH, SM_PATH, horizon=5)
    X_source = bundle["X_source"]
    X_target = bundle["X_target"]
    surface_model = bundle["surface_model"]
    K = bundle["K"]
    pc_stds_k = bundle["pc_stds_k"]
    moneyness_grid = bundle["moneyness_grid"]
    maturity_grid = bundle["maturity_grid"]
    dates = bundle.get("dates")
    N = len(X_source)

    split = int(0.8 * N)
    X_src_train = X_source[:split]
    X_tgt_train = X_target[:split]
    X_src_test = X_source[split:]
    X_tgt_test = X_target[split:]
    dates_train = dates[:split] if dates is not None else None
    print(f"  Train: {len(X_src_train):,}  Test: {len(X_src_test):,}")

    print("\n-- Heston calibration --")
    cal = HestonCalibrator(
        surface_model=surface_model,
        K=K,
        pc_stds_k=pc_stds_k,
        moneyness_grid=moneyness_grid,
        maturity_grid=maturity_grid,
        r=0.0,
        window=60,
        n_mc=500,
        seed=42,
    )

    t0 = time.perf_counter()
    cal.fit_with_dates(X_src_train, X_tgt_train, dates=dates_train,
                       horizon=5, max_iter=150)
    print(f"  Calibration completed in {time.perf_counter() - t0:.1f}s")

    print("\n-- Evaluation --")

    def _pad_and_reconstruct(scores_k: np.ndarray) -> np.ndarray:
        shape = scores_k.shape[:-1]
        flat = scores_k.reshape(-1, K).astype(np.float32)
        flat_r = flat * pc_stds_k
        K_full = surface_model.pca.n_components
        pad = np.zeros((len(flat_r), K_full - K), dtype=np.float32)
        full = np.concatenate([flat_r, pad], axis=1)
        surfs = surface_model.pca.inverse_transform(full)
        P, M = len(maturity_grid), len(moneyness_grid)
        return surfs.reshape(*shape, P, M) if shape else surfs[0]

    n_eval = min(200, len(X_src_test))
    idx = np.random.choice(len(X_src_test), n_eval, replace=False)
    X_eval = X_src_test[idx]
    X_true = X_tgt_test[idx]

    t0 = time.perf_counter()
    preds = cal.predict(X_eval)
    print(f"  Predict {n_eval} points: {time.perf_counter()-t0:.2f}s")

    errs_pc = np.abs(preds - X_true).mean(axis=1)
    print(f"\nMAE mean (PCA space):  {errs_pc.mean():.5f}")
    print(f"MAE p25/p75:           {np.percentile(errs_pc,25):.5f} / "
          f"{np.percentile(errs_pc,75):.5f}")

    pred_surfs = _pad_and_reconstruct(preds)
    true_surfs = _pad_and_reconstruct(X_true)
    errs_iv = np.abs(pred_surfs - true_surfs).mean(axis=(1, 2))
    print(f"\nMAE mean (IV scale):   {errs_iv.mean():.6f}")
    print(f"MAE p25/p75 (IV):      {np.percentile(errs_iv,25):.6f} / "
          f"{np.percentile(errs_iv,75):.6f}")
    print(f"MAE p90 (IV):          {np.percentile(errs_iv,90):.6f}")

    print(f"\n-- MAE by maturity (ATM) --")
    mats_label = [f"{t:.2f}Y" for t in maturity_grid]
    atm_idx = len(moneyness_grid) // 2
    ts_mae = np.abs(pred_surfs[:, :, atm_idx] - true_surfs[:, :, atm_idx]).mean(axis=0)
    ts_span_pred = (pred_surfs[:, :, atm_idx].max(axis=1) - pred_surfs[:, :, atm_idx].min(axis=1)).mean()
    ts_span_true = (true_surfs[:, :, atm_idx].max(axis=1) - true_surfs[:, :, atm_idx].min(axis=1)).mean()
    for i, lbl in enumerate(mats_label):
        print(f"  {lbl}  MAE_ATM={ts_mae[i]:.5f}")
    print(f"  Term structure span -- pred: {ts_span_pred:.4f}  true: {ts_span_true:.4f}")

    per_pc_mae = np.abs(preds - X_true).mean(axis=0)
    var_expl = surface_model.pca.explained_variance_ratio_
    print(f"\n  {'PC':5s}  {'MAE':>8}  {'var_expl':>9}")
    for k in range(K):
        print(f"  PC{k+1:<3d}  {per_pc_mae[k]:8.5f}  {var_expl[k]*100:8.1f}%")

    pc1_test = X_eval[:, 0]
    atm_bias = pred_surfs[:, 0, atm_idx] - true_surfs[:, 0, atm_idx]
    quartis = np.percentile(pc1_test, [0, 25, 50, 75, 100])
    print(f"\n-- Bias diagnostics by regime (PC1) --")
    print(f"  {'Regime':12s}  {'N':>5}  {'bias_ATM':>10}  {'MAE_IV':>8}")
    for i, label in enumerate(["Q1 (low)", "Q2", "Q3", "Q4 (high)"]):
        mask = (pc1_test >= quartis[i]) & (pc1_test < quartis[i + 1])
        if mask.sum() == 0:
            continue
        print(f"  {label:12s}  {mask.sum():5d}  "
              f"{atm_bias[mask].mean():+10.5f}  {errs_iv[mask].mean():8.5f}")
    print(f"  {'TOTAL':12s}  {n_eval:5d}  "
          f"{atm_bias.mean():+10.5f}  {errs_iv.mean():8.5f}")

    print(f"\nDemo (x_test[0]) -- distribution of 1000 surfaces:")
    x_demo = X_src_test[0]
    score_samples = cal.sample(x_demo, n=1000)
    surf_samples = _pad_and_reconstruct(score_samples)
    true_surf = _pad_and_reconstruct(X_tgt_test[0])

    print(f"  {'Mat':>5}  {'ATM_true':>9}  {'ATM_mean':>9}  {'ATM_std':>8}  "
          f"{'skew_true':>10}  {'skew_mean':>10}")
    for i, T in enumerate(maturity_grid):
        print(f"  {T:.2f}Y  {true_surf[i, atm_idx]:9.4f}  "
              f"{surf_samples[:, i, atm_idx].mean():9.4f}  "
              f"{surf_samples[:, i, atm_idx].std():8.4f}  "
              f"{true_surf[i, 0] - true_surf[i, -1]:10.4f}  "
              f"{(surf_samples[:, i, 0] - surf_samples[:, i, -1]).mean():10.4f}")

    cal.save(HESTON_PATH)
    print("\nPipeline complete.")
    print(f"\nCalibrated parameters by regime:")
    if cal._regime_params is not None:
        print(f"  {'Regime':8s}  {'kappa':>7}  {'theta':>7}  {'sigma':>7}  {'rho':>7}  "
              f"{'vol_lt':>7}  {'half-life':>10}")
        for r in range(cal.N_REGIMES):
            kappa, theta, sigma, rho = cal._regime_params[r]
            hl = np.log(2) / kappa * 252 if kappa > 0 else float("inf")
            print(f"  R{r+1:d}        "
                  f"{kappa:7.4f}  {theta:7.4f}  {sigma:7.4f}  {rho:7.4f}  "
                  f"{np.sqrt(theta)*100:6.2f}%  {hl:8.1f}d")
    else:
        kappa, theta, sigma, rho = cal._global_params
        print(f"  kappa={kappa:.4f}  theta={theta:.4f}  sigma={sigma:.4f}  rho={rho:.4f}")
        print(f"  long-run vol: {np.sqrt(theta)*100:.2f}%  "
              f"half-life: {np.log(2)/kappa*252:.1f} business days")


class HestonUnified:
    def __init__(
        self,
        S0: float = 100.0,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        r: float = 0.0,
        seed: int = 42,
    ):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self._rng = np.random.default_rng(seed)

    def option_prices_mc(
        self,
        strikes: np.ndarray,
        T: float = 1.0,
        n_paths: int = 10_000,
        n_steps: int = 252,
    ) -> np.ndarray:
        strikes = np.asarray(strikes, dtype=np.float64)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        logS = np.zeros(n_paths, dtype=np.float64)
        v = np.full(n_paths, max(self.v0, 1e-4), dtype=np.float64)

        sqrt_1mrho2 = np.sqrt(max(1.0 - self.rho ** 2, 0.0))
        for _ in range(n_steps):
            vc = np.maximum(v, 0.0)
            Z1 = self._rng.standard_normal(n_paths)
            Z2 = self._rng.standard_normal(n_paths)
            Wv = Z1
            Ws = self.rho * Z1 + sqrt_1mrho2 * Z2

            v = vc + self.kappa * (self.theta - vc) * dt \
                   + self.sigma * np.sqrt(vc) * Wv * sqrt_dt
            np.maximum(v, 1e-6, out=v)

            logS += (self.r - 0.5 * vc) * dt + np.sqrt(vc) * Ws * sqrt_dt

        S_T = self.S0 * np.exp(logS)
        df = np.exp(-self.r * T)

        prices = np.empty(len(strikes), dtype=np.float64)
        for i, K in enumerate(strikes):
            payoffs = np.maximum(S_T - K, 0.0)
            prices[i] = df * payoffs.mean()

        return prices


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()
    main()
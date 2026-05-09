import numpy as np
import os
from datetime import date
from pathlib import Path
from numba import njit, prange

# número de cores físicos disponíveis
try:
    import psutil
    N_PHYSICAL = psutil.cpu_count(logical=False) or os.cpu_count() or 1
except ImportError:
    N_PHYSICAL = os.cpu_count() or 1

N_CORES = os.cpu_count() or 1

PARALLEL_THRESHOLD = 1000
PREDICT_CHUNK      = 256
_DIST_STD_WARN     = 0.002   # [FIX-5] threshold para aviso de colapso


@njit(parallel=True, cache=True, fastmath=True)
def _sinkhorn_numba(Ceps: np.ndarray, epsilon: float,
                    max_iter: int = 500, tol: float = 1e-6) -> np.ndarray:
    n = Ceps.shape[0]
    f = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)

    for _ in range(max_iter):
        f_old = f.copy()

        for i in prange(n):
            mx = -1e38
            for j in range(n):
                val = g[j] - Ceps[i, j]
                if val > mx:
                    mx = val
            s = 0.0
            for j in range(n):
                s += np.exp(g[j] - Ceps[i, j] - mx)
            f[i] = -(epsilon * (np.log(s) + mx))

        for j in prange(n):
            mx = -1e38
            for i in range(n):
                val = f[i] - Ceps[i, j]
                if val > mx:
                    mx = val
            s = 0.0
            for i in range(n):
                s += np.exp(f[i] - Ceps[i, j] - mx)
            g[j] = -(epsilon * (np.log(s) + mx))

        diff = 0.0
        for i in range(n):
            d = f[i] - f_old[i]
            if d < 0.0:
                d = -d
            if d > diff:
                diff = d
        if diff < tol:
            break

    return g


@njit(parallel=True, cache=True, fastmath=True)
def _predict_numba(XS: np.ndarray, Y_scaled: np.ndarray,
                   Y_target: np.ndarray, dual_g: np.ndarray,
                   epsilon: float) -> np.ndarray:
    n_q = XS.shape[0]
    n = Y_scaled.shape[0]
    d = XS.shape[1]
    d2 = Y_target.shape[1]

    out = np.zeros((n_q, d2), dtype=np.float64)

    for q in prange(n_q):
        log_w = np.empty(n, dtype=np.float64)
        mx = -1e38
        for i in range(n):
            dist_sq = 0.0
            for k in range(d):
                diff = XS[q, k] - Y_scaled[i, k]
                dist_sq += diff * diff
            lw = (dual_g[i] - 0.5 * dist_sq) / epsilon
            log_w[i] = lw
            if lw > mx:
                mx = lw

        s = 0.0
        for i in range(n):
            log_w[i] = np.exp(log_w[i] - mx)
            s += log_w[i]
        for i in range(n):
            log_w[i] /= s

        for i in range(n):
            w = log_w[i]
            for k in range(d2):
                out[q, k] += w * Y_target[i, k]

    return out


class ConditionalBrenierEstimator:

    def __init__(self, d1: int, d2: int, t: float = 0.01,
                 epsilon: float = 0.001,
                 girsanov_alpha: float = 1.0,   
                 n_jobs: int = -1):
        self.d1 = d1
        self.d2  = d2
        self.t = t
        self.epsilon = float(epsilon)
        self.girsanov_alpha = girsanov_alpha     
        self.n_jobs = n_jobs

        d = d1 + d2
        self.A_t = np.eye(d)
        self.A_t[d1:, d1:] *= np.sqrt(t)

        self.X_train = None
        self.Y_train = None
        self.dual_potentials= None
        self._Y_scaled = None
        self._Y_target = None
        self._pc_stds_x1 = None
        self._pc_stds_x2 = None
        self._bias_models = None

    # Internal helpers

    def _sq_dist_half(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq_A = np.einsum('id,id->i', A, A)[:, np.newaxis]
        sq_B = np.einsum('id,id->i', B, B)[np.newaxis, :]
        return 0.5 * (sq_A + sq_B - 2.0 * (A @ B.T))

    def _compute_rescaled_cost(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Rescaled cost matrix  c_t(xᵢ, yⱼ) = ½‖A_t (xᵢ − yⱼ)‖²

        X : (m, d)  joint points (d = d1 + d2)
        Y : (n, d)
        Returns (m, n) cost matrix
        """
        Xs = X @ self.A_t.T
        Ys = Y @ self.A_t.T
        return self._sq_dist_half(Xs, Ys)

    def _sinkhorn_logdomain(self, C: np.ndarray, max_iter: int = 500,
                            tol: float = 1e-6) -> np.ndarray:
        """Sinkhorn log-domain inplace em float32 — fallback serial."""
        Ceps = (C / self.epsilon).astype(np.float32)
        n = Ceps.shape[0]
        f = np.zeros(n, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)
        buf = np.empty_like(Ceps)

        for _ in range(max_iter):
            f_old = f.copy()

            np.subtract(g[np.newaxis, :], Ceps, out=buf)
            buf -= buf.max(axis=1, keepdims=True)
            np.exp(buf, out=buf)
            f = (-self.epsilon * np.log(buf.sum(axis=1))).astype(np.float32)

            np.subtract(f[:, np.newaxis], Ceps, out=buf)
            buf -= buf.max(axis=0, keepdims=True)
            np.exp(buf, out=buf)
            g = (-self.epsilon * np.log(buf.sum(axis=0))).astype(np.float32)

            if np.max(np.abs(f - f_old)) < tol:
                break

        return g.astype(np.float64)

    def _sinkhorn_parallel(self, C: np.ndarray,
                           max_iter: int = 500,
                           tol: float = 1e-6) -> np.ndarray:
        """Sinkhorn paralelizado via joblib — fallback se Numba falhar."""
        from scipy.special import logsumexp
        from joblib import Parallel, delayed

        n, m = C.shape
        Ceps = C / self.epsilon
        n_workers  = N_PHYSICAL if self.n_jobs == -1 else max(1, self.n_jobs)
        chunk_size = max(1, n // n_workers)

        def _f_chunk(start, end, g_cur):
            return -self.epsilon * logsumexp(
                g_cur[np.newaxis, :] / self.epsilon - Ceps[start:end], axis=1
            )

        f = np.zeros(n)
        g = np.zeros(m)

        for _ in range(max_iter):
            f_old = f.copy()
            chunks  = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
            f_parts = Parallel(n_jobs=n_workers, backend="threading")(
                delayed(_f_chunk)(s, e, g) for s, e in chunks
            )
            f = np.concatenate(f_parts)
            g = -self.epsilon * logsumexp(
                f[:, np.newaxis] / self.epsilon - Ceps, axis=0
            )
            if np.max(np.abs(f - f_old)) < tol:
                break

        return g

    def _sinkhorn(self, C: np.ndarray, max_iter: int = 500,
                  tol: float = 1e-6) -> np.ndarray:
        """Despacha para Numba → paralelo → serial."""
        try:
            Ceps = (C / self.epsilon).astype(np.float32)
            g    = _sinkhorn_numba(Ceps, float(self.epsilon), max_iter, tol)
            return g.astype(np.float64)
        except Exception:
            if C.shape[0] > PARALLEL_THRESHOLD and N_PHYSICAL > 1:
                return self._sinkhorn_parallel(C, max_iter, tol)
            return self._sinkhorn_logdomain(C, max_iter, tol)

    # Public API

    def fit(self, X1: np.ndarray, X2: np.ndarray,
            weights: np.ndarray | None = None) -> "ConditionalBrenierEstimator":

        self._pc_stds_x1 = np.maximum(X1.std(axis=0), 1e-6)
        self._pc_stds_x2 = np.maximum(X2.std(axis=0), 1e-6)
        X1w = X1 / self._pc_stds_x1
        X2w = X2 / self._pc_stds_x2

        joint        = np.hstack([X1w, X2w])
        self.X_train = joint
        self.Y_train = joint

        if weights is None:
            pc1   = X1w[:, 0]
            pc1_z = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)

            vol_regime = np.abs(pc1_z).mean()  
            alpha_adaptive = self.girsanov_alpha * (1.0 + np.clip(vol_regime - 0.8, 0, 1.5))
            alpha_adaptive = np.clip(alpha_adaptive, 1.0, 3.0)

            log_w = -0.5 * pc1_z ** 2 * alpha_adaptive  
            w     = np.exp(log_w - log_w.max())

            low_vol_mask  = pc1_z < -1.5   
            high_vol_mask = pc1_z >  1.5   

            boost_q1 = 4.0
            boost_q4 = np.clip(2.0 + 2.0 * np.clip(vol_regime - 0.8, 0, 1.0), 2.0, 4.0)

            w[low_vol_mask]  *= boost_q1
            w[high_vol_mask] *= boost_q4

            w       = np.clip(w, 0.1, None)
            weights = (w / w.mean()).astype(np.float32)
            _using_girsanov = True
        else:
            alpha_adaptive = self.girsanov_alpha
            low_vol_mask = np.zeros(joint.shape[0], dtype=bool)
            high_vol_mask = np.zeros(joint.shape[0], dtype=bool)
            boost_q1 = 1.0
            boost_q4 = 1.0
            _using_girsanov = False
            weights = np.asarray(weights, dtype=np.float32)
            weights = weights / (weights.mean() + 1e-9)
        self._weights = weights

        Ys = joint @ self.A_t.T
        C = self._sq_dist_half(Ys, Ys)
        C_w  = C * weights[:, np.newaxis]

        n = joint.shape[0]
        _parallel_str = 'sim' if n > PARALLEL_THRESHOLD and N_PHYSICAL > 1 else 'não'
        if weights is None:
            print(f"  Sinkhorn: n={n}  d={joint.shape[1]}  "
                  f"ε={self.epsilon:.2e}  t={self.t:.2e}  "
                  f"parallel={_parallel_str}  "
                  f"girsanov_alpha={self.girsanov_alpha:.2f}→{alpha_adaptive:.2f}(adapt)  "
                  f"boost {boost_q1:.0f}×Q1({low_vol_mask.sum()}pts) {boost_q4:.1f}×Q4({high_vol_mask.sum()}pts)")
        else:
            print(f"  Sinkhorn: n={n}  d={joint.shape[1]}  "
                  f"ε={self.epsilon:.2e}  t={self.t:.2e}  "
                  f"parallel={_parallel_str}  "
                  f"girsanov_alpha={self.girsanov_alpha:.2f}  weights=external")

        self.dual_potentials = self._sinkhorn(C_w)
        self._Y_scaled = Ys
        self._Y_target = joint[:, self.d1:]

        return self

    def predict(self, x1_query: np.ndarray,
                return_distribution: bool = False,
                n_samples: int = 1000,
                chunk_size: int = PREDICT_CHUNK,
                n_mean_samples: int = 50):
        if self.dual_potentials is None:
            raise RuntimeError("Call fit() before predict().")

        x1_query = np.atleast_2d(x1_query).astype(np.float64)
        n_q      = x1_query.shape[0]

        x1_w  = x1_query / self._pc_stds_x1
        x_ext = np.zeros((n_q, self.d1 + self.d2))
        x_ext[:, :self.d1] = x1_w
        XS = x_ext @ self.A_t.T

        all_preds = []
        w_last    = None
        w_all_list = [] 

        for start in range(0, n_q, chunk_size):
            end  = min(start + chunk_size, n_q)
            xs   = XS[start:end]

            dist = self._sq_dist_half(xs, self._Y_scaled)
            log_w = (self.dual_potentials[np.newaxis, :] - dist) / self.epsilon
            log_w -= log_w.max(axis=1, keepdims=True)
            w_mat  = np.exp(log_w)
            w_mat /= w_mat.sum(axis=1, keepdims=True)

            preds_w = w_mat @ self._Y_target
            all_preds.append(preds_w)
            w_all_list.append(w_mat)  # [FIX-14]

            if end == n_q:
                w_last = w_mat[-1]

        self._last_w_all = np.concatenate(w_all_list, axis=0)  # [FIX-14]

        predictions_w = np.concatenate(all_preds, axis=0)
        predictions   = predictions_w * self._pc_stds_x2

        if n_mean_samples > 0 and not return_distribution:
            jitter_scale = np.sqrt(self.epsilon) * 2.0
            mean_preds = np.zeros_like(predictions_w)
            w_all = self._last_w_all if hasattr(self, "_last_w_all") else None
            if w_all is not None:
                for qi in range(predictions_w.shape[0]):
                    counts = np.random.multinomial(n_mean_samples, w_all[qi])
                    idx_s = np.repeat(np.arange(len(counts)), counts)
                    samp = self._Y_target[idx_s] + np.random.randn(n_mean_samples, self._Y_target.shape[1]) * jitter_scale
                    mean_preds[qi] = samp.mean(axis=0)
                predictions = mean_preds * self._pc_stds_x2

        if getattr(self, "_global_bias", None) is not None:
            predictions = predictions - self._conditional_bias(x1_query)

        prediction = predictions.squeeze()

        if return_distribution:
            counts = np.random.multinomial(n_samples, w_last)
            idx = np.repeat(np.arange(len(counts)), counts)
            np.random.shuffle(idx)

            samples_w = self._Y_target[idx].copy()

            jitter_scale = np.sqrt(self.epsilon) * 2.0
            samples_w  += np.random.randn(*samples_w.shape) * jitter_scale
            samples = samples_w * self._pc_stds_x2

            sample_stds = samples.std(axis=0)
            if np.any(sample_stds < _DIST_STD_WARN):
                import warnings
                warnings.warn(
                    f"[Brenier] sample standard deviation too low: "
                    f"{np.round(sample_stds, 5).tolist()}. "
                    f"Consider increasing ε (current={self.epsilon:.2e}) "
                    f"or reviewing the PC selection. "
                    f"Suggested method: make_brenier_estimator(..., method='adaptive').",
                    UserWarning, stacklevel=2
                    )

            if getattr(self, "_global_bias", None) is not None:
                samples = samples - self._conditional_bias(x1_query[-1:])

            return prediction, samples

        return prediction

    def calibrate_conformal(
        self,
        X1_cal: np.ndarray,
        X2_cal: np.ndarray,
        coverage: float = 0.90,
        n_mean_samples: int = 0,
    ):
        preds = self.predict(X1_cal, n_mean_samples=n_mean_samples)
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]

        residuals = np.abs(preds - X2_cal)   # (n_cal, d2)
        n_cal = residuals.shape[0]
        q_level = np.ceil((n_cal + 1) * coverage) / n_cal
        q_level = min(q_level, 1.0)
        self._conformal_bands   = np.quantile(residuals, q_level, axis=0)
        self._conformal_coverage = coverage
        print(f"  [Conformal] target coverage ={coverage:.0%}  "
              f"n_cal={n_cal}  "
              f"bands (por PC): {np.round(self._conformal_bands, 5).tolist()}")

    def predict_interval(
        self,
        x1_query: np.ndarray,
        n_mean_samples: int = 0,
    ) -> tuple:
        if not hasattr(self, "_conformal_bands"):
            raise RuntimeError("Call calibrate_conformal() before predict_interval().")
        pred = self.predict(x1_query, n_mean_samples=n_mean_samples)
        pred = np.atleast_2d(pred)
        band  = self._conformal_bands[np.newaxis, :]
        return pred, pred - band, pred + band


    def calibrate_bias_conditional(
        self,
        X1_cal: np.ndarray,
        X2_cal: np.ndarray,
        n_quantiles: int = 8,
        min_n_quantile: int = 10,
    ):
        preds = self.predict(X1_cal)
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]

        raw_residuals = preds - X2_cal
        self._global_bias = raw_residuals.mean(axis=0)
        self._bias_models = []

        pc1_cal = X1_cal[:, 0]
        edges = np.percentile(pc1_cal, np.linspace(0, 100, n_quantiles + 1))
        edges[0]  -= 1e-6
        edges[-1] += 1e-6

        quartile_bias = np.tile(self._global_bias, (n_quantiles, 1)).copy()
        print(f"Calibração de bias por octil de PC1 (n_quantiles={n_quantiles}):")
        print(f"bias global (fallback): {np.round(self._global_bias, 5).tolist()}")

        bin_bias_pc1 = np.zeros(n_quantiles)
        bin_ns = np.zeros(n_quantiles, dtype=int)

        for q in range(n_quantiles):
            mask = (pc1_cal >= edges[q]) & (pc1_cal < edges[q + 1])
            n_q  = mask.sum()
            bin_ns[q] = n_q
            if n_q >= min_n_quantile:
                quartile_bias[q]  = raw_residuals[mask].mean(axis=0)
                bin_bias_pc1[q]   = quartile_bias[q][0]
                flag = ""
            else:
                bin_bias_pc1[q] = self._global_bias[0]
                flag = f"  ← fallback global (n={n_q} < {min_n_quantile})"
            b     = quartile_bias[q]
            label = f"O{q+1} [PC1∈({edges[q]:.2f},{edges[q+1]:.2f})]"
            print(f"    {label}  n={n_q:4d}  "
                  f"bias_PC1={b[0]:+.5f}  bias_PC2={b[1]:+.5f}{flag}")

        ranks   = np.arange(n_quantiles, dtype=float)
        corr_lin = float(np.corrcoef(ranks, bin_bias_pc1)[0, 1])

        half = n_quantiles // 2
        low_mean = bin_bias_pc1[:half].mean()
        high_mean = bin_bias_pc1[half:].mean()
        antis = abs(low_mean + high_mean) < 0.5 * (abs(low_mean) + abs(high_mean) + 1e-9)
        u_shape = (bin_bias_pc1[0] > 0 and bin_bias_pc1[-1] > 0 and
                     bin_bias_pc1[half - 1] < 0 and bin_bias_pc1[half] < 0)

        print(f"\n Bias antisymmetry diagnostics (PC1)")
        print(f"Rank-bias correlation (linear):  {corr_lin:+.3f}")
        print(f"Average bias in lower bins (Q1–Q{half}): {low_mean:+.5f}")
        print(f"Average bias in upper bins (Q{half+1}–Q{n_quantiles}): {high_mean:+.5f}")

        if antis and abs(low_mean) > 0.03:
            print(f"ANTISYMMETRIC pattern detected: low volatility overestimated, "
                f"high volatility underestimated (or vice versa). "
                f"Probable cause: Girsanov does not properly capture the transition "
                f"between intermediate regimes (Q{half}↔Q{half+1}). "
                f"Suggested action: increase girsanov_alpha or add PC2 as a "
                f"secondary conditioning dimension in the boost.")

        elif u_shape:
            print(f"U-SHAPE pattern detected: extremes with positive bias, "
                f"intermediate regimes with negative bias. "
                f"Suggests excessive smoothing (large t) in middle regimes.")

        elif abs(corr_lin) > 0.7:
            direction = "increasing" if corr_lin > 0 else "decreasing"
            print(f"MONOTONIC {direction} pattern: Girsanov systematically "
                f"underrepresents one side of the PC1 distribution. "
                f"Review the asymmetric boost in fit().")

        else:
            print(f"No strong structural antisymmetry pattern detected.")
        if abs(quartile_bias[0, 0]) > 0.05:
            import warnings
            warnings.warn(
                f"[Brenier] high residual O1 bias after boost: "
                f"bias_PC1={quartile_bias[0,0]:+.5f}. "
                f"Consider increasing boost_factor in fit() or girsanov_alpha.",
                UserWarning, stacklevel=2
            )

        self._quartile_bias  = quartile_bias
        self._quartile_edges = edges

    def _conditional_bias(self, X1: np.ndarray) -> np.ndarray:
        global_b = getattr(self, "_global_bias", None)
        X1 = np.atleast_2d(X1)
        n  = X1.shape[0]

        qbias  = getattr(self, "_quartile_bias",  None)
        qedges = getattr(self, "_quartile_edges", None)

        if qbias is None or qedges is None:
            if global_b is None:
                return np.zeros((n, self.d2), dtype=np.float32)
            return np.broadcast_to(global_b[np.newaxis, :], (n, self.d2)).astype(np.float32)

        pc1  = X1[:, 0]
        bins = np.searchsorted(qedges[1:-1], pc1, side="right")
        bins = np.clip(bins, 0, len(qbias) - 1)
        return qbias[bins].astype(np.float32)

    def compute_wasserstein_error(self, x1_test: np.ndarray,
                                  x2_test: np.ndarray,
                                  verbose: bool = False) -> float:
        preds = self.predict(x1_test)
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]

        mse_original = float(np.sqrt(np.mean((preds - x2_test) ** 2)))

        if verbose:
            # erro no espaço whitened — elimina amplificação de pc_stds_x2
            preds_w  = preds  / self._pc_stds_x2
            x2_test_w = x2_test / self._pc_stds_x2
            mse_white = float(np.sqrt(np.mean((preds_w - x2_test_w) ** 2)))
            per_pc    = np.sqrt(np.mean((preds - x2_test) ** 2, axis=0))
            per_pc_w  = np.sqrt(np.mean((preds_w - x2_test_w) ** 2, axis=0))
            print(f"  W₂ proxy (escala original):  {mse_original:.5f}")
            print(f"  W₂ proxy (whitened):         {mse_white:.5f}")
            print(f"  RMSE por PC (original): {np.round(per_pc, 5).tolist()}")
            print(f"  RMSE por PC (whitened):  {np.round(per_pc_w, 5).tolist()}")

        return mse_original

    def save(self, path: str):
        """Serializa o estimador via pickle."""
        import pickle
        state = {
            "d1": self.d1,
            "d2": self.d2,
            "t":  self.t,
            "epsilon": self.epsilon,
            "girsanov_alpha": self.girsanov_alpha,   
            "A_t": self.A_t,
            "X_train": self.X_train,
            "dual_potentials": self.dual_potentials,
            "_Y_scaled": self._Y_scaled,
            "_Y_target": self._Y_target,
            "_pc_stds_x1": self._pc_stds_x1,
            "_pc_stds_x2": self._pc_stds_x2,
            "_weights": getattr(self, "_weights", None),
            "_bias_models": getattr(self, "_bias_models", None),
            "_global_bias": getattr(self, "_global_bias", None),
            "_quartile_bias": getattr(self, "_quartile_bias", None),
            "_quartile_edges": getattr(self, "_quartile_edges", None),
            "_conformal_bands": getattr(self, "_conformal_bands", None),
            "_conformal_coverage": getattr(self, "_conformal_coverage", None),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=5)
        print(f"Brenier salvo em: {path}")

    @classmethod
    def load(cls, path: str) -> "ConditionalBrenierEstimator":
        """Carrega estimador serializado."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        est = cls(
            d1=state["d1"], d2=state["d2"],
            t=state["t"],   epsilon=state["epsilon"],
            girsanov_alpha=state.get("girsanov_alpha", 1.0),  # [FIX-2] compat
        )
        for k, v in state.items():
            setattr(est, k, v)
        print(f"Brenier carregado de: {path}")
        return est

# Convenience factor

def make_brenier_estimator(n_samples: int, d1: int, d2: int,
                            method: str = "adaptive",
                            girsanov_alpha: float = 1.0,
                            n_jobs: int = -1) -> ConditionalBrenierEstimator:
    if method == "theory":
        t       = 0.1  * (n_samples ** (-1 / 3))
        epsilon = t ** 2
    elif method == "practical":
        t       = 0.1  * (n_samples ** (-1 / 5))
        epsilon = t / 5.0
    elif method == "adaptive":
        # [FIX-4] usa escala theory para t mas floors ε em t/3
        t       = 0.1  * (n_samples ** (-1 / 3))
        epsilon = float(max(t ** 2, t / 3.0))
    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Choose 'theory', 'practical', or 'adaptive'.")
    return ConditionalBrenierEstimator(d1=d1, d2=d2, t=t, epsilon=epsilon,
                                       girsanov_alpha=girsanov_alpha,
                                       n_jobs=n_jobs)
# Pipeline helper

def _filter_covid(surfaces: np.ndarray, dates: list,
                   covid_start: str = "2020-02-15",
                   covid_end:   str = "2021-01-01") -> tuple[np.ndarray, list]:

    from datetime import datetime

    start = datetime.strptime(covid_start, "%Y-%m-%d").date()
    end   = datetime.strptime(covid_end,   "%Y-%m-%d").date()

    keep = np.array([not (start <= d < end) for d in dates])
    n_removed = int((~keep).sum())

    surfaces_clean = surfaces[keep]
    dates_clean    = [d for d, k in zip(dates, keep) if k]

    print(f"  [COVID filter] Removidos {n_removed} dias "
          f"({covid_start} → {covid_end}). "
          f"Série restante: {len(dates_clean)} dias "
          f"({dates_clean[0]} → {dates_clean[-1]})")
    return surfaces_clean, dates_clean


def load_surface_brenier_data(
    surface_history_path: str,
    surface_model_path: str,
    npz_path: str,
    horizon: int  = 5,
    filter_covid: bool = True,
    covid_start: str  = "2020-02-15",
    covid_end: str  = "2021-01-01",
) -> dict:
    import importlib.util

    print(f"\n  Carregando surface_history de: {surface_history_path}")
    hist = np.load(surface_history_path, allow_pickle=False)
    surfaces_raw = hist["surfaces"].astype(np.float32)      # (N, 4, 21)
    dates_raw    = [date.fromisoformat(str(d)) for d in hist["dates"]]
    print(f"History: {surfaces_raw.shape}  "
          f"({dates_raw[0]} → {dates_raw[-1]})")

    if filter_covid:
        surfaces_use, dates_use = _filter_covid(
            surfaces_raw, dates_raw, covid_start, covid_end
        )
    else:
        surfaces_use, dates_use = surfaces_raw, dates_raw
        print("  [COVID filter] using complete series.")

    _tmp_surf_path = Path(surface_history_path).parent / "_surface_history_filtered.npz"
    np.savez(
        str(_tmp_surf_path),
        surfaces = surfaces_use,
        dates= np.array([str(d) for d in dates_use]),
        moneyness_grid = hist.get("moneyness_grid", np.linspace(0.80, 1.20, 21)),
        maturity_grid  = hist.get("maturity_grid",  np.array([0.25, 0.50, 0.75, 1.00])),
    )
    print(f" Filtered surface history filtrado saved: {_tmp_surf_path}")

    _sm_dir  = Path(__file__).resolve().parent
    _sm_file = _sm_dir / "surface_model.py"
    if not _sm_file.exists():
        _sm_file = Path("surface_model.py")
    if not _sm_file.exists():
        raise FileNotFoundError("surface_model.py not found")

    spec   = importlib.util.spec_from_file_location("surface_model", str(_sm_file))
    sm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm_mod)

    sm = sm_mod.SurfaceModel()
    if Path(surface_model_path).exists():
        sm.load(surface_model_path)
        print("SurfaceModel loaded")
    else:
        print("SurfaceModel not found")
        sm.fit(npz_path, surface_history_path=str(_tmp_surf_path))
        sm.save(surface_model_path)
        print(f" SurfaceModel saved: {surface_model_path}")

    scores_full = sm.pc_history.astype(np.float32)
    var_expl = sm.pca.explained_variance_ratio_
    pc_stds  = scores_full.std(axis=0)

    VAR_EXPL_MIN = 0.001   
    n_components_avail = len(pc_stds)

    mask_std = pc_stds > 0.01
    mask_var = var_expl[:n_components_avail] > VAR_EXPL_MIN
    mask_ok  = mask_std & mask_var

    n_natural = int(mask_ok.sum())

    n_floor   = min(5, n_components_avail)
    n_floor_ok = int(mask_var[:n_floor].sum())

    meaningful = max(n_natural, n_floor_ok)
    meaningful = min(meaningful, n_components_avail)

    dropped = [i + 1 for i in range(meaningful, n_components_avail)
               if not mask_var[i]]
    if any(not mask_var[i] for i in range(meaningful)):
        pass

    scores = scores_full[:, :meaningful].copy()
    N, K = scores.shape
    print(f"  PCs retidas: {K} de {n_components_avail}"
          f"  ({var_expl[:K].sum()*100:.2f}% variância)")
    print(f"  pc_stds: {np.round(scores.std(axis=0), 4).tolist()}")
    print(f"  var_expl por PC: {[f'{v*100:.3f}%' for v in var_expl[:K]]}")

    quasi_zero = [(i+1, float(var_expl[i])*100) for i in range(K)
                  if var_expl[i] <= VAR_EXPL_MIN]
    if quasi_zero:
        import warnings
        warnings.warn(
            f"[FIX-12] Quasi-zero PCs within the retained set: "
            f"{quasi_zero}. "
            f"Increase VAR_EXPL_MIN or reduce n_components in SurfaceModel.",
            UserWarning, stacklevel=2,
        )

    dropped_by_var = [(i+1, float(var_expl[i])*100)
                    for i in range(K, min(K + 5, n_components_avail))
                    if var_expl[i] <= VAR_EXPL_MIN]

    if dropped_by_var:
        print(f"  [FIX-12] PCs discarded due to var_expl < {VAR_EXPL_MIN*100:.1f}%: "
            f"{dropped_by_var}")

    try:
        _test_scores = np.zeros((1, sm.pca.n_components), dtype=np.float32)
        _test_scores[0, :K] = scores.mean(axis=0)

        _rec = sm.pca.inverse_transform(_test_scores).reshape(4, 21)
        _atm = _rec[:, 10]

        _ts_spread = _atm.max() - _atm.min()

        if _ts_spread < 0.002:
            import warnings
            warnings.warn(
                f"[FIX-8] Collapsed term structure: ATM span={_ts_spread:.5f} < 0.002. "
                f"The retained PCs do not differentiate maturities. "
                f"Check whether historical_surface_builder is generating real "
                f"term structure variation "
                f"(VIX3M/VIX6M/VIX1Y available?).",
                UserWarning, stacklevel=2,
            )
        else:
            print(f"  Term structure span (ATM 0.25Y→1.00Y): {_ts_spread:.4f}  ✓")

    except Exception:
        pass

    X_source = scores[:N - horizon].copy()
    X_target = scores[horizon:].copy()
    print(f"  N={len(X_source):,}  K={K}  horizon={horizon}d")

    return {
        "X_source": X_source,
        "X_target": X_target,
        "surface_model": sm,
        "K": K,
        "horizon": horizon,
    }

# Main

def main():
    import time

    print("  Conditional Brenier Estimator — Volatility surface")
    print(f"  CPU cores: {N_CORES}  |  físicos: {N_PHYSICAL}")
    

    SURFACE_HISTORY_PATH = r"C:\volatility-options\data\surface_history.npz"
    SM_PATH = r"C:\volatility-options\data\surface_model.npz"
    NPZ_PATH = r"C:\volatility-options\data\live_spx_data_extended.npz"
    BRENIER_PATH = r"C:\volatility-options\data\brenier_surface.pkl"

    bundle = load_surface_brenier_data(
        surface_history_path = SURFACE_HISTORY_PATH,
        surface_model_path = SM_PATH,
        npz_path = NPZ_PATH,
        horizon = 5,
        filter_covid = True,   
        covid_start = "2020-02-15",
        covid_end = "2021-01-01",
    )
    X_source = bundle["X_source"]
    X_target = bundle["X_target"]
    surface_model = bundle["surface_model"]
    K = bundle["K"]
    N = len(X_source)

    split = int(0.8 * N)
    X_src_train = X_source[:split]
    X_tgt_train = X_target[:split]
    X_src_test = X_source[split:]
    X_tgt_test = X_target[split:]
    print(f"  Train: {len(X_src_train):,}  Test: {len(X_src_test):,}")

    print("\n Grid search: t (rescaling) ")
    cal_start  = int(0.60 * split)
    X_src_val  = X_src_train[cal_start:]
    X_tgt_val  = X_tgt_train[cal_start:]
    n_train_gs = cal_start

    t_grid = [
        0.010 * (n_train_gs ** (-1/3)),
        0.025 * (n_train_gs ** (-1/3)),
        0.050 * (n_train_gs ** (-1/3)),   
        0.075 * (n_train_gs ** (-1/3)),  
        0.100 * (n_train_gs ** (-1/3)),
        0.050 * (n_train_gs ** (-1/5)),
        0.100 * (n_train_gs ** (-1/5)),
        0.150 * (n_train_gs ** (-1/5)),
        0.200 * (n_train_gs ** (-1/5)),
    ]

    def _eps_adaptive(t):
        return max(t ** 2, t / 3.0)

    pc_stds_val = X_tgt_val.std(axis=0) + 1e-8

    best_t, best_mae, best_eps = None, np.inf, None
    print(f"  {'t':>10}  {'ε':>10}  {'MAE_wh (val)':>14}")
    for t_cand in t_grid:
        eps_cand = _eps_adaptive(t_cand)
        gs_est   = ConditionalBrenierEstimator(
            d1=K, d2=K, t=t_cand, epsilon=eps_cand,
            girsanov_alpha=1.0, n_jobs=-1,
        )
        gs_est.fit(X_src_train[:cal_start], X_tgt_train[:cal_start])
        pv = gs_est.predict(X_src_val)
        if pv.ndim == 1:
            pv = pv[np.newaxis, :]

        mae_wh = np.abs((pv - X_tgt_val) / pc_stds_val).mean()
        flag   = "  ←" if mae_wh < best_mae else ""
        print(f"  {t_cand:10.5f}  {eps_cand:10.6f}  {mae_wh:14.6f}{flag}")
        if mae_wh < best_mae:
            best_mae, best_t, best_eps = mae_wh, t_cand, eps_cand
    print(f"  → melhor t={best_t:.5f}  ε={best_eps:.6f}  MAE_wh={best_mae:.6f}")

    print("\nFit final")
    est = ConditionalBrenierEstimator(
        d1=K, d2=K, t=best_t, epsilon=best_eps,
        girsanov_alpha=1.0, n_jobs=-1,
    )
    print(f"  t={est.t:.4f}  ε={est.epsilon:.6f}")
    t0 = time.perf_counter()
    est.fit(X_src_train, X_tgt_train)
    print(f"{time.perf_counter() - t0:.1f}s")

    print("\nCalibração de bias")
    est.calibrate_bias_conditional(
        X_src_train[cal_start:],
        X_tgt_train[cal_start:],
    )

    print("\n Conformal prediction")
    cal_conf_start = int(0.80 * split)
    est.calibrate_conformal(
        X_src_train[cal_conf_start:],
        X_tgt_train[cal_conf_start:],
        coverage=0.90,
        n_mean_samples=50,
    )

    def _pad_and_reconstruct(scores_k: np.ndarray) -> np.ndarray:
        shape = scores_k.shape[:-1]
        flat = scores_k.reshape(-1, K).astype(np.float32)
        K_full = surface_model.pca.n_components
        pad = np.zeros((len(flat), K_full - K), dtype=np.float32)
        full = np.concatenate([flat, pad], axis=1)
        surfs = surface_model.pca.inverse_transform(full)
        return surfs.reshape(*shape, 4, 21) if shape else surfs[0]

    n_eval = min(200, len(X_src_test))
    idx = np.random.choice(len(X_src_test), n_eval, replace=False)
    X_eval = X_src_test[idx]
    X_true = X_tgt_test[idx]

    t0 = time.perf_counter()
    preds = est.predict(X_eval, n_mean_samples=50)   # [FIX-14] média de 50 amostras
    print(f"  Predict {n_eval} pontos: {time.perf_counter()-t0:.2f}s")

    errs_pc = np.abs(preds - X_true).mean(axis=1)
    print(f"\n mean MAE:  {errs_pc.mean():.5f}")
    print(f"MAE p25/p75:             {np.percentile(errs_pc,25):.5f} / "
          f"{np.percentile(errs_pc,75):.5f}")

    pred_surfs = _pad_and_reconstruct(preds)
    true_surfs = _pad_and_reconstruct(X_true)
    errs_iv    = np.abs(pred_surfs - true_surfs).mean(axis=(1, 2))
    print(f"\nMAE médio (escala IV):   {errs_iv.mean():.6f}")
    print(f"MAE p25/p75 (IV): {np.percentile(errs_iv,25):.6f} / "
          f"{np.percentile(errs_iv,75):.6f}")
    print(f"MAE p90 (IV): {np.percentile(errs_iv,90):.6f}")

    print(f"\n MAE ATM")
    mats_label = ["0.25Y", "0.50Y", "0.75Y", "1.00Y"]
    ts_mae = np.abs(pred_surfs[:, :, 10] - true_surfs[:, :, 10]).mean(axis=0)
    ts_spread_pred = (pred_surfs[:, :, 10].max(axis=1) - pred_surfs[:, :, 10].min(axis=1)).mean()
    ts_spread_true = (true_surfs[:, :, 10].max(axis=1) - true_surfs[:, :, 10].min(axis=1)).mean()
    for i, lbl in enumerate(mats_label):
        print(f"  {lbl}  MAE_ATM={ts_mae[i]:.5f}")
    print(f"  Term structure span médio — pred: {ts_spread_pred:.4f}  true: {ts_spread_true:.4f}")
    if ts_spread_pred < 0.3 * ts_spread_true:
        print(f" Collapse detected: predicted span is <30% of true span. "
      f"Consider increasing the number of retained PCs or reviewing surface_history.")

    per_pc_mae = np.abs(preds - X_true).mean(axis=0)
    var_expl   = surface_model.pca.explained_variance_ratio_
    print(f"\n  {'PC':5s}  {'MAE':>8}  {'var_expl':>9}")
    for k in range(K):
        print(f"  PC{k+1:<3d}  {per_pc_mae[k]:8.5f}  {var_expl[k]*100:8.1f}%")

    pc1_test = X_eval[:, 0]
    atm_bias = (pred_surfs[:, 0, 10] - true_surfs[:, 0, 10])
    octis    = np.percentile(pc1_test, np.linspace(0, 100, 9))
    print(f"\n── Diagnóstico de bias por regime (PC1, 8 octis) ─")
    print(f"  {'Regime':14s}  {'N':>5}  {'bias_ATM':>10}  {'MAE_IV':>8}")
    labels_o = ["O1 (vol baixa)", "O2", "O3", "O4", "O5", "O6", "O7", "O8 (vol alta)"]
    for i, label in enumerate(labels_o):
        mask = (pc1_test >= octis[i]) & (pc1_test < octis[i + 1])
        if mask.sum() == 0:
            continue
        print(f"  {label:14s}  {mask.sum():5d}  "
              f"{atm_bias[mask].mean():+10.5f}  {errs_iv[mask].mean():8.5f}")
    print(f"  {'TOTAL':14s}  {len(atm_bias):5d}  "
          f"{atm_bias.mean():+10.5f}  {errs_iv.mean():8.5f}")

    half_n   = 4
    bias_low  = np.array([atm_bias[(pc1_test >= octis[i]) & (pc1_test < octis[i+1])].mean()
                           for i in range(half_n) if ((pc1_test >= octis[i]) & (pc1_test < octis[i+1])).sum() > 0])
    bias_high = np.array([atm_bias[(pc1_test >= octis[i]) & (pc1_test < octis[i+1])].mean()
                           for i in range(half_n, 8) if ((pc1_test >= octis[i]) & (pc1_test < octis[i+1])).sum() > 0])
    if len(bias_low) and len(bias_high):
        low_m, high_m = bias_low.mean(), bias_high.mean()
        antis = abs(low_m + high_m) < 0.5 * (abs(low_m) + abs(high_m) + 1e-9)
        if antis and (abs(low_m) > 0.03 or abs(high_m) > 0.03):
            print(f"Collapse detected: predicted span is <30% of true span. "
      f"Consider increasing the number of retained PCs or reviewing surface_history.")
            
    print(f"\n W₂ error (proxy)")
    est.compute_wasserstein_error(X_eval, X_true, verbose=True)

    x_demo = X_src_test[0]
    p_demo, samps = est.predict(x_demo, return_distribution=True, n_samples=1000)
    surf_samples = _pad_and_reconstruct(samps)
    true_surf = _pad_and_reconstruct(X_tgt_test[0])

    mats = [0.25, 0.50, 0.75, 1.00]
    print(f"  {'Mat':>5}  {'ATM_true':>9}  {'ATM_mean':>9}  {'ATM_std':>8}  "
          f"{'skew_true':>10}  {'skew_mean':>10}")
    for i, T in enumerate(mats):
        print(f"  {T:.2f}Y  {true_surf[i,10]:9.4f}  "
              f"{surf_samples[:,i,10].mean():9.4f}  "
              f"{surf_samples[:,i,10].std():8.4f}  "
              f"{true_surf[i,0]-true_surf[i,20]:10.4f}  "
              f"{(surf_samples[:,i,0]-surf_samples[:,i,20]).mean():10.4f}")

    est.save(BRENIER_PATH)


if __name__ == "__main__":
    main()
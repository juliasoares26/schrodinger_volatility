import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

MONEYNESS_GRID = np.linspace(0.80, 1.20, 21)
MATURITY_GRID = np.array([0.25, 0.50, 0.75, 1.00])

N_T = len(MATURITY_GRID)
N_K = len(MONEYNESS_GRID)
N_FLAT = N_T * N_K


def _reconstruct_surface_history(
    X1: np.ndarray,
    feature_names: list,
    anchor_surface: np.ndarray,
    anchor_idx: int = -1,
) -> np.ndarray:
    names = list(feature_names)

    idx = {n: names.index(n) for n in
           ['log_vix', 'vix_term', 'skew_proxy', 'vol_regime', 'rv21']
           if n in names}

    N = len(X1)

    a = anchor_idx % N
    lv_anchor = X1[a, idx['log_vix']] if 'log_vix' in idx else 0.0
    vt_anchor = X1[a, idx['vix_term']] if 'vix_term' in idx else 0.0
    sp_anchor = X1[a, idx['skew_proxy']] if 'skew_proxy' in idx else 0.0
    vr_anchor = X1[a, idx['vol_regime']] if 'vol_regime' in idx else 0.0

    atm_anchor = anchor_surface[:, N_K // 2]

    log_m = np.log(MONEYNESS_GRID)
    sqrt_T = np.sqrt(MATURITY_GRID)

    surfaces = np.empty((N, N_T, N_K), dtype=np.float32)

    for t in range(N):
        lv = X1[t, idx['log_vix']] if 'log_vix' in idx else lv_anchor
        vt = X1[t, idx['vix_term']] if 'vix_term' in idx else vt_anchor
        sp = X1[t, idx['skew_proxy']] if 'skew_proxy' in idx else sp_anchor
        vr = X1[t, idx['vol_regime']] if 'vol_regime' in idx else vr_anchor

        delta_lv = lv - lv_anchor
        delta_vt = vt - vt_anchor
        delta_sp = sp - sp_anchor

        level_factor = np.exp(delta_lv * 0.4)

        term_deform = delta_vt * 0.03 * sqrt_T

        skew_deform = -delta_sp * 0.06 * log_m

        surf = anchor_surface * level_factor
        surf = surf + term_deform[:, np.newaxis]
        surf = surf + skew_deform[np.newaxis, :]
        surf = np.clip(surf, 0.02, 2.0)

        surfaces[t] = surf.astype(np.float32)

    print(f"  Reconstructed history: {N} surfaces  shape={surfaces.shape}")
    print(f"  IV range: [{surfaces.min():.4f}, {surfaces.max():.4f}]")
    return surfaces


class SurfacePCA:
    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, surfaces: np.ndarray) -> "SurfacePCA":
        N = len(surfaces)
        X = np.log(surfaces.reshape(N, N_FLAT).astype(np.float64) + 1e-8)

        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        K = min(self.n_components, len(S))

        self.components_ = Vt[:K]
        total_var = (S ** 2).sum()
        self.explained_variance_ratio_ = (S[:K] ** 2) / total_var
        self.singular_values_ = S[:K]

        cum = np.cumsum(self.explained_variance_ratio_)
        print(f"\n  PCA: {K} components explain {cum[-1]*100:.1f}% of variance")
        for k in range(K):
            print(f"    PC{k+1}: {self.explained_variance_ratio_[k]*100:.1f}%  "
                  f"(cumul {cum[k]*100:.1f}%)")
        return self

    def transform(self, surfaces: np.ndarray) -> np.ndarray:
        N = len(surfaces)
        X = np.log(surfaces.reshape(N, N_FLAT).astype(np.float64) + 1e-8)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        X_rec = scores @ self.components_ + self.mean_
        surfaces = np.exp(X_rec).reshape(-1, N_T, N_K)
        return np.clip(surfaces, 0.02, 2.0).astype(np.float32)

    def state(self) -> dict:
        return {
            "n_components": self.n_components,
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "singular_values_": self.singular_values_,
        }

    def load(self, d: dict):
        self.n_components = int(d["n_components"])
        self.mean_ = d["mean_"]
        self.components_ = d["components_"]
        self.explained_variance_ratio_ = d["explained_variance_ratio_"]
        self.singular_values_ = d.get("singular_values_", None)


class HARModel:
    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.coef_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None

    def _build_features(self, pc_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(pc_series)
        min_t = 21

        rows_X, rows_y = [], []
        for t in range(min_t, T - self.horizon):
            pc_d = pc_series[t]
            pc_5 = pc_series[t - 4: t + 1].mean()
            pc_21 = pc_series[t - 20: t + 1].mean()
            y_t = pc_series[t + self.horizon]
            rows_X.append([1.0, pc_d, pc_5, pc_21])
            rows_y.append(y_t)

        return np.array(rows_X), np.array(rows_y)

    def fit(self, pc_series: np.ndarray) -> "HARModel":
        X, y = self._build_features(pc_series)
        lam = 1e-6 * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(X.T @ X + lam, X.T @ y)
        y_hat = X @ self.coef_
        self.residuals_ = y - y_hat
        r2 = 1 - np.var(self.residuals_) / (np.var(y) + 1e-10)
        return self, r2

    def predict_one(self, pc_series: np.ndarray) -> float:
        t = len(pc_series) - 1
        pc_d = pc_series[t]
        pc_5 = pc_series[max(0, t - 4): t + 1].mean()
        pc_21 = pc_series[max(0, t - 20): t + 1].mean()
        x = np.array([1.0, pc_d, pc_5, pc_21])
        return float(x @ self.coef_)

    def state(self) -> dict:
        return {"coef_": self.coef_, "residuals_": self.residuals_, "horizon": self.horizon}

    def load(self, d: dict):
        self.coef_ = d["coef_"]
        self.residuals_ = d["residuals_"]
        self.horizon = int(d["horizon"])


def _load_or_reconstruct_surfaces(
    npz_path: str,
    surface_history_path: str | None,
    X1: np.ndarray,
    feature_names: list,
    anchor_surface: np.ndarray,
) -> np.ndarray:
    N = len(X1)

    if surface_history_path is not None and Path(surface_history_path).exists():
        print(f"\n  Loading real surfaces from: {surface_history_path}")
        hist = np.load(surface_history_path, allow_pickle=True)
        surf_real = hist["surfaces"].astype(np.float32)
        surf_dates = [str(d) for d in hist["dates"]]
        M = len(surf_real)

        print(f"  surface_history: {M} days  ({surf_dates[0]} -> {surf_dates[-1]})")
        print(f"  Real ATM IV 0.25Y: mean={surf_real[:,0,10].mean():.4f}  "
              f"range=[{surf_real[:,0,10].min():.4f}, {surf_real[:,0,10].max():.4f}]")

        npz_data = np.load(npz_path, allow_pickle=True)
        if "fetch_date" in npz_data:
            npz_dates = [str(d) for d in np.array(npz_data["fetch_date"]).flatten()]
            surf_date_set = {d: i for i, d in enumerate(surf_dates)}
            idx_surf, idx_npz = [], []
            for j, d in enumerate(npz_dates):
                if d in surf_date_set:
                    idx_surf.append(surf_date_set[d])
                    idx_npz.append(j)

            if len(idx_surf) >= max(100, N // 4):
                print(f"  Date alignment: {len(idx_surf)} common days")
                surfaces_aligned = surf_real[idx_surf]
                n_common = len(idx_surf)
                if n_common != N:
                    print(f"  Warning: aligned N ({n_common}) != X1 N ({N}) -- using intersection")
                return surfaces_aligned
            else:
                print(f"  Warning: few common days ({len(idx_surf)}) -- using last {N} days of history")

        if M >= N:
            print(f"  Using last {N} of {M} days from history")
            return surf_real[-N:]
        else:
            print(f"  Warning: real history has only {M} days (X1 has {N}) -- "
                  f"filling start with synthetic reconstruction")
            n_synth = N - M
            synth = _reconstruct_surface_history(
                X1[:n_synth], feature_names, anchor_surface, anchor_idx=-1
            )
            return np.concatenate([synth, surf_real], axis=0)

    print(f"\n  Warning: surface_history_path not provided or not found.")
    print(f"    Using synthetic reconstruction (ATM will be inaccurate).")
    print(f"    Run: python historical_surface_builder.py --out <path>")
    return _reconstruct_surface_history(X1, feature_names, anchor_surface, anchor_idx=-1)


class SurfaceModel:
    def __init__(self, n_components: int = 6, horizon: int = 5):
        self.n_components = n_components
        self.horizon = horizon
        self.pca = SurfacePCA(n_components)
        self.har_models: list[HARModel] = []
        self.pc_history: Optional[np.ndarray] = None
        self._feature_names: list = []
        self._anchor_surface: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, npz_path: str, surface_history_path: str | None = None) -> "SurfaceModel":
        print("\n" + "=" * 52)
        print("  SurfaceModel -- PCA + HAR fit")
        print("=" * 52)

        data = np.load(npz_path, allow_pickle=True)

        X1 = data["full_X1"].astype(np.float32)
        feature_names = [str(n) for n in data["feature_names"]]
        anchor_surface = data["vol_surface"].astype(np.float64)
        horizon = int(data.get("full_horizon", self.horizon))
        self.horizon = horizon

        self._feature_names = feature_names
        self._anchor_surface = anchor_surface

        print(f"\nData: N={len(X1)}  features={len(feature_names)}  horizon={horizon}d")
        print(f"Anchor: ATM_0.25Y={anchor_surface[0,10]:.4f}  ATM_1Y={anchor_surface[3,10]:.4f}")

        surfaces = _load_or_reconstruct_surfaces(
            npz_path=npz_path,
            surface_history_path=surface_history_path,
            X1=X1,
            feature_names=feature_names,
            anchor_surface=anchor_surface,
        )

        self.pca.fit(surfaces)
        scores = self.pca.transform(surfaces)
        self.pc_history = scores

        print(f"\n  HAR fit per component (horizon={horizon}d):")
        self.har_models = []
        for k in range(self.n_components):
            har = HARModel(horizon=horizon)
            har, r2 = har.fit(scores[:, k])
            self.har_models.append(har)
            print(f"    PC{k+1}: R2={r2:.4f}  coef={har.coef_}")

        self._fitted = True
        print("\n  SurfaceModel trained.")
        return self

    def predict(self, X1_row: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call .fit() first."

        pred_scores = np.array([
            har.predict_one(self.pc_history[:, k])
            for k, har in enumerate(self.har_models)
        ], dtype=np.float64)

        surface = self.pca.inverse_transform(pred_scores[np.newaxis])[0]
        return surface

    def predict_distribution(
        self,
        X1_row: np.ndarray,
        n: int = 500,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        assert self._fitted, "Call .fit() first."

        rng = np.random.default_rng(seed)

        point_scores = np.array([
            har.predict_one(self.pc_history[:, k])
            for k, har in enumerate(self.har_models)
        ])

        residual_samples = np.column_stack([
            rng.choice(har.residuals_, size=n, replace=True)
            for har in self.har_models
        ])

        sampled_scores = point_scores[np.newaxis, :] + residual_samples

        surfaces = self.pca.inverse_transform(sampled_scores)
        return surfaces

    def evaluate(self, test_fraction: float = 0.2) -> dict:
        assert self._fitted

        N = len(self.pc_history)
        n_test = max(21 + self.horizon + 1, int(N * test_fraction))
        split = N - n_test

        errors = []
        for t in range(split, N - self.horizon):
            pred_scores = np.array([
                har.predict_one(self.pc_history[:t, k])
                for k, har in enumerate(self.har_models)
            ])
            pred_surf = self.pca.inverse_transform(pred_scores[np.newaxis])[0]

            true_scores = self.pc_history[t + self.horizon]
            true_surf = self.pca.inverse_transform(true_scores[np.newaxis])[0]

            mae = np.abs(pred_surf - true_surf).mean()
            errors.append(mae)

        results = {
            "mae_mean": float(np.mean(errors)),
            "mae_p25": float(np.percentile(errors, 25)),
            "mae_p75": float(np.percentile(errors, 75)),
            "mae_p90": float(np.percentile(errors, 90)),
            "n_test": len(errors),
        }

        print(f"\n-- Out-of-sample evaluation --")
        print(f"  MAE mean (IV):  {results['mae_mean']:.6f}")
        print(f"  MAE p25/p75:    {results['mae_p25']:.6f} / {results['mae_p75']:.6f}")
        print(f"  MAE p90:        {results['mae_p90']:.6f}")
        print(f"  N test:         {results['n_test']}")
        return results

    def save(self, path: str):
        np.savez(path,
                 pca_state=np.array([self.pca.state()], dtype=object),
                 har_states=np.array([h.state() for h in self.har_models], dtype=object),
                 pc_history=self.pc_history,
                 anchor_surface=self._anchor_surface,
                 feature_names=np.array(self._feature_names),
                 n_components=np.array(self.n_components),
                 horizon=np.array(self.horizon),
                 fitted=np.array(True),
                 )
        print(f"SurfaceModel saved to: {path}")

    def load(self, path: str) -> "SurfaceModel":
        d = np.load(path, allow_pickle=True)
        self.n_components = int(d["n_components"])
        self.horizon = int(d["horizon"])
        self._feature_names = list(d["feature_names"])
        self._anchor_surface = d["anchor_surface"]
        self.pc_history = d["pc_history"]
        self._fitted = bool(d["fitted"])

        self.pca = SurfacePCA(self.n_components)
        self.pca.load(d["pca_state"][0])

        self.har_models = []
        for state in d["har_states"]:
            har = HARModel()
            har.load(state)
            self.har_models.append(har)

        print(f"SurfaceModel loaded from: {path}")
        return self


def main():
    import sys
    NPZ_PATH = r"C:\volatility-options\data\live_spx_data_extended.npz"
    SURF_HIST = r"C:\volatility-options\data\surface_history.npz"
    SAVE_PATH = "data/surface_model.npz"

    model = SurfaceModel(n_components=6, horizon=5)
    model.fit(NPZ_PATH, surface_history_path=SURF_HIST)

    results = model.evaluate(test_fraction=0.2)

    data = np.load(NPZ_PATH, allow_pickle=True)
    X1 = data["full_X1"]
    x_demo = X1[-1]

    surf_pred = model.predict(x_demo)
    print(f"\nPredicted surface (t+5d):")
    print(f"  ATM IVs by maturity: {surf_pred[:, 10]}")
    print(f"  Skew (0.80-1.20) by maturity: {surf_pred[:, 0] - surf_pred[:, 20]}")

    surfs_dist = model.predict_distribution(x_demo, n=500)
    print(f"\nDistribution (500 samples):")
    print(f"  ATM IV 0.25Y: mean={surfs_dist[:, 0, 10].mean():.4f}  "
          f"std={surfs_dist[:, 0, 10].std():.4f}")
    print(f"  ATM IV 1.00Y: mean={surfs_dist[:, 3, 10].mean():.4f}  "
          f"std={surfs_dist[:, 3, 10].std():.4f}")

    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    model.save(SAVE_PATH)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
import numpy as np
from scipy import stats
from scipy.special import logsumexp


def pointwise_metrics(
    scores_pred: np.ndarray,
    scores_true: np.ndarray,
    surfaces_pred: np.ndarray,
    surfaces_true: np.ndarray,
    var_expl: np.ndarray,
    pc1_vals: np.ndarray,
    maturity_grid: np.ndarray,
    moneyness_grid: np.ndarray,
) -> dict:
    N, K = scores_pred.shape
    atm_idx = len(moneyness_grid) // 2
    otm_idx = len(moneyness_grid) // 4

    abs_err_pc = np.abs(scores_pred - scores_true)
    mae_pc = float(abs_err_pc.mean())
    rmse_pc = float(np.sqrt(((scores_pred - scores_true) ** 2).mean()))
    mae_per_pc = abs_err_pc.mean(axis=0)

    abs_err_iv = np.abs(surfaces_pred - surfaces_true)
    mae_iv = float(abs_err_iv.mean())
    rmse_iv = float(np.sqrt(((surfaces_pred - surfaces_true) ** 2).mean()))
    per_sample_mae = abs_err_iv.reshape(N, -1).mean(axis=1)
    mae_iv_p25 = float(np.percentile(per_sample_mae, 25))
    mae_iv_p75 = float(np.percentile(per_sample_mae, 75))
    mae_iv_p90 = float(np.percentile(per_sample_mae, 90))

    atm_pred = surfaces_pred[:, :, atm_idx]
    atm_true = surfaces_true[:, :, atm_idx]
    mae_atm_per_mat = np.abs(atm_pred - atm_true).mean(axis=0)

    ts_span_pred = float((atm_pred.max(axis=1) - atm_pred.min(axis=1)).mean())
    ts_span_true = float((atm_true.max(axis=1) - atm_true.min(axis=1)).mean())

    atm_bias = atm_pred[:, 0] - atm_true[:, 0]
    quartis = np.percentile(pc1_vals, [0, 25, 50, 75, 100])
    regime_diag = []
    labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    for i, lbl in enumerate(labels):
        mask = (pc1_vals >= quartis[i]) & (pc1_vals < quartis[i + 1])
        if mask.sum() == 0:
            continue
        regime_diag.append({
            "label": lbl,
            "n": int(mask.sum()),
            "bias_atm": float(atm_bias[mask].mean()),
            "std_bias": float(atm_bias[mask].std()),
            "mae_iv": float(per_sample_mae[mask].mean()),
        })

    skew_pred = surfaces_pred[:, 0, otm_idx] - surfaces_pred[:, 0, atm_idx]
    skew_true = surfaces_true[:, 0, otm_idx] - surfaces_true[:, 0, atm_idx]
    skew_mae = float(np.abs(skew_pred - skew_true).mean())
    skew_bias = float((skew_pred - skew_true).mean())

    return {
        "mae_pc": mae_pc,
        "rmse_pc": rmse_pc,
        "mae_per_pc": mae_per_pc.tolist(),
        "var_expl": var_expl[:K].tolist(),
        "mae_iv": mae_iv,
        "rmse_iv": rmse_iv,
        "mae_iv_p25": mae_iv_p25,
        "mae_iv_p75": mae_iv_p75,
        "mae_iv_p90": mae_iv_p90,
        "mae_atm_per_mat": mae_atm_per_mat.tolist(),
        "maturity_grid": maturity_grid.tolist(),
        "ts_span_pred": ts_span_pred,
        "ts_span_true": ts_span_true,
        "ts_span_ratio": float(ts_span_pred / (ts_span_true + 1e-8)),
        "atm_bias_mean": float(atm_bias.mean()),
        "atm_bias_std": float(atm_bias.std()),
        "regime_diag": regime_diag,
        "skew_mae": skew_mae,
        "skew_bias": skew_bias,
        "skew_sign_ok": bool(np.sign(skew_pred.mean()) == np.sign(skew_true.mean())),
    }


def distributional_metrics(
    samples_pred: np.ndarray,
    scores_true: np.ndarray,
    surfaces_samp: np.ndarray,
    surfaces_true: np.ndarray,
    moneyness_grid: np.ndarray,
    confidence: float = 0.90,
) -> dict:
    N, n_samp, K = samples_pred.shape
    atm_idx = len(moneyness_grid) // 2

    alpha = (1 - confidence) / 2
    lo = np.percentile(samples_pred, alpha * 100, axis=1)
    hi = np.percentile(samples_pred, (1 - alpha) * 100, axis=1)
    in_band = (scores_true >= lo) & (scores_true <= hi)
    coverage_per_pc = in_band.mean(axis=0)
    coverage_mean = float(in_band.mean())

    sharpness_per_pc = samples_pred.std(axis=1).mean(axis=0)

    crps_per_pc = []
    for k in range(K):
        samp_k = samples_pred[:, :, k]
        true_k = scores_true[:, k]
        term1 = np.abs(samp_k - true_k[:, None]).mean(axis=1)
        n_sub = min(n_samp, 50)
        sub = samp_k[:, :n_sub]
        term2 = np.abs(sub[:, :, None] - sub[:, None, :]).mean(axis=(1, 2))
        crps_per_pc.append(float((term1 - 0.5 * term2).mean()))

    w1_per_pc = []
    for k in range(K):
        w1_vals = []
        for i in range(N):
            w1_vals.append(_wasserstein1_samples_vs_point(
                samples_pred[i, :, k], scores_true[i, k]
            ))
        w1_per_pc.append(float(np.mean(w1_vals)))

    coverage_atm_iv = None
    if surfaces_samp is not None:
        atm_samp = surfaces_samp[:, :, 0, atm_idx]
        atm_true = surfaces_true[:, 0, atm_idx]
        lo_iv = np.percentile(atm_samp, alpha * 100, axis=1)
        hi_iv = np.percentile(atm_samp, (1 - alpha) * 100, axis=1)
        coverage_atm_iv = float(((atm_true >= lo_iv) & (atm_true <= hi_iv)).mean())

    return {
        "confidence": confidence,
        "coverage_mean": coverage_mean,
        "coverage_per_pc": coverage_per_pc.tolist(),
        "sharpness_per_pc": sharpness_per_pc.tolist(),
        "crps_per_pc": crps_per_pc,
        "crps_mean": float(np.mean(crps_per_pc)),
        "w1_per_pc": w1_per_pc,
        "w1_mean": float(np.mean(w1_per_pc)),
        "coverage_atm_iv": coverage_atm_iv,
    }


def _wasserstein1_samples_vs_point(samples: np.ndarray, true_val: float) -> float:
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    cdf_s = np.arange(1, n + 1) / n
    all_pts = np.sort(np.append(sorted_s, true_val))
    dist = 0.0
    for i in range(len(all_pts) - 1):
        mid = (all_pts[i] + all_pts[i + 1]) / 2.0
        F_s = np.interp(mid, sorted_s, cdf_s, left=0.0, right=1.0)
        F_t = 0.0 if mid < true_val else 1.0
        dist += abs(F_s - F_t) * (all_pts[i + 1] - all_pts[i])
    return float(dist)


def surface_diagnostics(
    surfaces: np.ndarray,
    moneyness_grid: np.ndarray,
    maturity_grid: np.ndarray,
) -> dict:
    N = len(surfaces)
    atm = len(moneyness_grid) // 2

    n_spread_viol = 0
    n_convex_viol = 0
    n_skew_wrong = 0
    n_ts_inverted = 0
    n_iv_oor = 0
    skews = []
    ts_spans = []

    for surf in surfaces:
        for p in range(len(maturity_grid)):
            iv_row = surf[p]
            d2 = np.diff(iv_row, n=2)
            if np.any(d2 < -1e-3):
                n_convex_viol += 1
                break

        skew_025 = surf[0, len(moneyness_grid) // 4] - surf[0, atm]
        skews.append(float(skew_025))
        if skew_025 < 0:
            n_skew_wrong += 1

        atm_curve = surf[:, atm]
        ts_span = float(atm_curve[-1] - atm_curve[0])
        ts_spans.append(ts_span)
        if ts_span < -0.05:
            n_ts_inverted += 1

        if surf.min() < 0.03 or surf.max() > 1.50:
            n_iv_oor += 1

    return {
        "n_surfaces": N,
        "pct_convex_ok": float(1 - n_convex_viol / N),
        "pct_skew_correct": float(1 - n_skew_wrong / N),
        "mean_skew_025Y": float(np.mean(skews)),
        "std_skew_025Y": float(np.std(skews)),
        "pct_ts_normal": float(1 - n_ts_inverted / N),
        "mean_ts_span": float(np.mean(ts_spans)),
        "pct_iv_in_range": float(1 - n_iv_oor / N),
    }


def check_call_spread_arbitrage(prices: np.ndarray, strikes: np.ndarray) -> dict:
    violations = []
    for i in range(len(prices) - 1):
        spread = prices[i] - prices[i + 1]
        if spread < -1e-6:
            violations.append({
                "strike_low": float(strikes[i]),
                "strike_high": float(strikes[i + 1]),
                "spread": float(spread),
            })
    return {
        "n_violations": len(violations),
        "violations": violations,
        "is_arbitrage_free": len(violations) == 0,
    }


def check_convexity(prices: np.ndarray, strikes: np.ndarray) -> dict:
    if len(prices) < 3:
        return {"is_convex": True, "n_violations": 0}
    d2C = np.diff(prices, n=2)
    dk2 = np.diff(strikes, n=2)
    d2C_dk2 = d2C / (dk2 + 1e-10)
    viol = int(np.sum(d2C_dk2 < -1e-6))
    return {
        "is_convex": bool(viol == 0),
        "n_violations": viol,
        "min_curvature": float(np.min(d2C_dk2)),
        "mean_curvature": float(np.mean(d2C_dk2)),
    }


def compare_models(
    results: dict[str, dict],
    maturity_grid: np.ndarray,
) -> str:
    header = (
        f"\n{'='*78}\n"
        f"{'Model':<12} {'MAE_IV':>9} {'RMSE_IV':>9} {'MAE_PC':>8} "
        f"{'Bias_ATM':>10} {'Skew_OK':>8} {'TS_ratio':>9}\n"
        f"{'-'*78}"
    )
    lines = [header]

    for name, r in results.items():
        skew_ok = "ok" if r.get("skew_sign_ok", False) else "fail"
        lines.append(
            f"{name:<12} "
            f"{r['mae_iv']:>9.5f} "
            f"{r['rmse_iv']:>9.5f} "
            f"{r['mae_pc']:>8.5f} "
            f"{r['atm_bias_mean']:>+10.5f} "
            f"{skew_ok:>8} "
            f"{r['ts_span_ratio']:>9.3f}"
        )

    lines.append(f"{'='*78}")

    lines.append(f"\n-- MAE ATM by maturity --")
    mat_header = f"  {'Model':<12}" + "".join(
        f"  {t:.2f}Y" for t in maturity_grid
    )
    lines.append(mat_header)
    for name, r in results.items():
        row = f"  {name:<12}"
        for mae in r.get("mae_atm_per_mat", []):
            row += f"  {mae:.4f}"
        lines.append(row)

    lines.append(f"\n-- ATM bias by PC1 regime --")
    for name, r in results.items():
        lines.append(f"  {name}:")
        for reg in r.get("regime_diag", []):
            lines.append(
                f"    {reg['label']:12s}  N={reg['n']:3d}  "
                f"bias={reg['bias_atm']:+.4f}  MAE_IV={reg['mae_iv']:.4f}"
            )

    return "\n".join(lines)


def ensemble_weights_from_mae(
    results: dict[str, dict],
    temperature: float = 2.0,
) -> dict[str, float]:
    names = list(results.keys())
    maes = np.array([results[n]["mae_iv"] for n in names])
    scores = -maes / temperature
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    return {n: float(w) for n, w in zip(names, weights)}


def reconstruct_surfaces(
    scores_k: np.ndarray,
    pc_stds_k: np.ndarray,
    surface_model,
    maturity_grid: np.ndarray,
    moneyness_grid: np.ndarray,
) -> np.ndarray:
    shape = scores_k.shape[:-1]
    K = scores_k.shape[-1]
    flat = scores_k.reshape(-1, K).astype(np.float32)
    flat_r = flat * pc_stds_k
    K_full = surface_model.pca.n_components
    pad = np.zeros((len(flat_r), K_full - K), dtype=np.float32)
    full = np.concatenate([flat_r, pad], axis=1)
    surfs = surface_model.pca.inverse_transform(full)
    P, M = len(maturity_grid), len(moneyness_grid)
    return surfs.reshape(*shape, P, M) if shape else surfs[0].reshape(P, M)


def effective_sample_size(weights: np.ndarray) -> float:
    w = weights / (np.sum(weights) + 1e-10)
    return float(1.0 / (np.sum(w ** 2) + 1e-10))


def convergence_diagnostics(errors: np.ndarray, sample_sizes: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=float)
    sample_sizes = np.asarray(sample_sizes, dtype=float)
    valid = errors > 0
    if valid.sum() < 2:
        return {"rate": np.nan, "r_squared": np.nan}
    slope, intercept, r, p, se = stats.linregress(
        np.log(sample_sizes[valid]), np.log(errors[valid])
    )
    return {
        "rate": float(slope),
        "r_squared": float(r ** 2),
        "p_value": float(p),
        "std_err": float(se),
        "is_monte_carlo_rate": bool(abs(slope - (-0.5)) < 0.1),
        "is_conditional_brenier_rate": bool(abs(slope - (-2 / 3)) < 0.1),
        "fit_params": (float(intercept), float(slope)),
    }


def generate_report(
    model_name: str,
    pw: dict,
    dist: dict | None,
    surf_diag: dict | None,
) -> str:
    lines = [
        f"\n{'='*60}",
        f"  {model_name} -- Full Evaluation",
        f"{'='*60}",
        f"\n-- Pointwise metrics --",
        f"MAE mean (PCA space):  {pw['mae_pc']:.5f}",
        f"MAE mean (IV scale):   {pw['mae_iv']:.6f}",
        f"MAE p25/p75 (IV):      {pw['mae_iv_p25']:.6f} / {pw['mae_iv_p75']:.6f}",
        f"MAE p90 (IV):          {pw['mae_iv_p90']:.6f}",
        f"\n-- MAE by PCA component --",
        f"  {'PC':5s}  {'MAE':>8}  {'var_expl':>9}",
    ]
    for k, (mae, ve) in enumerate(zip(pw["mae_per_pc"], pw["var_expl"])):
        lines.append(f"  PC{k+1:<3d}  {mae:8.5f}  {ve*100:8.1f}%")

    lines += [f"\n-- MAE ATM by maturity --"]
    for t, mae in zip(pw["maturity_grid"], pw["mae_atm_per_mat"]):
        lines.append(f"  {t:.2f}Y  MAE_ATM={mae:.5f}")
    lines.append(
        f"  Term structure span -- pred: {pw['ts_span_pred']:.4f}  "
        f"true: {pw['ts_span_true']:.4f}  ratio: {pw['ts_span_ratio']:.3f}"
    )

    lines += [
        f"\n-- Skew (OTM_put - ATM, 0.25Y) --",
        f"  MAE skew:   {pw['skew_mae']:.5f}",
        f"  Bias skew:  {pw['skew_bias']:+.5f}",
        f"  Sign ok:    {'yes' if pw['skew_sign_ok'] else 'no  <- inverted skew'}",
        f"\n-- ATM bias by regime (PC1) --",
        f"  {'Regime':12s}  {'N':>5}  {'bias_ATM':>10}  {'bias_std':>10}  {'MAE_IV':>8}",
    ]
    for r in pw["regime_diag"]:
        lines.append(
            f"  {r['label']:12s}  {r['n']:5d}  "
            f"{r['bias_atm']:+10.5f}  {r['std_bias']:10.5f}  {r['mae_iv']:8.5f}"
        )
    lines.append(
        f"  {'TOTAL':12s}  {'':5s}  {pw['atm_bias_mean']:+10.5f}  "
        f"{pw['atm_bias_std']:10.5f}  {pw['mae_iv']:8.5f}"
    )

    if dist is not None:
        lines += [
            f"\n-- Distributional metrics ({dist['confidence']*100:.0f}% CI) --",
            f"  Mean coverage:      {dist['coverage_mean']:.3f}  "
            f"(target: {dist['confidence']:.2f})",
            f"  Coverage per PC:    {', '.join(f'{c:.3f}' for c in dist['coverage_per_pc'])}",
            f"  Mean CRPS:          {dist['crps_mean']:.5f}",
            f"  Mean W1 (per PC):   {dist['w1_mean']:.5f}",
        ]
        if dist.get("coverage_atm_iv") is not None:
            lines.append(f"  ATM IV coverage:    {dist['coverage_atm_iv']:.3f}")

    if surf_diag is not None:
        lines += [
            f"\n-- Surface diagnostics ({surf_diag['n_surfaces']} samples) --",
            f"  Convexity ok:       {surf_diag['pct_convex_ok']*100:.1f}%",
            f"  Correct skew:       {surf_diag['pct_skew_correct']*100:.1f}%  "
            f"(mean={surf_diag['mean_skew_025Y']:+.4f})",
            f"  Normal TS:          {surf_diag['pct_ts_normal']*100:.1f}%  "
            f"(span={surf_diag['mean_ts_span']:+.4f})",
            f"  IV in range [3%,150%]: {surf_diag['pct_iv_in_range']*100:.1f}%",
        ]

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def main():
    import sys
    import importlib.util
    from pathlib import Path

    print("\n" + "=" * 60)
    print("  metrics.py -- Vol Surface Pipeline Evaluation")
    print("=" * 60)

    DATA_DIR = Path(r"C:\volatility-options\data")
    NPZ_PATH = str(DATA_DIR / "live_spx_data_extended.npz")
    SM_PATH = str(DATA_DIR / "surface_model.npz")
    BRIDGE_PATH = str(DATA_DIR / "bridge_surface.pt")
    HESTON_PATH = str(DATA_DIR / "heston_surface.pkl")

    N_EVAL = 200
    N_SAMP = 500
    HORIZON = 5
    SEED = 42
    rng = np.random.default_rng(SEED)

    _sm_file = Path(__file__).resolve().parent / "surface_model.py"
    if not _sm_file.exists():
        _sm_file = Path("surface_model.py")
    if not _sm_file.exists():
        print("ERROR: surface_model.py not found.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("surface_model", str(_sm_file))
    sm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm_mod)

    print("\n-- Loading data --")
    sm = sm_mod.SurfaceModel()
    sm.load(SM_PATH)
    print(f"  SurfaceModel loaded: {sm.pca.n_components} PCs")

    scores_full = sm.pc_history.astype(np.float32)

    from datetime import datetime
    _surf_hist = np.load(str(DATA_DIR / "surface_history.npz"), allow_pickle=False)
    _dates = [datetime.strptime(str(d), "%Y-%m-%d").date()
              for d in _surf_hist["dates"]]
    _start_d = datetime.strptime("2020-02-15", "%Y-%m-%d").date()
    _end_d = datetime.strptime("2021-01-01", "%Y-%m-%d").date()
    _n_align = min(len(_dates), len(scores_full))
    _dates_use = _dates[-_n_align:]
    _keep = np.array([not (_start_d <= d < _end_d) for d in _dates_use])
    scores_full = scores_full[-_n_align:][_keep]
    print(f"  COVID filter: {(~_keep).sum()} days removed. "
          f"Remaining series: {len(scores_full)} days.")

    var_expl = sm.pca.explained_variance_ratio_
    pc_stds = scores_full.std(axis=0)
    meaningful = max(int(np.sum(pc_stds > 0.01)), 2)
    scores = scores_full[:, :meaningful].copy()
    N_total, K = scores.shape
    pc_stds_k = np.maximum(scores.std(axis=0), 1e-6)
    scores = scores / pc_stds_k
    print(f"  PCs retained: {K}  whitening: {np.round(pc_stds_k, 4).tolist()}")

    X_source = scores[:N_total - HORIZON]
    X_target = scores[HORIZON:]

    split = int(0.8 * len(X_source))
    X_src_test = X_source[split:]
    X_tgt_test = X_target[split:]
    print(f"  N_test: {len(X_src_test):,}  K={K}  horizon={HORIZON}d")

    try:
        _hist = np.load(str(DATA_DIR / "surface_history.npz"), allow_pickle=False)
        moneyness_grid = _hist.get("moneyness_grid", np.linspace(0.80, 1.20, 21))
        maturity_grid = _hist.get("maturity_grid", np.array([0.25, 0.50, 0.75, 1.00]))
    except Exception:
        moneyness_grid = np.linspace(0.80, 1.20, 21)
        maturity_grid = np.array([0.25, 0.50, 0.75, 1.00])

    n_eval = min(N_EVAL, len(X_src_test))
    idx = rng.choice(len(X_src_test), n_eval, replace=False)
    X_eval = X_src_test[idx]
    X_true = X_tgt_test[idx]

    def _recon(scores_k: np.ndarray) -> np.ndarray:
        return reconstruct_surfaces(
            scores_k, pc_stds_k, sm, maturity_grid, moneyness_grid
        )

    surfs_true = _recon(X_true)

    results_pw = {}
    results_dist = {}
    results_diag = {}

    if Path(BRIDGE_PATH).exists():
        print("\n-- Bridge --")
        import torch

        _br_file = Path(__file__).resolve().parent / "bridge.py"
        if not _br_file.exists():
            _br_file = Path("bridge.py")
        if _br_file.exists():
            spec_br = importlib.util.spec_from_file_location("bridge", str(_br_file))
            br_mod = importlib.util.module_from_spec(spec_br)
            spec_br.loader.exec_module(br_mod)
            bridge = br_mod.ConditionalBrenierSinkhorn(
                d_in=K, d_out=K, d_noise=K*2, hidden=128, n_blocks=3
            )
            bridge.load(BRIDGE_PATH)
            bridge.net.eval()

            print(f"  Generating {N_SAMP} samples for {n_eval} points...")
            samps_pc = bridge.sample_batch(X_eval, n=N_SAMP)
            means_pc = samps_pc.mean(axis=1)
            surfs_pred = _recon(means_pc)
            surfs_samp = np.stack([_recon(samps_pc[:, i, :])
                                   for i in range(min(N_SAMP, 200))], axis=1)

            results_pw["bridge"] = pointwise_metrics(
                means_pc, X_true, surfs_pred, surfs_true,
                var_expl[:K], X_eval[:, 0], maturity_grid, moneyness_grid,
            )
            results_dist["bridge"] = distributional_metrics(
                samps_pc, X_true, surfs_samp, surfs_true, moneyness_grid,
            )
            demo_samps = bridge.sample(X_src_test[0], n=1000)
            demo_surfs = _recon(demo_samps)
            results_diag["bridge"] = surface_diagnostics(
                demo_surfs, moneyness_grid, maturity_grid
            )
            print(generate_report(
                "Bridge",
                results_pw["bridge"],
                results_dist["bridge"],
                results_diag["bridge"],
            ))
        else:
            print("  bridge.py not found -- skipping.")
    else:
        print(f"\n  Bridge checkpoint not found at {BRIDGE_PATH} -- skipping.")

    if Path(HESTON_PATH).exists():
        print("\n-- Heston --")
        _hs_file = Path(__file__).resolve().parent / "heston.py"
        if not _hs_file.exists():
            _hs_file = Path("heston.py")
        if _hs_file.exists():
            spec_hs = importlib.util.spec_from_file_location("heston", str(_hs_file))
            hs_mod = importlib.util.module_from_spec(spec_hs)
            spec_hs.loader.exec_module(hs_mod)

            cal = hs_mod.HestonCalibrator.load(HESTON_PATH, sm)
            cal._horizon = HORIZON

            print(f"  Generating predictions for {n_eval} points...")
            import time
            t0 = time.perf_counter()
            preds = cal.predict(X_eval)
            print(f"  Predict: {time.perf_counter()-t0:.1f}s")

            surfs_pred_hs = _recon(preds)

            n_dist = min(50, n_eval)
            samps_hs = np.stack(
                [cal.sample(X_eval[i], n=N_SAMP) for i in range(n_dist)],
                axis=0,
            )

            results_pw["heston"] = pointwise_metrics(
                preds, X_true, surfs_pred_hs, surfs_true,
                var_expl[:K], X_eval[:, 0], maturity_grid, moneyness_grid,
            )
            results_dist["heston"] = distributional_metrics(
                samps_hs, X_true[:n_dist], None, surfs_true[:n_dist], moneyness_grid,
            )
            demo_samps_hs = cal.sample(X_src_test[0], n=1000)
            demo_surfs_hs = _recon(demo_samps_hs)
            results_diag["heston"] = surface_diagnostics(
                demo_surfs_hs, moneyness_grid, maturity_grid
            )
            print(generate_report(
                "Heston",
                results_pw["heston"],
                results_dist["heston"],
                results_diag["heston"],
            ))
        else:
            print("  heston.py not found -- skipping.")
    else:
        print(f"\n  Heston checkpoint not found at {HESTON_PATH} -- skipping.")

    if len(results_pw) >= 2:
        print("\n-- Ensemble Comparison --")
        print(compare_models(results_pw, maturity_grid))

        weights = ensemble_weights_from_mae(results_pw)
        print("\n  Suggested weights (inverse softmax of MAE_IV):")
        for name, w in weights.items():
            print(f"    {name:<12}: {w:.3f}")

    elif len(results_pw) == 1:
        name = list(results_pw.keys())[0]
        print(f"\n  Only {name} evaluated -- ensemble comparison requires >= 2 models.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
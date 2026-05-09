import numpy as np
import sys
from pathlib import Path

# calibration/base.py has CalibrationMethod and CalibrationResult
sys.path.insert(0, str(Path(__file__).parent))
# models/ has bridge.py and heston.py
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from base import CalibrationMethod, CalibrationResult, HestonUnified
from bridge import ConditionalBrenierSinkhorn


class MSBCalibration(CalibrationMethod):
    """
    Martingale Schrödinger Bridge calibration via ConditionalBrenierSinkhorn

    Parameters
    heston_model: HestonUnified instance (naked model P₀)
    n_paths: MC paths for training (default 4000)
    n_iter: training iterations (default 2000)
    batch: mini-batch size (default 256)
    lr: learning rate (default 3e-4)
    n_paths_price: MC paths for price_calls() (default 5000)
    seed: random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        heston_model:  HestonUnified,
        n_paths: int = 4000,
        n_iter: int = 2000,
        batch: int = 256,
        lr: float = 3e-4,
        n_paths_price: int = 5000,
        seed: int   = 42,
    ):
        self._heston = heston_model
        self._n_paths = n_paths
        self._n_iter = n_iter
        self._batch = batch
        self._lr = lr
        self._n_paths_price = n_paths_price
        self._seed = seed
        self._bridge = None
        self._strikes = None
        self._T = None
        self._df = None

    # Identity

    @property
    def name(self) -> str:
        return "MSB"

    # Internal helpers
   
    def _simulate_payoffs(self, strikes: np.ndarray, T: float,
                          n_paths: int, seed: int) -> np.ndarray:
        """Returns discounted call payoff matrix (n_paths, n_strikes)."""
        S_T, _, _ = self._heston.simulate_paths(
            T=T, n_paths=n_paths, seed=seed, full_paths=False
        )
        # S_T is (n_paths,) terminal stock prices
        print(f"    [_simulate_payoffs] seed={seed}  S_T: "
              f"mean={S_T.mean():.2f}  std={S_T.std():.2f}  "
              f"min={S_T.min():.2f}  max={S_T.max():.2f}")
        df      = np.exp(-self._heston.r * T)
        payoffs = df * np.maximum(S_T[:, None] - strikes[None, :], 0)
        print(f"    [_simulate_payoffs] payoffs: "
              f"mean={payoffs.mean():.4f}  max={payoffs.max():.4f}")
        return payoffs  # (n_paths, n_strikes)

    def _importance_weights(self, payoffs: np.ndarray,
                            market_prices: np.ndarray) -> np.ndarray:
        """
        Schrödinger dual: find omega* s.t. E_w[payoffs] ≈ market_prices.
        Uses the same L-BFGS-B dual optimisation as HestonUnified.
        """
        from scipy.optimize import minimize

        n_mc = payoffs.shape[0]
        phi  = payoffs.T  # (n_strikes, n_mc)

        def _obj(omega):
            score = omega @ phi
            s_max = score.max()
            e     = np.exp(score - s_max)
            Z     = e.mean()
            logZ  = np.log(Z) + s_max
            E_phi = (phi @ e) / (Z * n_mc)
            reg   = 0.5 * 1e-4 * float(omega @ omega)
            return -float(omega @ market_prices) + logZ + reg, \
                   -market_prices + E_phi + 1e-4 * omega

        result = minimize(
            _obj, np.zeros(len(market_prices)),
            method='L-BFGS-B', jac=True,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        omega_star = result.x
        score      = omega_star @ phi
        s_max      = score.max()
        w          = np.exp(score - s_max)
        w         /= w.sum()
        return w  # (n_mc,) normalised IS weights

    # CalibrationMethod interface
  
    def calibrate(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        T:             float,
        **kwargs,
    ) -> CalibrationResult:
        self._strikes = strikes
        self._T = T
        self._df = np.exp(-self._heston.r * T)
        n_strikes = len(strikes)

        # Step 1: simulate payoffs under P₀
        print(f"  [MSB] Simulating {self._n_paths} paths …")
        payoffs = self._simulate_payoffs(strikes, T, self._n_paths, self._seed)
        # payoffs: (n_paths, n_strikes)

        # Step 2: IS weights → reweighted target
        print(f"  [MSB] Computing IS weights …")
        w = self._importance_weights(payoffs, market_prices)

        ess     = 1.0 / float(np.sum(w ** 2))
        ess_pct = 100.0 * ess / self._n_paths
        print(f"  [MSB] ESS = {ess:.0f} / {self._n_paths}  ({ess_pct:.1f}%)")

        # Step 3: resample P* and train bridge P₀ → P* 
        # Resample WITH replacement using IS weights to get n_paths draws
        # from P*.  Then pair each P₀ path with its IS-resampled P* counterpart
        # so the bridge learns: given a P₀ payoff vector, predict the P* one.
        rng = np.random.default_rng(self._seed + 1)
        idx_tgt = rng.choice(self._n_paths, size=self._n_paths,
                                 replace=True, p=w)
        payoffs_tgt = payoffs[idx_tgt]   # (n_paths, n_strikes) — draws from P*

        print(f"  [MSB] P₀ payoffs mean: {payoffs.mean(axis=0).round(4)}")
        print(f"  [MSB] P* payoffs mean: {payoffs_tgt.mean(axis=0).round(4)}")
        print(f"  [MSB] Market prices:   {market_prices.round(4)}")

        print(f"  [MSB] Training ConditionalBrenierSinkhorn "
              f"({self._n_iter} iters) …")
        self._bridge = ConditionalBrenierSinkhorn(
            d_in = n_strikes,
            d_out = n_strikes,
            d_noise = max(4, n_strikes // 2),
            hidden  = 128,
            n_blocks= 3,
        )
        self._bridge.fit(
            payoffs, payoffs_tgt,      # P₀ → P* transport
            n_iter = self._n_iter,
            batch = self._batch,
            lr = self._lr,
            alpha_final = 0.7,
            warmup_mse_iters = max(100, self._n_iter // 10),
            log_every  = max(200, self._n_iter // 5),
        )

        # Step 4: price and evaluate 
        model_prices = self.price_calls(strikes, T)
        result = CalibrationResult(
            method_name  = self.name,
            params = np.zeros(0),
            model_prices = model_prices,
            n_iterations = self._n_iter,
            success = True,
            message = f"ESS={ess_pct:.1f}%",
        )
        result.compute_errors(market_prices)
        return result

    def price_calls(
        self,
        strikes: np.ndarray,
        T:       float,
        **kwargs,
    ) -> np.ndarray:
        if self._bridge is None:
            raise RuntimeError("Call calibrate() first.")

        # Source: fresh P₀ payoffs
        payoffs_src = self._simulate_payoffs(
            strikes, T, self._n_paths_price, self._seed + 99
        )
        # sample_batch: (M, d_in) -> (M, n_samples, d_out)
        payoffs_mapped = self._bridge.sample_batch(
            payoffs_src, n=20, chunk_m=128
        )  # (n_paths_price, 20, n_strikes)

        # Diagnostics: check raw output scale
        flat = payoffs_mapped.reshape(-1, payoffs_mapped.shape[-1])
        print(f"  [MSB price_calls] sample_batch output:"
              f"  mean={flat.mean():.4f}  std={flat.std():.4f}"
              f"  min={flat.min():.4f}  max={flat.max():.4f}")
        print(f"  [MSB price_calls] payoffs_src:"
              f"  mean={payoffs_src.mean():.4f}  std={payoffs_src.std():.4f}")

        # Per-strike means
        per_strike = payoffs_mapped.mean(axis=(0, 1))
        print(f"  [MSB price_calls] per-strike mean: {np.round(per_strike, 4)}")
        return per_strike

    def reset(self) -> None:
        self._bridge  = None
        self._strikes = None
        self._T = None
        self._df = None

# Main — smoke test / benchmark

if __name__ == "__main__":
    import time

    print("MSBCalibration: smoke test")

    strikes = np.linspace(80, 120, 10)
    T = 1.0

    # "True" market: higher vol-of-vol and stronger skew 
    # These parameters are DIFFERENT from the model P₀ below,
    # creating a real model-vs-market gap for the bridge to correct.
    heston_true = HestonUnified(
        S0=100, v0=0.06, kappa=1.5, theta=0.06,
        sigma=0.5, rho=-0.8, mode="financial"
    )
    print(f"\nTrue market params (unknown to model):")
    print(f"  v₀={heston_true.v0}  κ={heston_true.kappa}  "
          f"θ={heston_true.theta}  σ={heston_true.sigma}  ρ={heston_true.rho}")

    print(f"\nGenerating market prices …")
    t0 = time.perf_counter()
    market_prices = heston_true.option_prices_mc(strikes, T, n_paths=200_000)
    print(f"  Done in {time.perf_counter()-t0:.2f}s")
    print(f"  Market prices: {np.round(market_prices, 4)}")

    # 2. Model P₀: misspecified (lower vol, weaker skew)
    heston_model = HestonUnified(
        S0=100, v0=0.04, kappa=2.0, theta=0.04,
        sigma=0.3, rho=-0.7, mode="financial"
    )
    print(f"\nModel P₀ params (what MSB starts from):")
    print(f"  v₀={heston_model.v0}  κ={heston_model.kappa}  "
          f"θ={heston_model.theta}  σ={heston_model.sigma}  ρ={heston_model.rho}")

    # Baseline: P₀ prices before calibration
    t0 = time.perf_counter()
    baseline_prices = heston_model.option_prices_mc(strikes, T, n_paths=100_000)
    print(f"\nBaseline P₀ prices (before calibration):")
    print(f"  {np.round(baseline_prices, 4)}")
    print(f"  Baseline MAE = {np.mean(np.abs(baseline_prices - market_prices)):.4f}")

    # 3. Calibrate MSB 
    print(f"\nCalibrating MSB …")
    msb = MSBCalibration(
        heston_model = heston_model,
        n_paths = 4000,
        n_iter = 2000,
        batch = 256,
        lr = 3e-4,
        n_paths_price = 5000,
        seed = 42,
    )

    t0 = time.perf_counter()
    result = msb.calibrate(strikes, market_prices, T)
    calib_time = time.perf_counter() - t0

    # 4. Results
    baseline_mae = float(np.mean(np.abs(baseline_prices - market_prices)))
    improvement  = (baseline_mae - result.mae) / baseline_mae * 100

    print(f"\n{'─'*60}")
    print(f"Calibration time: {calib_time:.1f}s")
    print(f"Status: {'OK' if result.success else 'FAILED'} — {result.message}")
    print(f"\nCalibration Quality:")
    print(f"Baseline MAE = {baseline_mae:.6f}  (P₀ without correction)")
    print(f"MSB MAE = {result.mae:.6f}  (after bridge)")
    print(f"RMSE = {result.rmse:.6f}")
    print(f"Max Err = {result.max_err:.6f}")
    print(f"Improvement = {improvement:+.1f}%")

    print(f"\n{'Strike':>8}  {'Market':>10}  {'P₀':>10}  {'MSB':>10}  {'Err P₀':>9}  {'Err MSB':>9}")
    print("─" * 65)
    for K, mkt, base, msb_p in zip(strikes, market_prices, baseline_prices, result.model_prices):
        print(f"  {K:6.1f}  {mkt:10.4f}  {base:10.4f}  {msb_p:10.4f}"
              f"  {abs(base-mkt):9.4f}  {abs(msb_p-mkt):9.4f}")

    print(f"\nDone!")
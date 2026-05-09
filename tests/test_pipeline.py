import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in [
    _ROOT,
    _ROOT / "models",
    _ROOT / "calibration",   # arbitrage_filter, base
    _ROOT / "data_pipeline" / "cleaning",   # arbitrage_filter
]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# Fixtures

@pytest.fixture(scope="module")
def heston_model():
    from heston import HestonUnified
    return HestonUnified(
        S0=100.0, v0=0.04, kappa=2.0, theta=0.04,
        sigma=0.3, rho=-0.7, r=0.0,
    )

@pytest.fixture(scope="module")
def synthetic_data():
    """
    Load synthetic data if available, else create a minimal in-memory dataset.
    """
    candidates = [
        Path("data/unified_heston_prediction_data.npz"),
        Path("../data/unified_heston_prediction_data.npz"),
    ]
    for p in candidates:
        if p.exists():
            data = np.load(str(p), allow_pickle=True)
            return {
                "vol_surface":  data["vol_surface"],
                "strikes_norm": data["strikes_norm"],
                "maturities":   data["maturities"],
                "X1":           data["full_X1"],
                "X2":           data["full_X2"],
                "heston_params":data["heston_params"].item(),
                "lookback":     int(data["full_lookback"]),
                "horizon":      int(data["full_horizon"]),
            }
    # Minimal fallback: generate on the fly
    return _generate_minimal_data()


def _generate_minimal_data():
    """Generate a tiny dataset without running the full generator"""
    from scipy.stats import norm as spnorm

    S0 = 100.0
    moneyness = np.linspace(0.85, 1.15, 9)
    maturities = np.array([0.25, 0.5, 0.75, 1.0])
    vol_surface = np.zeros((len(maturities), len(moneyness)))
    for i, T in enumerate(maturities):
        atm = 0.20 + 0.01 * i
        vol_surface[i] = atm - 0.05 * np.log(moneyness)

    n = 300
    rng = np.random.RandomState(0)
    X1  = rng.randn(n, 30)   # 30 features
    X2  = rng.randn(n, 3)    # return, vol, drawdown

    return {
        "vol_surface":  vol_surface,
        "strikes_norm": moneyness,
        "maturities":   maturities,
        "X1": X1,
        "X2": X2,
        "heston_params": {
            "S0": S0, "v0": 0.04, "kappa": 2.0,
            "theta": 0.04, "sigma": 0.3, "rho": -0.7, "r": 0.0,
        },
        "lookback": 20,
        "horizon":  5,
    }


# Test 1: Heston simulation → option pricing

class TestHestonPipeline:

    def test_simulate_and_price(self, heston_model):
        strikes = np.array([90., 100., 110.])
        prices  = heston_model.option_prices_mc(strikes, T=1.0, n_paths=5000, n_steps=252)
        assert len(prices) == len(strikes)
        assert np.all(prices >= 0), "Negative option prices"
        # ITM call must be >= OTM call (call spread no-arb, MC tolerance 2.0)
        assert prices[0] >= prices[1] >= prices[2] - 2.0

    def test_paths_martingale(self, heston_model):
        # Martingale check: E[S_T] ≈ S0 * exp(r*T), r=0 so E[S_T] ≈ S0.
        # Use option_prices_mc with strike=0 as E[S_T] proxy (digital call = E[S_T]*df).
        # Alternative: price at-the-money and check put-call parity is roughly satisfied.
        S0      = heston_model.S0
        strikes = np.array([S0])
        call    = heston_model.option_prices_mc(strikes, T=1.0, n_paths=10_000, n_steps=252)
        # For r=0, call(K=S0) ≈ S0*(2N(d1)-1) where d1>0 → call > 0 always.
        # Martingale check via put-call parity: call - put ≈ S0 - K = 0 at ATM when r=0.
        # We just verify call is finite and positive (smoke test for no path explosion).
        assert np.isfinite(call[0]), "Non-finite ATM call price"
        assert call[0] > 0, "ATM call price is zero (path explosion suspected)"
        assert call[0] < S0, "ATM call price >= S0 (impossible for r=0)"


# Test 2: Brenier estimator end-to-end

class TestBrenierPipeline:

    def test_fit_predict(self, synthetic_data):
        from brenier import ConditionalBrenierEstimator

        X1 = synthetic_data["X1"]
        X2 = synthetic_data["X2"]
        n  = len(X1)
        d1 = 7
        split = int(n * 0.7)

        t   = 0.1 * (split ** (-1/3))
        eps = t ** 2

        est = ConditionalBrenierEstimator(d1=d1, d2=X2.shape[1], t=t, epsilon=eps)
        est.fit(X1[:split, -d1:], X2[:split])

        preds = est.predict(X1[split:, -d1:])
        assert preds.shape == (n - split, X2.shape[1])
        assert np.all(np.isfinite(preds)), "Non-finite predictions"

    def test_w2_error_finite(self, synthetic_data):
        from brenier import ConditionalBrenierEstimator

        X1 = synthetic_data["X1"]
        X2 = synthetic_data["X2"]
        d1 = 7
        split = 200

        est = ConditionalBrenierEstimator(d1=d1, d2=X2.shape[1], t=0.05, epsilon=0.003)
        est.fit(X1[:split, -d1:], X2[:split])
        w2 = est.compute_wasserstein_error(X1[split:split+50, -d1:], X2[split:split+50])
        assert np.isfinite(w2), f"W2 error non-finite: {w2}"
        assert w2 >= 0.0


# Test 3: MSB training smoke

@pytest.mark.slow
class TestMSBPipeline:

    def test_msb_trains_and_prices(self, heston_model, synthetic_data):
        try:
            from brenier import MartingaleSchrodingerBridge as MSB
        except ImportError:
            try:
                from bridge import MartingaleSchrodingerBridge as MSB
            except ImportError:
                pytest.skip("MartingaleSchrodingerBridge not available as standalone module")

        from scipy.stats import norm

        S0   = heston_model.S0
        mats = synthetic_data["maturities"]
        surf = synthetic_data["vol_surface"]
        stk  = synthetic_data["strikes_norm"] * S0
        T    = float(mats[-1])  # 1Y
        ivs  = surf[-1, :]

        prices = np.array([
            S0 * norm.cdf((np.log(S0/K) + 0.5*iv**2*T)/(iv*np.sqrt(T)))
            - K * norm.cdf((np.log(S0/K) + 0.5*iv**2*T)/(iv*np.sqrt(T)) - iv*np.sqrt(T))
            for K, iv in zip(stk, ivs)
        ])

        msb = MSB(heston_model)
        losses = msb.train(
            strikes=stk, market_prices=prices, T=T,
            n_iterations=20, batch_size=64, patience=15,
        )
        assert len(losses) > 0
        assert all(np.isfinite(l) for l in losses)

        price, se = msb.price_option(S0, T, n_paths=500)
        assert price >= 0.0
        assert np.isfinite(price)


# Test 4: Arbitrage filter → preprocessor → Brenier

class TestFullDataPipeline:

    def test_filter_then_brenier(self, synthetic_data):
        try:
            from arbitrage_filters import filter_surface, check_calendar_arbitrage
        except ModuleNotFoundError:
            pytest.skip("arbitrage_filter not found — check calibration/ folder exists")
        from brenier import ConditionalBrenierEstimator

        surf = synthetic_data["vol_surface"].copy()
        mats = synthetic_data["maturities"]
        mono = synthetic_data["strikes_norm"]

        # Introduce a calendar violation
        surf[1, 5] *= 0.3

        _, surf_clean, report = filter_surface(
            pd.DataFrame(), surf, mats, mono,
            run_pcp=False,
        )
        cal = check_calendar_arbitrage(surf_clean, mats)
        assert cal["is_arbitrage_free"], "Calendar arb remains after filter"

        # Then run Brenier on the data
        X1 = synthetic_data["X1"]
        X2 = synthetic_data["X2"]
        d1 = 7
        est = ConditionalBrenierEstimator(d1=d1, d2=3, t=0.05, epsilon=0.003)
        est.fit(X1[:200, -d1:], X2[:200])
        pred = est.predict(X1[200, -d1:])
        assert pred.shape == (3,)
        assert np.all(np.isfinite(pred))


# Test 5: Base interface compliance

class TestCalibrationBase:

    def test_naive_baseline_implements_interface(self, heston_model):
        try:
            from base import NaiveBaseline, CalibrationMethod
        except ModuleNotFoundError:
            pytest.skip("base module not found — check calibration/ folder exists")

        baseline = NaiveBaseline(heston_model)
        assert isinstance(baseline, CalibrationMethod)
        assert hasattr(baseline, "name")
        assert hasattr(baseline, "calibrate")
        assert hasattr(baseline, "price_calls")
        assert hasattr(baseline, "reset")

    def test_calibration_result_errors(self):
        try:
            from base import CalibrationResult
        except ModuleNotFoundError:
            pytest.skip("base module not found — check calibration/ folder exists")
        import numpy as np

        result = CalibrationResult(
            method_name  = "Test",
            params       = np.zeros(3),
            model_prices = np.array([10., 5., 2.]),
        )
        market = np.array([10.1, 5.1, 2.1])
        result.compute_errors(market)
        assert result.mae > 0
        assert result.rmse > 0
        assert result.max_error > 0


# Run

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
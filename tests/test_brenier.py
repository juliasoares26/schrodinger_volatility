import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from brenier import ConditionalBrenierEstimator, make_brenier_estimator

# Helpers

def make_gaussian_data(n=200, d1=3, d2=2, seed=0):
    """Joint Gaussian (X1, X2) with known conditional mean."""
    rng = np.random.RandomState(seed)
    X1  = rng.randn(n, d1)
    # X2 = A @ X1 + noise  so E[X2|X1] = A @ X1
    A   = rng.randn(d2, d1) * 0.5
    X2  = (A @ X1.T).T + rng.randn(n, d2) * 0.3
    return X1, X2, A

# Test: basic fit / predict shape

class TestBrenierShapes:

    def test_predict_single_query_shape(self):
        X1, X2, _ = make_gaussian_data(n=100, d1=3, d2=2)
        est = ConditionalBrenierEstimator(d1=3, d2=2, t=0.1, epsilon=0.01)
        est.fit(X1, X2)

        pred = est.predict(X1[0])   # single query (d1,)
        assert pred.shape == (2,), f"Expected (2,), got {pred.shape}"

    def test_predict_batch_query_shape(self):
        X1, X2, _ = make_gaussian_data(n=100, d1=3, d2=2)
        est = ConditionalBrenierEstimator(d1=3, d2=2, t=0.1, epsilon=0.01)
        est.fit(X1, X2)

        preds = est.predict(X1[:10])   # batch (10, d1)
        assert preds.shape == (10, 2), f"Expected (10, 2), got {preds.shape}"

    def test_predict_with_distribution(self):
        X1, X2, _ = make_gaussian_data(n=100, d1=2, d2=2)
        est = ConditionalBrenierEstimator(d1=2, d2=2, t=0.1, epsilon=0.01)
        est.fit(X1, X2)

        pred, samples = est.predict(X1[0], return_distribution=True, n_samples=50)
        assert samples.shape == (50, 2)

# Test: identity on trivial data

class TestBrenierIdentity:

    def test_identity_map(self):
        """On constant data X2 = c, predictions should equal c regardless of X1"""
        rng = np.random.RandomState(42)
        n   = 150
        d1, d2 = 2, 2
        X1 = rng.randn(n, d1)
        X2 = np.ones((n, d2)) * np.array([3.0, -1.0])  # constant

        est = ConditionalBrenierEstimator(d1=d1, d2=d2, t=0.05, epsilon=0.005)
        est.fit(X1, X2)

        pred = est.predict(X1[0])
        np.testing.assert_allclose(pred, [3.0, -1.0], atol=0.5)


# Test: Wasserstein-2 error

class TestWasserstein:

    def test_w2_positive(self):
        X1, X2, _ = make_gaussian_data(n=200, d1=3, d2=2)
        split = 150
        est = ConditionalBrenierEstimator(d1=3, d2=2, t=0.08, epsilon=0.006)
        est.fit(X1[:split], X2[:split])
        w2 = est.compute_wasserstein_error(X1[split:], X2[split:])
        assert w2 >= 0.0, f"W2 should be non-negative, got {w2}"

    def test_w2_decreases_with_more_data(self):
        """Fitting on more data should reduce W2 error."""
        rng = np.random.RandomState(7)
        n   = 600
        d1, d2 = 4, 2
        X1 = rng.randn(n, d1)
        X2 = 0.5 * X1[:, :d2] + rng.randn(n, d2) * 0.2

        test_X1 = X1[500:]
        test_X2 = X2[500:]

        w2_small = _fit_and_eval(X1[:100], X2[:100], test_X1, test_X2, d1, d2)
        w2_large = _fit_and_eval(X1[:400], X2[:400], test_X1, test_X2, d1, d2)
        # w2_large should generally be <= w2_small (statistical, may occasionally fail)
        # Use a loose bound to avoid flakiness
        assert w2_large <= w2_small * 1.5, (
            f"W2 did not decrease with more data: {w2_small:.4f} → {w2_large:.4f}"
        )


def _fit_and_eval(X1_tr, X2_tr, X1_te, X2_te, d1, d2):
    n = len(X1_tr)
    t = 0.1 * (n ** (-1/3))
    eps = t ** 2
    est = ConditionalBrenierEstimator(d1=d1, d2=d2, t=t, epsilon=eps)
    est.fit(X1_tr, X2_tr)
    return est.compute_wasserstein_error(X1_te, X2_te)

# Test: make_brenier_estimator factory

class TestFactory:

    def test_theory_scaling(self):
        n = 1000
        est = make_brenier_estimator(n, d1=3, d2=2, method="theory")
        expected_t = 0.1 * (n ** (-1/3))
        np.testing.assert_allclose(est.t, expected_t, rtol=1e-6)
        np.testing.assert_allclose(est.epsilon, expected_t ** 2, rtol=1e-6)

    def test_practical_scaling(self):
        n = 1000
        est = make_brenier_estimator(n, d1=3, d2=2, method="practical")
        expected_t = 0.1 * (n ** (-1/5))
        np.testing.assert_allclose(est.t, expected_t, rtol=1e-6)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            make_brenier_estimator(100, d1=2, d2=2, method="bogus")

# Test: Sinkhorn convergence

class TestSinkhorn:

    def test_dual_potentials_finite(self):
        """Sinkhorn should not produce NaN or Inf in dual potentials"""
        X1, X2, _ = make_gaussian_data(n=50, d1=2, d2=2, seed=3)
        est = ConditionalBrenierEstimator(d1=2, d2=2, t=0.1, epsilon=0.01)
        est.fit(X1, X2)
        assert np.all(np.isfinite(est.dual_potentials)), "Dual potentials contain non-finite values"

    def test_rescaled_cost_symmetry(self):
        """Rescaled cost matrix should be symmetric when X == Y."""
        X1, X2, _ = make_gaussian_data(n=40, d1=2, d2=2)
        est = ConditionalBrenierEstimator(d1=2, d2=2, t=0.05, epsilon=0.005)
        joint = np.hstack([X1, X2])
        C = est._compute_rescaled_cost(joint, joint)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

# Test: batched predict consistency

class TestBatchConsistency:

    def test_batch_vs_single(self):
        """Batched predict should match single-query predict."""
        X1, X2, _ = make_gaussian_data(n=100, d1=3, d2=2)
        est = ConditionalBrenierEstimator(d1=3, d2=2, t=0.1, epsilon=0.01)
        est.fit(X1, X2)

        queries = X1[:5]
        batch_pred   = est.predict(queries, n_mean_samples=0)                    # (5, 2)
        single_preds = np.array([est.predict(q, n_mean_samples=0) for q in queries])  # (5, 2)

        np.testing.assert_allclose(batch_pred, single_preds, atol=1e-8,
                                   err_msg="Batch and single predictions disagree")

# Run

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
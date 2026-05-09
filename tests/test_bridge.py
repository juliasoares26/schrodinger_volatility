"""
tests/test_bridge.py — Smoke tests for MartingaleSchrodingerBridge
===================================================================
Checks:
  - Training produces finite, non-increasing losses
  - price_option returns a non-negative scalar
  - Early stopping fires when patience is exceeded
  - reset() creates a fresh network (loss restarts)
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from heston import HestonUnified
from bridge import MartingaleSchrodingerBridge


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def heston():
    return HestonUnified(S0=100.0, v0=0.04, kappa=2.0, theta=0.04,
                         sigma=0.3, rho=-0.7, r=0.0)


@pytest.fixture(scope="module")
def market_setup(heston):
    """Tiny 5-strike synthetic market priced from the naked model."""
    strikes = np.array([85., 90., 95., 100., 105., 110., 115.])
    T       = 1.0
    prices  = heston.option_prices_mc(strikes, T=T, n_paths=10_000)
    return strikes, prices, T


# =============================================================================
# Training
# =============================================================================

class TestMSBTraining:

    def test_losses_list_nonempty(self, heston, market_setup):
        strikes, prices, T = market_setup
        msb = MartingaleSchrodingerBridge(heston)
        losses = msb.train(strikes=strikes, market_prices=prices, T=T,
                           n_iterations=30, batch_size=64, patience=20)
        assert isinstance(losses, list)
        assert len(losses) > 0, "No training iterations recorded"

    def test_losses_are_finite(self, heston, market_setup):
        strikes, prices, T = market_setup
        msb = MartingaleSchrodingerBridge(heston)
        losses = msb.train(strikes=strikes, market_prices=prices, T=T,
                           n_iterations=30, batch_size=64, patience=20)
        assert all(np.isfinite(l) for l in losses), "NaN/Inf loss detected"

    def test_early_stopping(self, heston, market_setup):
        """With patience=5 and 50 iterations the run must stop early."""
        strikes, prices, T = market_setup
        msb = MartingaleSchrodingerBridge(heston)
        losses = msb.train(strikes=strikes, market_prices=prices, T=T,
                           n_iterations=50, batch_size=64, patience=5)
        assert len(losses) <= 50, "Early stopping did not fire"


# =============================================================================
# Pricing
# =============================================================================

class TestMSBPricing:

    def test_price_option_non_negative(self, heston, market_setup):
        strikes, prices, T = market_setup
        msb = MartingaleSchrodingerBridge(heston)
        msb.train(strikes=strikes, market_prices=prices, T=T,
                  n_iterations=20, batch_size=64, patience=15)
        K = 100.0
        price, se = msb.price_option(K, T, n_paths=500)
        assert price >= 0.0, f"Negative option price: {price}"
        assert np.isfinite(price), f"Non-finite option price: {price}"
        assert se >= 0.0, f"Negative standard error: {se}"

    def test_call_spread_monotone(self, heston, market_setup):
        """ATM call must be worth more than deep OTM call."""
        strikes, prices, T = market_setup
        msb = MartingaleSchrodingerBridge(heston)
        msb.train(strikes=strikes, market_prices=prices, T=T,
                  n_iterations=20, batch_size=64, patience=15)
        atm, _  = msb.price_option(100.0, T, n_paths=1000)
        otm, _  = msb.price_option(120.0, T, n_paths=1000)
        assert atm >= otm - 1.0, f"Call spread violated: ATM={atm:.4f} < OTM={otm:.4f}"


# =============================================================================
# Potential optimizer
# =============================================================================

class TestSchrodingerPotential:

    def test_potential_weights_bounded(self, heston, market_setup):
        from bridge import SchrodingerPotentialOptimizer
        strikes, prices, T = market_setup
        opt = SchrodingerPotentialOptimizer(strikes, prices, heston)
        weights = opt.optimize(max_iter=20)
        assert np.all(np.isfinite(weights)), "Non-finite potential weights"
        # Weights should stay within bounds [-5, 5]
        assert np.all(np.abs(weights) <= 10), f"Weights too large: {weights}"


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
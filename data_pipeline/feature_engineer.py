import numpy as np
import pandas as pd
from typing import Dict
import warnings

warnings.filterwarnings("ignore")

from surface_builder import MONEYNESS_GRID, MATURITY_GRID


def build_prediction_features(
    prices_series: pd.Series,
    vol_surface: np.ndarray,
    moneyness_grid: np.ndarray = None,
    maturity_grid: np.ndarray = None,
    lookback: int = 20,
    horizon: int = 5,
    r: float = 0.0,
    q: float = 0.015,
) -> Dict:
    """
    Build X1 / X2 prediction arrays from price history and IV surface

    Parameters
    prices_series: pd.Series  — daily closes, indexed by date
    vol_surface: (n_T, n_K) IV matrix on fixed grid
    lookback: past return window length
    horizon: forecast horizon in trading days

    Returns
    dict with keys:
        X1, X2, X1_raw, X1_mean, X1_std, lookback, horizon,
        feature_dim, target_dim, iv_1m, iv_3m, iv_slope
    """
    if moneyness_grid is None:
        moneyness_grid = MONEYNESS_GRID
    if maturity_grid is None:
        maturity_grid  = MATURITY_GRID

    prices = np.array(prices_series.values, dtype=float)
    returns = np.diff(np.log(prices))
    n = len(returns)

    # ATM IV proxies — map to the two shortest maturities available on the grid
    # argmin(|T - 1/12|) and argmin(|T - 3/12|) both collapse to index 0 when
    # the shortest grid point is 0.25Y, making iv_slope identically zero
    # Instead we explicitly anchor to the first two grid points
    atm_idx  = len(moneyness_grid) // 2
    sorted_T_idx = np.argsort(maturity_grid)          # ascending order
    T_1m_idx = int(sorted_T_idx[0])                   # shortest maturity
    T_3m_idx = int(sorted_T_idx[1]) if len(sorted_T_idx) > 1 else T_1m_idx

    iv_1m = float(vol_surface[T_1m_idx, atm_idx]) if not np.isnan(vol_surface[T_1m_idx, atm_idx]) else 0.20
    iv_3m = float(vol_surface[T_3m_idx, atm_idx]) if not np.isnan(vol_surface[T_3m_idx, atm_idx]) else 0.20
    iv_slope = iv_3m - iv_1m

    X1_list: list = []
    X2_list: list = []

    min_idx = lookback
    max_idx = n - horizon

    if min_idx >= max_idx:
        raise ValueError(
            f"Insufficient data: need ≥ {lookback + horizon + 1} return observations, got {n}."
        )

    for i in range(min_idx, max_idx):
        past_ret = returns[i - lookback: i]

        realized_vol = np.std(past_ret) * np.sqrt(252)
        current_vol = np.std(past_ret[-5:]) * np.sqrt(252) if len(past_ret) >= 5 else realized_vol
        momentum_5d  = float(np.mean(past_ret[-5:])) if len(past_ret) >= 5 else 0.0
        momentum_20d = float(np.mean(past_ret))
        trend = float(past_ret[-1] - past_ret[0]) if len(past_ret) > 1 else 0.0

        if len(past_ret) > 1:
            autocorr = float(np.corrcoef(past_ret[:-1], past_ret[1:])[0, 1])
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        rolling_vols = [
            np.std(past_ret[k - 5: k]) * np.sqrt(252)
            for k in range(max(5, lookback - 10), lookback)
        ]
        vol_of_vol = float(np.std(rolling_vols)) if len(rolling_vols) > 1 else 0.0

        raw_features = np.concatenate([
            past_ret,
            [realized_vol, current_vol, momentum_5d, momentum_20d,
             trend, autocorr, vol_of_vol, iv_1m, iv_3m, iv_slope]
        ])

        future_ret= returns[i: i + horizon]
        cumulative_return = float(np.sum(future_ret))
        future_vol = float(np.std(future_ret) * np.sqrt(252)) if len(future_ret) > 1 else 0.0

        cum_ret = np.cumsum(future_ret)
        running_max  = np.maximum.accumulate(cum_ret)
        max_drawdown = float(np.max(running_max - cum_ret)) if len(cum_ret) > 0 else 0.0

        X1_list.append(raw_features)
        X2_list.append([cumulative_return, future_vol, max_drawdown])

    X1_raw = np.array(X1_list, dtype=float)
    X2 = np.array(X2_list, dtype=float)

    X1_mean = X1_raw.mean(axis=0)
    X1_std = X1_raw.std(axis=0) + 1e-8
    X1 = (X1_raw - X1_mean) / X1_std

    print(f"\n Features built: {len(X1)} samples  d1={X1.shape[1]}  d2={X2.shape[1]}")
    print(f"Return: mean={X2[:, 0].mean():.5f}  std={X2[:, 0].std():.5f}")
    print(f"Fut. vol: mean={X2[:, 1].mean():.4f}")

    return {
        "X1": X1,
        "X2": X2,
        "X1_raw": X1_raw,
        "X1_mean": X1_mean,
        "X1_std": X1_std,
        "lookback": lookback,
        "horizon": horizon,
        "feature_dim": int(X1.shape[1]),
        "target_dim": int(X2.shape[1]),
        "iv_1m": iv_1m,
        "iv_3m": iv_3m,
        "iv_slope": iv_slope,
    }


def estimate_heston_params(
    prices_series: pd.Series,
    r: float = 0.0,
    q: float = 0.015,
) -> Dict:
    returns  = np.diff(np.log(np.array(prices_series.values, dtype=float)))
    rv_daily = returns ** 2

    v0  = float(np.mean(rv_daily[-21:]) * 252)
    theta = float(np.mean(rv_daily) * 252)

    window = 21
    roll_var = np.array([
        np.mean(rv_daily[i: i + window]) * 252
        for i in range(len(rv_daily) - window)
    ])

    sigma = float(np.std(roll_var) / (theta + 1e-8))
    sigma = float(np.clip(sigma, 0.1, 3.0))   # bound ampliado: SPX real ~2.3

    if len(roll_var) > 2:
        rho_ar = float(np.corrcoef(roll_var[:-1], roll_var[1:])[0, 1])
        rho_ar = float(np.clip(rho_ar, 0.0, 0.9999))
        kappa  = float(np.clip(-np.log(max(rho_ar, 1e-4)) * 252, 0.5, 10.0))
    else:
        kappa = 2.0

    # rho: corr(ret[t], Δrv[t]) — proxy correto para corr(dW_S, dW_v)
    # O código anterior usava corr(ret[t], rv[t+1]) que tem viés para zero
    # porque rv[t+1] = ret[t+1]^2 é independente de ret[t] sob H0
    if len(returns) > 5:
        delta_var = np.diff(rv_daily)          # rv[t+1] - rv[t], len N-2
        ret_sync = returns[:-1]               # ret[t], len N-2
        rho_raw  = float(np.corrcoef(ret_sync, delta_var)[0, 1])
        # Se o estimador momento ainda produzir valor próximo de zero
        # (comum com dados de alta frequência), ancora em -0.70 via shrinkage
        # ponderado pela magnitude: quanto mais próximo de zero, mais âncora.
        anchor = -0.70
        weight = np.clip(np.abs(rho_raw) * 5, 0.0, 1.0)   # 0 se |rho|<0.05
        rho_sv = float(weight * rho_raw + (1 - weight) * anchor)
        rho_sv = float(np.clip(rho_sv, -0.95, -0.10))
    else:
        rho_sv = -0.70

    S0 = float(prices_series.iloc[-1])

    params = {
        "S0": S0,
        "v0": max(v0, 0.01),
        "kappa": kappa,
        "theta": max(theta, 0.01),
        "sigma": sigma,
        "rho": rho_sv,
        "r": r,
    }

    print(f"\n  Estimated Heston params:")
    for k, v in params.items():
        print(f"{k}: {v:.4f}")

    return params
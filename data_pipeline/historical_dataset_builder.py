import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Config

DEFAULT_NPZ = r"C:\volatility-options\data\live_spx_data.npz"
DEFAULT_OUT = r"C:\volatility-options\data\live_spx_data_extended.npz"
DEFAULT_START = "2019-01-01"  
RV_WINDOWS = [5, 21, 63]   

HORIZON = 5


# 1. Download

def download_history(start: str, end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    tickers = {"^SPX": "spx", "^VIX": "vix", "^VIX3M": "vix3m", "SPY": "spy"}

    dfs = {}
    for ticker, name in tickers.items():
        print(f"Downloading {ticker}...", end=" ", flush=True)
        try:
            raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
            if raw.empty:
                print("VAZIO")
                continue
            s = raw["Close"].copy()
            s.index = pd.to_datetime(s.index).normalize()
            dfs[name] = s
            # guarda OHLC do SPX para Parkinson estimator real
            if ticker == "^SPX":
                for col in ["High", "Low", "Open"]:
                    if col in raw.columns:
                        c = raw[col].copy()
                        c.index = pd.to_datetime(c.index).normalize()
                        dfs[f"spx_{col.lower()}"] = c
            print(f"OK ({len(s)} dias)")
        except Exception as e:
            print(f"ERRO: {e}")

    if "spx" not in dfs:
        raise ValueError("Não foi possível baixar dados do SPX.")

    df = pd.DataFrame(dfs)
    df = df.sort_index().ffill().dropna(subset=["spx"])
    return df


# 2. Feature Engineering

def compute_features(df: pd.DataFrame) -> np.ndarray:

    spx   = df["spx"]
    vix   = df.get("vix",   pd.Series(np.nan, index=df.index)) / 100.0
    vix3m = df.get("vix3m", pd.Series(np.nan, index=df.index)) / 100.0

    log_spx = np.log(spx)
    ret_1d  = log_spx.diff(1)
    ret_21d = log_spx.diff(21)

    rv5  = ret_1d.rolling(5).std()  * np.sqrt(252)
    rv21 = ret_1d.rolling(21).std() * np.sqrt(252)
    rv63 = ret_1d.rolling(63).std() * np.sqrt(252)
    rv21 = rv21.clip(lower=1e-4)
    rv63 = rv63.clip(lower=1e-4)

    # VIX fallback
    vix_filled   = vix.where(vix.notna(), rv21)
    vix3m_filled = vix3m.where(vix3m.notna(), vix_filled)

    log_vix       = np.log(vix_filled.clip(lower=1e-4))
    vix_term      = (vix3m_filled - vix_filled).clip(-0.5, 0.5)
    vix_rv_spread = (vix_filled - rv21).clip(-0.3, 0.3)
    skew_proxy    = ((vix_filled - rv21) / rv21.clip(lower=1e-4)).clip(-3, 3)

    # Parkinson estimator 
    if "spx_high" in df.columns and "spx_low" in df.columns:
        log_hl       = np.log(df["spx_high"].clip(lower=1e-4) / df["spx_low"].clip(lower=1e-4))
        park_daily   = (log_hl ** 2) / (4.0 * np.log(2))
        parkinson_5d = np.sqrt(park_daily.rolling(5).mean() * 252).clip(0, 2.0)
    else:
        abs_ret = ret_1d.abs()
        parkinson_5d = (
            (abs_ret.rolling(3).mean() + abs_ret.shift(1).rolling(3).mean()) * np.sqrt(252 / 3)
        ).clip(0, 2.0)

    rv21_median = rv21.rolling(252, min_periods=63).median().fillna(rv21.median())
    vol_regime  = (rv21 / rv21_median.clip(lower=1e-4)).clip(0, 3)

    rv_accel = np.log((rv5 / rv21.clip(lower=1e-4)).clip(0.1, 10.0))

    # drawdown 21d
    roll_max = spx.rolling(21).max()
    dd_21    = ((spx - roll_max) / roll_max.clip(lower=1e-4)).clip(-1, 0)

    cols = [
        rv5,           # 0
        rv21,          # 1
        rv63,          # 2
        log_vix,       # 3
        vix_term,      # 4
        vix_rv_spread, # 5
        skew_proxy,    # 6
        parkinson_5d,  # 7
        vol_regime,    # 8
        rv_accel,      # 9
        ret_21d,       # 10
        dd_21,         # 11
    ]

    X = np.column_stack([c.values for c in cols]).astype(np.float32)
    return X, df.index


# 3. Main Builder

def _extract_heston_params(npz_val, default=(2.0, 0.04, 0.3, -0.7)):
    try:
        v = npz_val
        if isinstance(v, np.ndarray) and v.ndim == 0:
            v = v.item()
        if isinstance(v, dict):
            keys = ["kappa", "theta", "sigma", "rho"]
            v = [v.get(k, d) for k, d in zip(keys, default)]
        v = np.array(v, dtype=float).flatten()
        if len(v) >= 2:
            return v
    except Exception:
        pass
    return np.array(default, dtype=float)


def _patch_live_features(X: np.ndarray) -> np.ndarray:
    return X


def build_extended_dataset(
    existing_npz: str,
    output_npz: str,
    start: str = DEFAULT_START,
    horizon: int = HORIZON,
    min_valid_rows: int = 126,   
) -> None:

    print(" Historical Dataset Builder: SPX Bridge")

    print(f"\n[1/4] Loading NPZ: {existing_npz}")
    existing = np.load(existing_npz, allow_pickle=True)
    ex_keys = list(existing.keys())
    print(f"Keays: {ex_keys}")
    print(f"Samples: {existing['full_X1'].shape[0]:,}")

    d_existing = existing["full_X1"].shape[1]
    print(f"Features Dimension: {d_existing}")

    _hp_raw = existing["heston_params"] if "heston_params" in ex_keys \
                    else np.array([2.0, 0.04, 0.3, -0.7])
    heston_params = _extract_heston_params(_hp_raw)
    S0_current = float(existing.get("S0", np.array(5000.0)))
    r_current = float(existing.get("r",  np.array(0.05)))
    q_current = float(existing.get("q",  np.array(0.015)))

    print(f"\n[2/4] Download Historical SPX/VIX since {start}...")
    df_raw = download_history(start=start)
    print(f"  Total de dias: {len(df_raw)}")

    # feature engineering 
    print(f"\n[3/4] Building features ({d_existing} dims)...")
    X_all, dates = compute_features(df_raw)

    # remove warmup (rolling windows)
    valid_mask = ~np.isnan(X_all).any(axis=1)
    valid_mask[:min_valid_rows] = False   
    X_valid = X_all[valid_mask]
    d_valid = [d for d, m in zip(dates, valid_mask) if m]
    print(f"  Linhas válidas (pós-warmup): {len(X_valid):,}  ({X_valid.shape[1]} features)")

    N = len(X_valid)
    if N < horizon + 10:
        raise ValueError(f"Dados históricos insuficientes: {N} linhas (min={horizon+10})")

    X_source_hist = X_valid[:N - horizon].copy()
    X_target_hist = X_valid[horizon:].copy()
    N_pairs = len(X_source_hist)

    fetch_dates_hist = np.array([str(d.date()) for d in d_valid[:N - horizon]])

    theta_vol = np.sqrt(heston_params[1]) if len(heston_params) > 1 else 0.20
    vol_col = X_source_hist[:, 1].clip(1e-4, None)   
    log_ratio = np.log(vol_col / (theta_vol + 1e-8))
    log_w = -0.5 * log_ratio ** 2
    w = np.exp(log_w - log_w.max())
    w = np.clip(w, 0.1, None)   
    w = np.clip(w, 1e-6, None)
    weights_hist = (w / w.mean()).astype(np.float32)

    print(f"  Pares históricos gerados: {N_pairs:,}")
    print(f"  Período: {fetch_dates_hist[0]} → {fetch_dates_hist[-1]}")

    print(f"\n[4/4] Saving")

    X_src_existing = existing["full_X1"].astype(np.float32)
    X_tgt_existing = existing["X_target"].astype(np.float32) if "X_target" in ex_keys \
                     else X_src_existing   # fallback

    OLD_TO_NEW_COLS = [4, 6, 8, 14, 11, 13, 18, 17, 22, 27, 3, 23]
    N_NEW = 12

    def _remap_to_12(X: np.ndarray) -> np.ndarray:
        if X.shape[1] == N_NEW:
            return X
        if X.shape[1] >= 30:
            print(f"  [schema]  {X.shape[1]} → {N_NEW} features")
            return X[:, OLD_TO_NEW_COLS].copy()
        print(f"  [warning] unexpected schema {X.shape[1]} colunas — {N_NEW}")
        out = np.zeros((len(X), N_NEW), dtype=np.float32)
        n_copy = min(X.shape[1], N_NEW)
        out[:, :n_copy] = X[:, :n_copy]
        return out

    X_src_existing = _remap_to_12(X_src_existing)
    X_tgt_existing = _remap_to_12(X_tgt_existing)

    W_existing = np.ones(len(X_src_existing), dtype=np.float32)
    if "heston_params" in ex_keys:
        vol_ex = X_src_existing[:, min(1, X_src_existing.shape[1]-1)].clip(1e-4, None)
        log_ratio = np.log(vol_ex / (theta_vol + 1e-8))
        lw_ex = -0.5 * log_ratio ** 2
        w_ex = np.exp(lw_ex - lw_ex.max())
        w_ex = np.clip(w_ex, 0.1, None)
        W_existing = (w_ex / w_ex.mean()).astype(np.float32)

    X_src_full = np.concatenate([X_source_hist, X_src_existing], axis=0)
    X_tgt_full = np.concatenate([X_target_hist, X_tgt_existing], axis=0)
    W_full = np.concatenate([weights_hist,  W_existing],      axis=0)

    fd_existing = existing.get("fetch_date", np.array(["current"] * len(X_src_existing)))
    _fd_ex = np.array(fd_existing, dtype=str).flatten()
    if _fd_ex.ndim == 0 or len(_fd_ex) == 0:
        _fd_ex = np.array(["current"] * len(X_src_existing), dtype=str)
    fd_full = np.concatenate([fetch_dates_hist, _fd_ex], axis=0)

    print(f"Amostras históricas: {N_pairs:,}")
    print(f"Amostras existentes: {len(X_src_existing):,}")
    print(f"Total combinado: {len(X_src_full):,}  ({len(X_src_full)/len(X_src_existing):.1f}x aumento)")

    feat_names_diag = [
        "rv5", "rv21", "rv63",
        "log_vix", "vix_term", "vix_rv_spread",
        "skew_proxy", "parkinson_5d", "vol_regime",
        "rv_accel", "ret_21d", "dd_21",
    ][:X_src_full.shape[1]]
    stds = X_src_full.std(axis=0)
    n_diag  = len(feat_names_diag)  
    low_var = [(feat_names_diag[i], stds[i]) for i in range(n_diag) if stds[i] < 0.01]
    const   = [(feat_names_diag[i], stds[i]) for i in range(n_diag) if stds[i] < 1e-4]
    if const:
        print(f"\n  Constant features (std<1e-4): {[n for n,_ in const]}")
    elif low_var:
        print(f"\n  Low variance features (std<0.01): {[(n,f'{s:.5f}') for n,s in low_var]}")

    save_dict = {
        "full_X1": X_src_full,
        "X_target": X_tgt_full,
        "weights": W_full,
        "fetch_date": fd_full,
        "heston_params": heston_params,
        "S0": S0_current,
        "r": r_current,
        "q": q_current,
        "n_historical": np.array(N_pairs),
        "n_live": np.array(len(X_src_existing)),
        "hist_start": np.array(start),
        "feature_names": np.array([
            "rv5", "rv21", "rv63",
            "log_vix", "vix_term", "vix_rv_spread",
            "skew_proxy", "parkinson_5d", "vol_regime",
            "rv_accel", "ret_21d", "dd_21",
        ]),
    }

    # preserva chaves extras do NPZ original
    for k in ex_keys:
        if k not in save_dict:
            save_dict[k] = existing[k]

    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **save_dict)
    print(f"\n Salvo em: {output_npz}")
    print(f"full_X1: {X_src_full.shape}")
    print(f"X_target: {X_tgt_full.shape}")
    print(f"weights: min={W_full.min():.3f}  max={W_full.max():.3f}")
    eff_N = int(min(1 / (np.mean((W_full / W_full.mean()) ** 2) + 1e-9), len(W_full)))
    print(f"eff_N: {eff_N:,}")
    print("\nDataset aumentado ✓")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Expande live_spx_data.npz com histórico SPX")
    parser.add_argument("--npz", default=DEFAULT_NPZ, help="NPZ existente (entrada)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="NPZ aumentado (saída)")
    parser.add_argument("--start", default=DEFAULT_START, help="Data inicial do histórico (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Horizon em dias úteis para X_target")
    args = parser.parse_args()

    build_extended_dataset(
        existing_npz = args.npz,
        output_npz = args.out,
        start = args.start,
        horizon = args.horizon,
    )


if __name__ == "__main__":
    main()
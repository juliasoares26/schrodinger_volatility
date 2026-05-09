import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MONEYNESS_GRID = np.linspace(0.80, 1.20, 21)   # K/S0
MATURITY_GRID  = np.array([0.25, 0.50, 0.75, 1.00])
N_T = len(MATURITY_GRID)
N_K = len(MONEYNESS_GRID)
LOG_M = np.log(MONEYNESS_GRID)   

ATM_IDX = N_K // 2   # 10

CBOE_TICKERS = {
    "^VIX9D":  9  / 365,
    "^VIX":   30  / 365,
    "^VIX3M": 93  / 365,   # ≈ 0.255Y
    "^VIX6M": 180 / 365,   # ≈ 0.493Y
    "^VIX1Y": 365 / 365,
}
SKEW_TICKER = "^SKEW"


def download_cboe_indices(start: str, end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    all_tickers = list(CBOE_TICKERS.keys()) + [SKEW_TICKER]
    col_names   = ["vix9d", "vix", "vix3m", "vix6m", "vix1y", "skew"]

    dfs = {}
    for ticker, col in zip(all_tickers, col_names):
        print(f"  {ticker:8s}...", end=" ", flush=True)
        try:
            raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
            if raw.empty:
                print("VAZIO")
                continue
            s = raw["Close"].copy()
            s.index = pd.to_datetime(s.index).normalize()
            dfs[col] = s
            print(f"OK ({len(s)} dias)")
        except Exception as e:
            print(f"ERRO: {e}")

    if "vix" not in dfs:
        raise ValueError("VIX unavailable")

    df = pd.DataFrame(dfs).sort_index()

    for col in ["vix9d", "vix", "vix3m", "vix6m", "vix1y"]:
        if col in df.columns:
            df[col] = df[col] / 100.0

    if "skew" not in df.columns:
        df["skew"] = 120.0   

    df = df.ffill().dropna(subset=["vix"])

    print(f"  Total dias válidos: {len(df)}")
    return df

_TS_SLOPE_PRIOR = 0.015   

def _interp_atm(row: pd.Series) -> np.ndarray:
    nodes_T  = []
    nodes_iv = []

    ticker_map = [
        ("vix9d",  9  / 365),
        ("vix",   30  / 365),
        ("vix3m", 93  / 365),
        ("vix6m", 180 / 365),
        ("vix1y", 365 / 365),
    ]
    for col, T in ticker_map:
        v = row.get(col, np.nan)
        if not np.isnan(v) and v > 0.01:
            nodes_T.append(T)
            nodes_iv.append(v)

    if len(nodes_T) < 2:
        v_base = nodes_iv[0] if nodes_iv else 0.20
        sqrt_base = np.sqrt(30 / 365)
        atm_iv = np.array([
            v_base + _TS_SLOPE_PRIOR * (np.sqrt(T) - sqrt_base)
            for T in MATURITY_GRID
        ], dtype=np.float32)
        return np.clip(atm_iv, 0.02, 2.0)

    nodes_T = np.array(nodes_T)
    nodes_iv = np.array(nodes_iv)
    sqrt_nodes  = np.sqrt(nodes_T)
    sqrt_target = np.sqrt(MATURITY_GRID)

    n_nodes = len(nodes_T)

    d_sqrt = sqrt_nodes[-1] - sqrt_nodes[-2]
    d_iv = nodes_iv[-1]  - nodes_iv[-2]
    obs_slope = d_iv / (d_sqrt + 1e-8)

    if n_nodes < 3:
        blend = 0.5  
        ext_slope = blend * obs_slope + (1.0 - blend) * _TS_SLOPE_PRIOR
    else:
        ext_slope = obs_slope   

    atm_iv = np.interp(sqrt_target, sqrt_nodes, nodes_iv,
                       left  = nodes_iv[0],   
                       right = nodes_iv[-1])  
    
    sqrt_last = sqrt_nodes[-1]
    iv_last   = nodes_iv[-1]
    for i, (sq_t, T) in enumerate(zip(sqrt_target, MATURITY_GRID)):
        if sq_t > sqrt_last:
            atm_iv[i] = iv_last + ext_slope * (sq_t - sqrt_last)

    atm_iv = np.clip(atm_iv, 0.02, 2.0)

    return atm_iv.astype(np.float32)


def _skew_to_rho(skew_index: float) -> float:
    rho = -0.035 * (skew_index - 100.0)
    return float(np.clip(rho, -0.97, 0.0))


def _svi_smile(atm_iv: float, T: float, rho: float, log_m: np.ndarray) -> np.ndarray:
    atm_var = atm_iv ** 2 * T       

    b   = 0.35 * atm_iv * np.sqrt(T)
    sigma_svi = max(atm_iv * np.sqrt(T) * 0.5, 1e-4)  

    a = atm_var - b * sigma_svi

    k = log_m   # (21,)
    discriminant = np.sqrt((k) ** 2 + sigma_svi ** 2)
    w = a + b * (rho * k + discriminant)   

    w   = np.clip(w, 1e-8, None)
    iv  = np.sqrt(w / (T + 1e-8))
    iv  = np.clip(iv, 0.02, 2.0)
    return iv.astype(np.float32)


def build_daily_surface(row: pd.Series) -> np.ndarray:
    atm_ivs = _interp_atm(row)                     # (4,)
    rho     = _skew_to_rho(float(row.get("skew", 120.0)))

    surface = np.empty((N_T, N_K), dtype=np.float32)
    for i, (T, atm_iv) in enumerate(zip(MATURITY_GRID, atm_ivs)):
        surface[i] = _svi_smile(float(atm_iv), float(T), rho, LOG_M)

    return surface


def build_surface_history(
    start: str = "2019-01-01",
    end:   str | None = None,
    min_valid_rows: int = 63,
) -> tuple[np.ndarray, list]:
    df = download_cboe_indices(start, end)

    df = df.iloc[min_valid_rows:].copy()

    print(f"\n Building {len(df)} daily surfaces...")
    surfaces = np.stack([build_daily_surface(row) for _, row in df.iterrows()], axis=0)
    dates    = [idx.date() for idx in df.index]

    # diagnóstico
    atm_vals = surfaces[:, :, ATM_IDX]   # (N, 4)
    print(f"ATM IV range (0.25Y): [{atm_vals[:,0].min():.4f}, {atm_vals[:,0].max():.4f}]")
    print(f"ATM IV range (1.00Y): [{atm_vals[:,3].min():.4f}, {atm_vals[:,3].max():.4f}]")
    print(f"ATM IV mean  (0.25Y): {atm_vals[:,0].mean():.4f}")
    print(f"Skew médio (0.25Y):   {(surfaces[:,0,0] - surfaces[:,0,-1]).mean():.4f}")

    ts_spread = atm_vals[:, 3] - atm_vals[:, 0]  
    print(f"\n Term structure (1.00Y - 0.25Y):")
    print(f"mean: {ts_spread.mean():+.4f}")
    print(f"std: {ts_spread.std():.4f}")
    print(f"p5: {np.percentile(ts_spread,  5):+.4f}")
    print(f"p95: {np.percentile(ts_spread, 95):+.4f}")
    ts_collapsed = np.mean(np.abs(ts_spread) < 0.001)
    if ts_collapsed > 0.10:
        print(f"{ts_collapsed*100:.1f}% of days with collapsed term structure "
          f"(|spread|<0.1pp) — VIX6M/VIX1Y with excessive gaps.")
    else:
        print(f"Term structure varying adequately "
          f"({ts_collapsed*100:.1f}% collapsed days)")
    return surfaces.astype(np.float32), dates


def save_surface_history(
    surfaces: np.ndarray,
    dates:    list,
    out_path: str,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        surfaces = surfaces,
        dates = np.array([str(d) for d in dates]),
        moneyness_grid = MONEYNESS_GRID,
        maturity_grid = MATURITY_GRID,
    )
    print(f"\n saved: {out_path}")
    print(f"surfaces: {surfaces.shape}  dtype={surfaces.dtype}")
    print(f"period:  {dates[0]} → {dates[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Builds a real historical SPX IV surface using CBOE subindices"
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default=r"C:\volatility-options\data\surface_history.npz")
    args = parser.parse_args()

    surfaces, dates = build_surface_history(start=args.start, end=args.end)
    save_surface_history(surfaces, dates, args.out)

if __name__ == "__main__":
    main()
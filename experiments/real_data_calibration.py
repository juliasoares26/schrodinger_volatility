from __future__ import annotations

import argparse
import logging
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("real_data_calibration")

# Path setup — works from project root or from experiments/

_THIS_FILE   = Path(__file__).resolve()
_SCRIPT_DIR  = _THIS_FILE.parent
_PROJECT_ROOT = (
    _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "experiments" else _SCRIPT_DIR
)

# Canonical directory layout
_DIR_RAW_OPTIONS   = _PROJECT_ROOT / "data" / "raw"    / "options"
_DIR_RAW_EQUITY    = _PROJECT_ROOT / "data" / "raw"    / "equity"
_DIR_PROC_SURFACES = _PROJECT_ROOT / "data" / "processed" / "surfaces"
_DIR_PROC_PRED     = _PROJECT_ROOT / "data" / "processed" / "prediction"
_DIR_LIVE          = _PROJECT_ROOT / "data"

for _d in [_DIR_RAW_OPTIONS, _DIR_RAW_EQUITY, _DIR_PROC_SURFACES, _DIR_PROC_PRED]:
    _d.mkdir(parents=True, exist_ok=True)

# Make sibling modules importable regardless of CWD
import sys
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "data_pipeline" / "downloaders",
           _PROJECT_ROOT, _PROJECT_ROOT / "data_pipeline" ,
           _PROJECT_ROOT / "experiments"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# Public entry point (called by pipeline.py)

def run_pipeline(
    source: str  = "yfinance",
    cboe_csv_path: str  = None,
    history_start: str  = "2020-01-01",
    lookback: int  = 20,
    horizon: int  = 5,
    run_plots: bool = True,
    output_filename: str  = "live_spx_data.npz",
) -> Path:
    today_str = date.today().strftime("%Y%m%d")
    log.info(f"  real_data_calibration  source={source}  date={today_str}")

    # Step 1 — option chain
    log.info("[1/4] Loading option chain …")
    df_raw, S0, r, q, fetch_date = _load_options(source, cboe_csv_path)

    raw_options_path = _DIR_RAW_OPTIONS / f"spx_{today_str}.parquet"
    df_raw.to_parquet(raw_options_path, index=False)
    log.info(f"  Saved raw options → {raw_options_path}  ({len(df_raw)} rows)")

    # Step 2 — price history
    log.info("[2/4] Loading SPX price history …")
    prices = _load_prices(history_start)

    prices_path = _DIR_RAW_EQUITY / "spx_prices.parquet"
    prices.to_frame("close").to_parquet(prices_path)
    log.info(f"  Saved prices → {prices_path}  ({len(prices)} days)")

    # Step 3 — IV surface
    log.info("[3/4] Building IV surface …")
    vol_surface, strikes_norm, maturities, coverage, n_clean = _build_surface(
        df_raw, S0
    )

    surface_path = _DIR_PROC_SURFACES / f"spx_surface_{today_str}.npz"
    np.savez(
        str(surface_path),
        vol_surface = vol_surface,
        strikes_norm  = strikes_norm,
        maturities  = maturities,
        S0  = S0,
        r = r,
        q = q,
        fetch_date = str(fetch_date),
        surface_coverage = coverage,
        n_raw_options = len(df_raw),
        n_filtered_options = n_clean,
    )
    log.info(f"  Saved surface → {surface_path}  (coverage {coverage*100:.1f}%)")

    if run_plots:
        _plot_surface(vol_surface, strikes_norm, maturities,
                      _DIR_PROC_SURFACES / f"spx_surface_{today_str}.png")

    # Step 4 — prediction features
    log.info("[4/4] Building prediction features …")
    from feature_engineer import build_prediction_features, estimate_heston_params

    features = build_prediction_features(
        prices, vol_surface, strikes_norm, maturities,
        lookback=lookback, horizon=horizon, r=r, q=q,
    )

    heston_params = estimate_heston_params(prices, r=r, q=q)
    heston_params["S0"] = S0          # override with live spot

    features_path = _DIR_PROC_PRED / f"spx_features_{today_str}.npz"
    np.savez(
        str(features_path),
        full_X1 = features["X1"],
        full_X2 = features["X2"],
        full_X1_raw = features["X1_raw"],
        full_X1_mean = features["X1_mean"],
        full_X1_std = features["X1_std"],
        full_lookback = features["lookback"],
        full_horizon = features["horizon"],
        vol_surface = vol_surface,
        strikes_norm = strikes_norm,
        maturities = maturities,
        heston_params = heston_params,
        S0 = S0,
        r = r,
        q = q,
        fetch_date = str(fetch_date),
        surface_coverage = coverage,
        n_raw_options = len(df_raw),
        n_filtered_options = n_clean,
        source = source,
        method = "real_spx_data",
        description   = (
            f"Real SPX data pipeline — source={source} "
            f"date={today_str} lookback={lookback} horizon={horizon}"
        ),
    )
    log.info(f"  Saved features → {features_path}  ({len(features['X1'])} samples)")

    # Step 5 - unified live snapshot 
    live_path = _DIR_LIVE / "live_spx_data.npz"
    import shutil
    shutil.copy(str(features_path), str(live_path))
    log.info(f"  Live snapshot → {live_path}")

    if output_filename != "live_spx_data.npz":
        custom_path = _DIR_LIVE / output_filename
        shutil.copy(str(features_path), str(custom_path))
        log.info(f"  Custom snapshot → {custom_path}")

    _print_summary(S0, r, q, coverage, features, heston_params, today_str)
    return live_path


# Step helpers

def _load_options(source: str, cboe_csv_path: str = None):
    if source == "yfinance":
        from yfinance_loader import load_spx_yfinance
        return load_spx_yfinance(min_dte=7, max_dte=400)

    elif source == "cboe_csv":
        if cboe_csv_path is None:
            raise ValueError("--cboe_csv required when source=cboe_csv")
        from cboe import load_spx_cboe_csv
        return load_spx_cboe_csv(cboe_csv_path)

    elif source == "ibkr":
        from ibkr import IBKRLoader
        loader = IBKRLoader()
        result = loader.load_spx_options(min_dte=7, max_dte=400)
        loader.disconnect()
        return result

    else:
        raise ValueError(f"Unknown source '{source}'. Choose: yfinance | cboe_csv | ibkr")


def _load_prices(history_start: str) -> pd.Series:
    from yfinance_loader import load_spx_history
    return load_spx_history(start=history_start)


def _build_surface(df_raw: pd.DataFrame, S0: float):
    from surface_builder import (
        apply_standard_filters,
        select_surface_grid,
        build_iv_surface,
        MONEYNESS_GRID,
        MATURITY_GRID,
    )
    from data_pipeline.cleaning.arbitrage_filters import filter_surface  # optional

    df_filt, _ = apply_standard_filters(df_raw, S0)
    df_grid    = select_surface_grid(df_filt)
    vol_raw, coverage = build_iv_surface(df_grid)

    # Optional arbitrage filter — skip gracefully if module not present
    try:
        _, vol_surf, _ = filter_surface(df_filt, vol_raw, MATURITY_GRID, MONEYNESS_GRID)
    except Exception:
        log.warning("  arbitrage_filters not available — using raw surface")
        vol_surf = vol_raw

    # Fill any remaining NaNs with ATM mean
    atm_fill = float(np.nanmean(vol_surf))
    vol_surf  = np.where(np.isnan(vol_surf), atm_fill, vol_surf)

    return vol_surf, MONEYNESS_GRID, MATURITY_GRID, coverage, len(df_filt)


def _plot_surface(vol_surface, strikes_norm, maturities, save_path: Path):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, T in enumerate(maturities):
            ax.plot(strikes_norm, vol_surface[i] * 100, label=f"T={T:.2f}Y", linewidth=1.8)
        ax.set_xlabel("Moneyness (K/S0)")
        ax.set_ylabel("Implied Volatility (%)")
        ax.set_title("SPX IV Surface (real data)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Plot saved → {save_path.name}")
    except Exception as e:
        log.warning(f"  Plot skipped: {e}")


def _print_summary(S0, r, q, coverage, features, heston_params, today_str):
    X1, X2 = features["X1"], features["X2"]
    log.info("\n" + "=" * 60)
    log.info("Pipeline complete")
    log.info(f"Date: {today_str}")
    log.info(f"SPX spot (S0): {S0:.2f}")
    log.info(f"Risk-free (r): {r*100:.3f}%")
    log.info(f"Div yield (q): {q*100:.3f}%")
    log.info(f"Surface cov.: {coverage*100:.1f}%")
    log.info(f"Samples (X1): {len(X1)}  dim={X1.shape[1]}")
    log.info(f"Return mean: {X2[:,0].mean():.5f}  std={X2[:,0].std():.5f}")
    log.info(f"Future vol: {X2[:,1].mean():.4f}")
    log.info(f"Heston kappa: {heston_params['kappa']:.4f}")
    log.info(f"Heston theta: {heston_params['theta']:.4f}")
    log.info(f"Heston rho: {heston_params['rho']:.4f}")
    log.info("=" * 60)


# CLI

def _parse_args():
    p = argparse.ArgumentParser(
        description="Real-data SPX pipeline → data/processed/ and data/live_spx_data.npz"
    )
    p.add_argument("--source",        default="yfinance",
                   choices=["yfinance", "cboe_csv", "ibkr"],
                   help="Option data source (default: yfinance)")
    p.add_argument("--cboe_csv",      default=None,
                   help="Path to CBOE CSV file (required if source=cboe_csv)")
    p.add_argument("--history_start", default="2020-01-01",
                   help="Start date for price history (default: 2020-01-01)")
    p.add_argument("--lookback",      type=int, default=20,
                   help="Feature lookback window in trading days (default: 20)")
    p.add_argument("--horizon",       type=int, default=5,
                   help="Forecast horizon in trading days (default: 5)")
    p.add_argument("--no_plots",      action="store_true",
                   help="Skip plot generation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        source = args.source,
        cboe_csv_path = args.cboe_csv,
        history_start = args.history_start,
        lookback = args.lookback,
        horizon = args.horizon,
        run_plots = not args.no_plots,
    )
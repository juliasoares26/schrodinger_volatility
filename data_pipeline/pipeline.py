import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_pipeline")

# Fixed grid — must match surface_builder.py and feature_engineer.py
from surface_builder  import MONEYNESS_GRID, MATURITY_GRID

DEFAULT_STATE_PATH = Path("data/pipeline_state.json")
DEFAULT_DATA_PATH  = Path("data/live_spx_data.npz")

# DataSnapshot - unified output

@dataclass
class DataSnapshot:
    # meta
    timestamp: str
    source: str
    is_stale: bool = False

    # market
    S0: float = 0.0
    r: float = 0.0
    q: float = 0.0

    # IV surface  (n_maturities × n_strikes)
    vol_surface: Optional[np.ndarray] = None
    strikes_norm: Optional[np.ndarray] = None
    maturities: Optional[np.ndarray] = None

    # prediction features
    X1: Optional[np.ndarray] = None   # normalised
    X2: Optional[np.ndarray] = None   # targets
    X1_raw: Optional[np.ndarray] = None
    lookback: int = 20
    horizon: int = 5

    # Heston params
    heston_params: Optional[Dict] = None

    # diagnostics
    n_options_raw: int = 0
    n_options_clean:int = 0
    surface_coverage: float = 0.0
    fetch_duration_s: float = 0.0

# PipelineState: persisted between run

@dataclass
class PipelineState:
    last_fetch_utc: str = ""
    last_S0: float = 0.0
    last_r: float = 0.0
    last_q: float = 0.0
    n_price_bars: int = 0
    history_start: str = "2020-01-01"
    data_path: str = str(DEFAULT_DATA_PATH)

    def save(self, path: Path = DEFAULT_STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> "PipelineState":
        if not path.exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        obj = cls()
        obj.__dict__.update(d)
        return obj

# LiveDataPipeline

class LiveDataPipeline:

    def __init__(
        self,
        source: str  = "yfinance",
        cboe_csv_path: Optional[str] = None,
        history_start: str  = "2020-01-01",
        lookback: int  = 20,
        horizon: int  = 5,
        state_path: Path = DEFAULT_STATE_PATH,
        data_path: Path = DEFAULT_DATA_PATH,
        min_refresh_s: int  = 60,
    ):
        self.source = source
        self.cboe_csv_path = cboe_csv_path
        self.history_start = history_start
        self.lookback = lookback
        self.horizon = horizon
        self.state_path = Path(state_path)
        self.data_path = Path(data_path)
        self.min_refresh_s = min_refresh_s

        self._state: PipelineState = PipelineState.load(self.state_path)
        self._last_snapshot: Optional[DataSnapshot] = None

    # Public API

    def refresh(self, force: bool = False) -> DataSnapshot:
        """Return a fresh DataSnapshot, using cache when recently fetched"""
        t0 = time.perf_counter()

        if not force and self._last_snapshot is not None:
            elapsed = self._seconds_since_last_fetch()
            if elapsed < self.min_refresh_s:
                log.info(f"Returning cached snapshot (last fetch {elapsed:.0f}s ago)")
                return self._last_snapshot

        needs_bootstrap = (
            not self.data_path.exists()
            or self._state.last_fetch_utc == ""
            or self._state.n_price_bars < self.lookback + self.horizon + 10
        )

        if needs_bootstrap:
            log.info("Bootstrap: full pipeline run")
            snap = self._full_pipeline()
        else:
            log.info("Incremental update")
            snap = self._incremental_update()

        snap.fetch_duration_s = time.perf_counter() - t0
        self._last_snapshot   = snap
        return snap

    def get_latest_surface(self) -> Optional[np.ndarray]:
        return (self._last_snapshot or self.refresh()).vol_surface

    def get_feature_matrix(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        snap = self._last_snapshot or self.refresh()
        return snap.X1, snap.X2

    def get_heston_params(self) -> Optional[Dict]:
        snap = self._last_snapshot or self.refresh()
        return snap.heston_params

    # Full bootstrap

    def _full_pipeline(self) -> DataSnapshot:
        try:
            from real_data_calibration import run_pipeline
            run_pipeline(
                source = self.source,
                cboe_csv_path = self.cboe_csv_path,
                history_start = self.history_start,
                lookback = self.lookback,
                horizon = self.horizon,
                run_plots = False,
                output_filename = self.data_path.name,
            )
            snap = self._load_from_cache()
            snap.source = self.source
            self._update_state(snap)
            return snap
        except Exception as e:
            log.warning(f"Full pipeline failed: {e}. Trying cache.")
            return self._load_from_cache()

    # Incremental update

    def _incremental_update(self) -> DataSnapshot:
        try:
            snap = self._load_from_cache()
            snap = self._refresh_surface(snap)
            snap = self._append_price_bar(snap)
            self._update_state(snap)
            self._save_snapshot(snap)
            return snap
        except Exception as e:
            log.warning(f"Incremental update failed ({e}). Returning stale cache.")
            snap = self._load_from_cache()
            snap.is_stale = True
            return snap

    def _refresh_surface(self, snap: DataSnapshot) -> DataSnapshot:
        try:
            from data_pipeline.downloaders import load_spx_yfinance, load_spx_history
            from data_pipeline.downloaders.cboe import load_spx_cboe_csv
            from data_pipeline.surface_builder  import (
                apply_standard_filters, select_surface_grid, build_iv_surface,
            )
            from data_pipeline.cleaning.arbitrage_filters import filter_surface

            if self.source == "yfinance":
                df_raw, S0, r, q, _ = load_spx_yfinance(min_dte=7, max_dte=400)
            elif self.source == "cboe_csv":
                df_raw, S0, r, q, _ = load_spx_cboe_csv(self.cboe_csv_path)
            else:
                raise ValueError(f"Unknown source '{self.source}'")

            df_filt, _ = apply_standard_filters(df_raw, S0)
            df_grid    = select_surface_grid(df_filt)
            vol_raw, coverage = build_iv_surface(df_grid)
            _, vol_surf, _    = filter_surface(df_filt, vol_raw, MATURITY_GRID, MONEYNESS_GRID)

            atm_fill = float(np.nanmean(vol_surf))
            vol_surf = np.where(np.isnan(vol_surf), atm_fill, vol_surf)

            snap.S0 = S0
            snap.r = r
            snap.q = q
            snap.vol_surface = vol_surf
            snap.n_options_raw = len(df_raw)
            snap.n_options_clean = len(df_filt)
            snap.surface_coverage = coverage
            snap.timestamp = datetime.utcnow().isoformat()
            snap.source = self.source
            log.info(f"Surface refreshed — S0={S0:.2f}, coverage={coverage*100:.1f}%")
        except Exception as e:
            log.warning(f"Surface refresh failed: {e}")
            snap.is_stale = True
        return snap

    def _append_price_bar(self, snap: DataSnapshot) -> DataSnapshot:
        try:
            from data_pipeline.downloaders import load_spx_history
            from data_pipeline.feature_engineer import build_prediction_features

            prices = load_spx_history(
                start = self.history_start,
                end   = date.today().strftime("%Y-%m-%d"),
            )
            if len(prices) < self.lookback + self.horizon + 5:
                return snap

            result = build_prediction_features(
                prices, snap.vol_surface, MONEYNESS_GRID, MATURITY_GRID,
                lookback=self.lookback, horizon=self.horizon,
                r=snap.r, q=snap.q,
            )
            snap.X1 = result["X1"]
            snap.X2 = result["X2"]
            snap.X1_raw = result["X1_raw"]
            log.info(f"Features updated — {snap.X1.shape[0]} samples, dim={snap.X1.shape[1]}")
        except Exception as e:
            log.warning(f"Price bar append failed: {e}")
        return snap

    # Cache helpers

    def _load_from_cache(self) -> DataSnapshot:
        path = self.data_path
        if not path.exists():
            # Search order: live real data first, then per-date processed files,
            # then synthetic fallback (last resort — clearly labelled as stale)
            today_str = date.today().strftime("%Y%m%d")
            candidates = [
                Path("data/live_spx_data.npz"),
                Path("../data/live_spx_data.npz"),
                Path(f"data/processed/prediction/spx_features_{today_str}.npz"),
                Path(f"../data/processed/prediction/spx_features_{today_str}.npz"),
                Path("data/synthetic/unified_heston_prediction_data.npz"),
                Path("../data/synthetic/unified_heston_prediction_data.npz"),
                Path("data/unified_heston_prediction_data.npz"),
                Path("../data/unified_heston_prediction_data.npz"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    log.info(f"Using fallback: {path}")
                    break
            else:
                raise FileNotFoundError(
                    "No cached data found. Run pipeline.refresh(force=True) "
                    "or: python experiments/real_data_calibration.py"
                )

        data = np.load(path, allow_pickle=True)
        hp = data["heston_params"].item() if "heston_params" in data else {}
        S0 = float(hp.get("S0", 100.0))
        r = float(hp.get("r",   0.0))
        q = float(data["q"]) if "q" in data else 0.015

        # Flag synthetic fallback so callers know the data is not real
        source_tag = str(data.get("source", "cache"))
        is_stale   = "synthetic" in str(path) or source_tag in ("", "cache")

        return DataSnapshot(
            timestamp = str(data.get("fetch_date", datetime.utcnow().isoformat())),
            source = source_tag,
            is_stale = is_stale,
            S0 = S0, r=r, q=q,
            vol_surface = data["vol_surface"],
            strikes_norm = data["strikes_norm"],
            maturities = data["maturities"],
            X1 = data["full_X1"],
            X2 = data["full_X2"],
            X1_raw  = data["full_X1_raw"],
            lookback = int(data["full_lookback"]),
            horizon = int(data["full_horizon"]),
            heston_params = hp,
            surface_coverage = float(data.get("surface_coverage", 1.0)),
            n_options_raw = int(data.get("n_raw_options", 0)),
            n_options_clean = int(data.get("n_filtered_options", 0)),
        )

    def _save_snapshot(self, snap: DataSnapshot) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        hp = snap.heston_params or {}
        np.savez(
            str(self.data_path),
            vol_surface = snap.vol_surface,
            strikes_norm = snap.strikes_norm  if snap.strikes_norm  is not None else MONEYNESS_GRID,
            maturities = snap.maturities if snap.maturities    is not None else MATURITY_GRID,
            full_X1 = snap.X1,
            full_X2 = snap.X2,
            full_X1_raw = snap.X1_raw,
            full_X1_mean = snap.X1_raw.mean(axis=0) if snap.X1_raw is not None else np.zeros(1),
            full_X1_std = snap.X1_raw.std(axis=0) + 1e-8 if snap.X1_raw is not None else np.ones(1),
            full_lookback = snap.lookback,
            full_horizon = snap.horizon,
            heston_params = hp,
            S0 = snap.S0,
            r = snap.r,
            q = snap.q,
            fetch_date = snap.timestamp,
            source = snap.source,
            surface_coverage = snap.surface_coverage,
            n_raw_options = snap.n_options_raw,
            n_filtered_options = snap.n_options_clean,
        )
        log.info(f"Snapshot saved → {self.data_path}")

    # State helpers

    def _update_state(self, snap: DataSnapshot) -> None:
        self._state.last_fetch_utc = snap.timestamp
        self._state.last_S0 = snap.S0
        self._state.last_r = snap.r
        self._state.last_q = snap.q
        self._state.n_price_bars = len(snap.X1) if snap.X1 is not None else 0
        self._state.history_start = self.history_start
        self._state.data_path = str(self.data_path)
        self._state.save(self.state_path)

    def _seconds_since_last_fetch(self) -> float:
        if not self._state.last_fetch_utc:
            return float("inf")
        try:
            last = datetime.fromisoformat(self._state.last_fetch_utc)
            return (datetime.utcnow() - last).total_seconds()
        except Exception:
            return float("inf")

    # Diagnostics

    def status(self) -> Dict:
        snap  = self._last_snapshot
        stale = self._seconds_since_last_fetch()
        return {
            "last_fetch": self._state.last_fetch_utc or "never",
            "seconds_since": round(stale, 1),
            "is_stale": snap.is_stale if snap else True,
            "S0": self._state.last_S0,
            "r_pct": round(self._state.last_r * 100, 3),
            "q_pct": round(self._state.last_q * 100, 3),
            "n_price_bars": self._state.n_price_bars,
            "data_path_exists": self.data_path.exists(),
        }

    def print_status(self) -> None:
        s = self.status()
        print("\n─Live Pipeline Status")
        for k, v in s.items():
            print(f"  {k:<22}: {v}")
# Module-level convenienc

_default_pipeline: Optional[LiveDataPipeline] = None


def get_pipeline(**kwargs) -> LiveDataPipeline:
    global _default_pipeline
    if _default_pipeline is None or kwargs:
        _default_pipeline = LiveDataPipeline(**kwargs)
    return _default_pipeline


def get_snapshot(force: bool = False, **kwargs) -> DataSnapshot:
    """One-liner: get the latest DataSnapshot"""
    return get_pipeline(**kwargs).refresh(force=force)
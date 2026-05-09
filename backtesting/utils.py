from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

COVID_PERIOD: List[Tuple[str, str]] = [("2020-02-20", "2020-06-30")]

def is_excluded(
    date_str: str,
    exclude_periods: Optional[List[Tuple[str, str]]],
) -> bool:
    if not exclude_periods:
        return False
    d = pd.Timestamp(date_str)
    return any(
        pd.Timestamp(s) <= d <= pd.Timestamp(e)
        for s, e in exclude_periods
    )


def apply_exclude_periods(
    index: pd.DatetimeIndex,
    exclude_periods: Optional[List[Tuple[str, str]]],
) -> pd.DatetimeIndex:
    if not exclude_periods:
        return index
    mask = pd.Series(True, index=index)
    for start, end in exclude_periods:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        mask &= ~((index >= s) & (index <= e))
    removed = int((~mask).sum())
    if removed:
        ranges = ", ".join(f"{s}–{e}" for s, e in exclude_periods)
        print(f"  [exclude_periods] {removed} dias removidos ({ranges})")
    return index[mask]


def fetch_spx_prices(hist_start: str, S0: float) -> pd.Series:
    import yfinance as yf
    spx = yf.download("^GSPC", start=hist_start, progress=False)["Close"].squeeze()
    spx.index = pd.to_datetime(spx.index)
    last = float(spx.iloc[-1])
    if abs(last - S0) / S0 > 0.05:
        print(f"  [WARN] S0 no NPZ ({S0:.1f}) difere yfinance ({last:.1f}) >5%")
    print(f"  SPX prices: {len(spx)} dias  "
          f"{spx.index[0].date()} -> {spx.index[-1].date()}")
    return spx


def unique_dates_from_npz(npz_path: str) -> list:
    raw  = np.load(npz_path, allow_pickle=True)
    seen = set()
    out  = []
    for d in raw["fetch_date"]:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def load_module_from_path(module_name: str, candidates: List[Path]):
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        pass

    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"{module_name}.py não encontrado. Tentativas: {[str(p) for p in candidates]}"
        )
    spec = importlib.util.spec_from_file_location(module_name, str(found))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"  {module_name} in {found}")
    return module


def load_surface_model(surface_model_path: str):
    sm_dir  = Path(surface_model_path).parent.parent / "models"
    module  = load_module_from_path(
        "surface_model",
        [sm_dir / "surface_model.py", Path("surface_model.py")],
    )
    sm = module.SurfaceModel()
    sm.load(surface_model_path)
    return sm


def load_brenier_estimator():
    sm_dir = Path(__file__).parent.parent / "models"
    module = load_module_from_path(
        "brenier",
        [sm_dir / "brenier.py", Path("models") / "brenier.py", Path("brenier.py")],
    )
    return module.ConditionalBrenierEstimator
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import time
import warnings

warnings.filterwarnings("ignore")

# COVID filter 
COVID_PERIOD: List[Tuple[str, str]] = [("2020-02-20", "2020-06-30")]


# Helpers 
def _fetch_spx_prices(hist_start: str, S0: float) -> pd.Series:
    import yfinance as yf
    spx = yf.download("^GSPC", start=hist_start, progress=False)["Close"].squeeze()
    spx.index = pd.to_datetime(spx.index)
    last = float(spx.iloc[-1])
    if abs(last - S0) / S0 > 0.05:
        print(f"  [WARN] S0 no NPZ ({S0:.1f}) difere yfinance ({last:.1f}) >5%")
    print(f"  SPX prices: {len(spx)} dias  "
          f"{spx.index[0].date()} -> {spx.index[-1].date()}")
    return spx


def _unique_dates_from_npz(npz_path: str) -> list:
    raw  = np.load(npz_path, allow_pickle=True)
    seen = set()
    out  = []
    for d in raw["fetch_date"]:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _load_surface_model(surface_model_path: str):
    import importlib.util
    _sm_dir  = Path(surface_model_path).parent.parent / "models"
    _sm_file = _sm_dir / "surface_model.py"
    if not _sm_file.exists():
        _sm_file = Path("surface_model.py")
    if not _sm_file.exists():
        raise FileNotFoundError(f"surface_model.py not found {_sm_dir}")
    spec   = importlib.util.spec_from_file_location("surface_model", str(_sm_file))
    sm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm_mod)
    sm = sm_mod.SurfaceModel()
    sm.load(surface_model_path)
    return sm


def _iv_to_call_price(S: float, K: float, T: float, iv: float, r: float = 0.0) -> float:
    from scipy.stats import norm
    if T <= 0 or iv <= 0:
        return max(S - K, 0.0)
    F  = S * np.exp(r * T)
    d1 = (np.log(F / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


# SPXSurfaceLoader 

class SPXSurfaceLoader:

    def __init__(
        self,
        npz_path: str,
        surface_model_path: str,
        target_maturity: float = 0.25,
        r: float = 0.0,
        exclude_periods: Optional[List[Tuple[str, str]]] = None,
    ):
        self.target_maturity = target_maturity
        self.r = r
        self.exclude_periods = exclude_periods or []
        self._snapshots: Dict[str, Dict] = {}
        self._load(npz_path, surface_model_path)

    def dates(self) -> List[str]:
        return sorted(self._snapshots.keys())

    def get(self, date_str: str) -> Optional[Dict]:
        return self._snapshots.get(date_str)

    def _is_excluded(self, date_str: str) -> bool:
        d = pd.Timestamp(date_str)
        return any(
            pd.Timestamp(s) <= d <= pd.Timestamp(e)
            for s, e in self.exclude_periods
        )

    def _load(self, npz_path: str, surface_model_path: str) -> None:
        print(f"\n SPXSurfaceLoader")
        print(f"NPZ: {npz_path}")
        print(f"SurfaceModel: {surface_model_path}")
        print(f"Maturidade: {self.target_maturity}Y")

        sm = _load_surface_model(surface_model_path)
        scores_full = sm.pc_history.astype(np.float64)

        raw = np.load(npz_path, allow_pickle=True)
        moneyness_grid = raw["strikes_norm"].astype(np.float64)
        maturity_grid = raw["maturities"].astype(np.float64)
        mat_idx = int(np.argmin(np.abs(maturity_grid - self.target_maturity)))
        T = float(maturity_grid[mat_idx])

        prices_full = _fetch_spx_prices(str(raw["hist_start"]), float(raw["S0"]))
        dates_unique = _unique_dates_from_npz(npz_path)

        n_align = min(len(dates_unique), len(scores_full))
        dates = dates_unique[-n_align:]
        scores = scores_full[-n_align:]

        prices_map: Dict[str, float] = {}
        for d in dates:
            try:
                prices_map[d] = float(prices_full.loc[pd.Timestamp(d)])
            except KeyError:
                pass

        P, M = len(maturity_grid), len(moneyness_grid)
        skipped = loaded = 0

        for date, score in zip(dates, scores):
            if self._is_excluded(date):
                skipped += 1
                continue
            S_t = prices_map.get(date)
            if S_t is None:
                continue

            surf_flat = sm.pca.inverse_transform(score[np.newaxis])
            iv_row = surf_flat.reshape(P, M)[mat_idx]
            strikes_abs = moneyness_grid * S_t

            market_prices = np.array([
                _iv_to_call_price(S_t, float(K), T, float(iv), self.r)
                for K, iv in zip(strikes_abs, iv_row)
            ], dtype=np.float64)

            self._snapshots[date] = {
                "strikes": strikes_abs,
                "market_prices": market_prices,
                "T": T,
                "S0": S_t,
                "iv_row": iv_row,
            }
            loaded += 1

        if skipped:
            print(f"Excluded: {skipped} dias")
        print(f"Snapshots: {loaded} dias  "
              f"({dates[0]} → {dates[-1]})")


# RollingRecord 
@dataclass
class RollingRecord:
    method_name: str
    date: str
    T: float
    strikes: np.ndarray
    market_prices: np.ndarray
    model_prices: np.ndarray
    params: np.ndarray
    mae: float
    rmse: float
    calibration_time: float
    success: bool
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "method": self.method_name,
            "date": self.date,
            "T": self.T,
            "mae": self.mae,
            "rmse": self.rmse,
            "calibration_time": self.calibration_time,
            "success": self.success,
        }


# RollingCalibrationStore 

class RollingCalibrationStore:
    def __init__(self):
        self.records: List[RollingRecord] = []

    def add(self, record: RollingRecord) -> None:
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.records])

    def filter_method(self, method_name: str) -> "RollingCalibrationStore":
        store = RollingCalibrationStore()
        store.records = [r for r in self.records if r.method_name == method_name]
        return store

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {"n_records": len(self.records)}
        for i, rec in enumerate(self.records):
            px = f"rec_{i}_"
            data[px + "method"] = rec.method_name
            data[px + "date"] = rec.date
            data[px + "T"] = rec.T
            data[px + "strikes"] = rec.strikes
            data[px + "market_prices"] = rec.market_prices
            data[px + "model_prices"] = rec.model_prices
            data[px + "params"] = rec.params
            data[px + "mae"] = rec.mae
            data[px + "rmse"] = rec.rmse
            data[px + "calibration_time"] = rec.calibration_time
            data[px + "success"] = rec.success
        np.savez(str(path), **data)
        print(f"Store saved → {path}  ({len(self.records)} records)")

    @classmethod
    def load(cls, path: str) -> "RollingCalibrationStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Store not found: {path}")
        data  = np.load(str(path), allow_pickle=True)
        n     = int(data["n_records"])
        store = cls()
        for i in range(n):
            px = f"rec_{i}_"
            store.add(RollingRecord(
                method_name = str(data[px + "method"]),
                date = str(data[px + "date"]),
                T = float(data[px + "T"]),
                strikes = data[px + "strikes"],
                market_prices = data[px + "market_prices"],
                model_prices = data[px + "model_prices"],
                params = data[px + "params"],
                mae = float(data[px + "mae"]),
                rmse = float(data[px + "rmse"]),
                calibration_time = float(data[px + "calibration_time"]),
                success = bool(data[px + "success"]),
            ))
        print(f"Store loaded ← {path}  ({n} records)")
        return store

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        return (
            df.groupby("method")
            .agg(
                n_days = ("date", "count"),
                mean_mae = ("mae", "mean"),
                median_mae = ("mae", "median"),
                mean_rmse = ("rmse", "mean"),
                pct_success = ("success", "mean"),
                mean_time_s = ("calibration_time", "mean"),
            )
            .round(6)
        )


# RollingCalibrator

class RollingCalibrator:

    def __init__(
        self,
        methods: List,
        npz_path: str,
        surface_model_path: str,
        target_maturity: float = 0.25,
        r: float = 0.0,
        n_paths: int = 20_000,
        exclude_periods: Optional[List[Tuple[str, str]]] = None,
        verbose: bool = True,
    ):
        self.methods = methods
        self.n_paths = n_paths
        self.verbose = verbose

        self._loader = SPXSurfaceLoader(
            npz_path=npz_path,
            surface_model_path=surface_model_path,
            target_maturity=target_maturity,
            r=r,
            exclude_periods=exclude_periods,
        )

    # Main loop 

    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        dates: Optional[List[str]] = None,
    ) -> RollingCalibrationStore:
    
        if dates is None:
            dates = self._filter_dates(start, end)

        if not dates:
            raise ValueError("No data available in the requested range.")

        store = RollingCalibrationStore()
        n = len(dates)

        print(f"\nRolling calibration: {n} datas × {len(self.methods)} métodos")
        print(f"Range: {dates[0]} → {dates[-1]}\n")

        for d_idx, date_str in enumerate(dates):
            if self.verbose:
                print(f"[{d_idx+1:>4}/{n}] {date_str}", end="  ")

            snapshot = self._loader.get(date_str)
            if snapshot is None:
                if self.verbose:
                    print("Skip (no snapshot)")
                continue

            strikes = snapshot["strikes"]
            market_prices = snapshot["market_prices"]
            T = snapshot["T"]

            for method in self.methods:
                t0 = time.perf_counter()
                try:
                    method.reset()
                    result = method.calibrate(strikes, market_prices, T)
                    model_prices = method.price_calls(strikes, T, n_paths=self.n_paths)
                    diff = model_prices - market_prices
                    mae = float(np.mean(np.abs(diff)))
                    rmse = float(np.sqrt(np.mean(diff ** 2)))
                    success = True
                except Exception as ex:
                    if self.verbose:
                        print(f"\n  {method.name} ERROR: {ex}")
                    model_prices = np.full_like(market_prices, np.nan)
                    mae = rmse = float("nan")
                    result = type("R", (), {"params": np.zeros(0)})()
                    success = False

                elapsed = time.perf_counter() - t0

                store.add(RollingRecord(
                    method_name = method.name,
                    date = date_str,
                    T = T,
                    strikes = strikes,
                    market_prices = market_prices,
                    model_prices = model_prices,
                    params = getattr(result, "params", np.zeros(0)),
                    mae = mae,
                    rmse = rmse,
                    calibration_time = elapsed,
                    success = success,
                ))

                if self.verbose:
                    status = "OK" if success else "FAIL"
                    print(f"{method.name}: MAE={mae:.4f} [{elapsed:.1f}s] {status}",
                          end="  ")

            if self.verbose:
                print()

        return store

    # Helpers

    def _filter_dates(
        self,
        start: Optional[str],
        end:   Optional[str],
    ) -> List[str]:
        all_dates = self._loader.dates()
        if start:
            all_dates = [d for d in all_dates if d >= start]
        if end:
            all_dates = [d for d in all_dates if d <= end]
        return all_dates


# Main 

if __name__ == "__main__":
    import tempfile

    EXCLUDE = COVID_PERIOD

    BASE = Path(r"C:\volatility-options")
    NPZ_PATH = str(BASE / "data" / "live_spx_data_extended.npz")
    SURFACE_MODEL_PATH = str(BASE / "data" / "surface_model.npz")
    RESULTS_DIR = BASE / "results" / "backtest"

    TARGET_MAT = 0.25
    R = 0.0

    print("rolling_calibration.py — smoke test (save/load)\n")

    # Testa RollingCalibrationStore save/load sem precisar de métodos reais
    store = RollingCalibrationStore()
    for i in range(3):
        store.add(RollingRecord(
            method_name = f"Method_{i % 2}",
            date = f"2023-01-0{i+1}",
            T = TARGET_MAT,
            strikes = np.array([90., 100., 110.]),
            market_prices = np.array([12., 5., 1.]),
            model_prices = np.array([12.1, 5.1, 1.1]),
            params = np.zeros(5),
            mae = 0.1,
            rmse = 0.1,
            calibration_time = 1.0,
            success = True,
        ))

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = f.name

    store.save(tmp)
    loaded = RollingCalibrationStore.load(tmp)
    assert len(loaded.records) == len(store.records), "Record count mismatch"
    print(loaded.summary().to_string())

    if Path(NPZ_PATH).exists() and Path(SURFACE_MODEL_PATH).exists():
        print("\n Testing SPXSurfaceLoader with real data")
        loader = SPXSurfaceLoader(
            npz_path=NPZ_PATH,
            surface_model_path=SURFACE_MODEL_PATH,
            target_maturity=TARGET_MAT,
            r=R,
            exclude_periods=EXCLUDE,
        )
        dates = loader.dates()
        print(f"Total dates available: {len(dates)}")
        if dates:
            snap = loader.get(dates[0])
            print(f" Snapshot [{dates[0]}]: "
                  f"strikes={snap['strikes'].shape}  "
                  f"S0={snap['S0']:.1f}  T={snap['T']:.2f}")
    else:
        print("\n [INFO] Dados reais não encontrados — pulando teste do loader.")

    print("\nSmoke test passed")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

COVID_PERIOD: List[Tuple[str, str]] = [("2020-02-20", "2020-06-30")]


def _apply_exclude_periods(
    index: pd.DatetimeIndex,
    exclude_periods: Optional[List[Tuple[str, str]]],
) -> pd.DatetimeIndex:
    """Remove datas que caem dentro de qualquer intervalo em exclude_periods."""
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


# Black-Scholes helpers

def bs_delta(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return float(np.exp(-q * T) * norm.cdf(d1))


def bs_price(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


def bs_vega(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


# Dataclasses 

@dataclass
class DailyPnL:
    date: str
    S: float
    K: float
    T_remaining: float
    sigma_model: float
    option_price: float
    delta: float
    hedge_pnl: float
    option_pnl: float
    financing: float
    daily_pnl: float
    cumulative_pnl: float


@dataclass
class SimulationResult:
    method_name: str
    K: float         
    T_entry: float
    r: float
    moneyness_k: float = 1.0          
    daily: List[DailyPnL] = field(default_factory=list)
    # rolling metadata: lista de (leg_idx, entry_date, K_leg)
    legs: List[Tuple[int, str, float]] = field(default_factory=list)

    @property
    def cumulative_pnl(self):
        return self.daily[-1].cumulative_pnl if self.daily else 0.0

    @property
    def sharpe(self):
        pnls = [d.daily_pnl for d in self.daily]
        if not pnls or np.std(pnls) < 1e-12:
            return 0.0
        return float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))

    @property
    def max_drawdown(self):
        cum = np.array([d.cumulative_pnl for d in self.daily])
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum
        return float(dd.max()) if len(dd) > 0 else 0.0

    def to_dataframe(self):
        return pd.DataFrame([d.__dict__ for d in self.daily])

    def summary(self):
        return {
            "method": self.method_name,
            "moneyness_k": self.moneyness_k,
            "T_entry": self.T_entry,
            "n_legs": len(self.legs),
            "cumulative_pnl": self.cumulative_pnl,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "n_days": len(self.daily),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n P&L Simulation: {self.method_name}")
        for k, v in s.items():
            print(f"  {k:<22}: {v}")
        if self.legs:
            print(f"  {'legs':<22}:", end="")
            for leg_idx, entry_date, K_leg in self.legs[:4]:
                print(f"  leg{leg_idx}@{entry_date}(K={K_leg:.1f})", end="")
            if len(self.legs) > 4:
                print(f"  ... (+{len(self.legs)-4} mais)", end="")
            print()
        print("─" * 58)

    def plot(self, save_path=None):
        df = self.to_dataframe()
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(df["date"], df["cumulative_pnl"], linewidth=2, color="steelblue")
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        for leg_idx, entry_date, K_leg in self.legs:
            axes[0].axvline(entry_date, color="orange", linestyle=":", linewidth=0.8, alpha=0.7)
        axes[0].set_ylabel("Cumulative P&L")
        axes[0].set_title(
            f"Rolling Delta-Hedged P&L — {self.method_name}  "
            f"(moneyness={self.moneyness_k:.2f}, T={self.T_entry:.2f}Y, {len(self.legs)} legs)",
            fontweight="bold",
        )
        axes[0].grid(True, alpha=0.3)
        axes[1].bar(df["date"], df["daily_pnl"], color="steelblue", alpha=0.6)
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        axes[1].set_ylabel("Daily P&L")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(df["date"], df["sigma_model"] * 100, linewidth=1.5, color="purple")
        axes[2].set_ylabel("Model IV (%)")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)
        for ax in axes:
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
        plt.tight_layout()
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved -> {save_path}")
        else:
            plt.show()


# Simulator 

class PnLSimulator:
    def __init__(
        self,
        store,
        price_series,
        r=0.0,
        q=0.0,
        dt=1 / 252,
        use_model_iv=True,
        exclude_periods: Optional[List[Tuple[str, str]]] = None,
    ):
        self.store = store
        self.r = r
        self.q = q
        self.dt = dt
        self.use_model_iv = use_model_iv
        self.exclude_periods = exclude_periods

        # Aplica exclusao na serie de precos
        raw = price_series.squeeze()
        raw.index = pd.to_datetime(raw.index)
        filtered_idx = _apply_exclude_periods(raw.index, exclude_periods)
        self.prices = raw.loc[filtered_idx]

    def run(
        self,
        K=None,
        T_entry=1.0,
        method_name=None,
        atm_vol_entry=0.20,
        moneyness_k=1.0,
    ):
        records_by_date = {}
        if self.store is not None:
            for rec in self.store.records:
                if method_name is None or rec.method_name == method_name:
                    if rec.date not in records_by_date:
                        records_by_date[rec.date] = rec

        dates_all      = sorted(self.prices.index)
        prices_str_map = {str(d.date()): d for d in dates_all}

        if records_by_date:
            sim_dates = [
                prices_str_map[d]
                for d in sorted(records_by_date)
                if d in prices_str_map
            ]
        else:
            sim_dates = dates_all

        if not sim_dates:
            raise ValueError(
                "No overlapping dates between price series and calibration store."
            )

        leg_steps = max(1, int(round(T_entry / self.dt)))  

        result = SimulationResult(
            method_name=method_name or "BS",
            K=float(self.prices.iloc[0]) * moneyness_k,   
            T_entry=T_entry,
            r=self.r,
            moneyness_k=moneyness_k,
        )
        cum_pnl = 0.0
        self._diag_count = 0 

        total_days = len(sim_dates)
        day_idx = 0  
        leg_idx = 0   
        while day_idx < total_days - 1:
            entry_date = sim_dates[day_idx]
            try:
                S_entry = float(self.prices.loc[entry_date])
            except KeyError:
                day_idx += 1
                continue

            K_leg = S_entry * moneyness_k
            entry_date_str = str(entry_date.date())
            result.legs.append((leg_idx, entry_date_str, K_leg))

            entry_pos = dates_all.index(entry_date) if entry_date in dates_all else -1
            past_window = dates_all[max(0, entry_pos - 30): entry_pos + 1]
            past_prices = [float(self.prices.loc[d]) for d in past_window if d in self.prices.index]
            if len(past_prices) > 1:
                leg_fallback = float(np.std(np.diff(np.log(past_prices))) * np.sqrt(252))
                leg_fallback = max(leg_fallback, 0.05)
            else:
                leg_fallback = atm_vol_entry

            leg_end = min(day_idx + leg_steps, total_days - 1)
            for step in range(day_idx, leg_end):
                date = sim_dates[step]
                next_date = sim_dates[step + 1]
                try:
                    S_t  = float(self.prices.loc[date])
                    S_t1 = float(self.prices.loc[next_date])
                except KeyError:
                    continue

                i_leg = step - day_idx         
                T_rem = max(T_entry - i_leg * self.dt, 1e-4)
                T_rem_next = max(T_entry - (i_leg + 1) * self.dt, 0.0)
                date_str = str(date.date())
                next_date_str = str(next_date.date())

                sigma_t = self._get_sigma(date_str, records_by_date, K_leg, S_t,  T_rem, leg_fallback)
                sigma_t1 = self._get_sigma(next_date_str, records_by_date, K_leg, S_t1, T_rem_next, sigma_t)
                C_t = bs_price(S_t,  K_leg, T_rem, sigma_t, self.r, self.q)
                C_t1 = bs_price(S_t1, K_leg, T_rem_next, sigma_t1, self.r, self.q)
                delta_t = bs_delta(S_t,  K_leg, T_rem, sigma_t,  self.r, self.q)

                dS = S_t1 - S_t
                dC = C_t1 - C_t
                hedge_pnl = delta_t * dS
                option_pnl = -dC
                financing = -self.r * (delta_t * S_t - C_t) * self.dt
                daily_pnl = hedge_pnl + option_pnl + financing
                cum_pnl  += daily_pnl

                result.daily.append(DailyPnL(
                    date=date_str, S=S_t, K=K_leg, T_remaining=T_rem,
                    sigma_model=sigma_t, option_price=C_t, delta=delta_t,
                    hedge_pnl=hedge_pnl, option_pnl=option_pnl,
                    financing=financing, daily_pnl=daily_pnl, cumulative_pnl=cum_pnl,
                ))

            # avança para o próximo leg (sem sobreposição)
            day_idx = leg_end
            leg_idx += 1

        print(f"  [run:{method_name}] rolling: {leg_idx} legs × ~{leg_steps} dias  "
              f"({sim_dates[0].date()} → {sim_dates[min(day_idx, total_days-1)].date()})  "
              f"total={len(result.daily)} dias")
        return result

    def compare_methods(self, T_entry=1.0, atm_vol_entry=0.20, moneyness_k=1.0):
        if self.store is None:
            return [self.run(T_entry=T_entry, atm_vol_entry=atm_vol_entry, moneyness_k=moneyness_k)]
        methods = list({rec.method_name for rec in self.store.records})
        return [
            self.run(T_entry=T_entry, method_name=m, atm_vol_entry=atm_vol_entry, moneyness_k=moneyness_k)
            for m in methods
        ]

    def _get_sigma(self, date, records, K, S, T, fallback):
        if not self.use_model_iv or date not in records:
            return fallback
        rec = records[date]
        strikes = rec.strikes
        model_prices = rec.model_prices
        if strikes is None or model_prices is None or len(strikes) == 0:
            return fallback
        idx = int(np.argmin(np.abs(strikes - K)))
        price = float(model_prices[idx])
        strike_hit = float(strikes[idx])
        sigma = float(np.clip(fallback, 0.01, 3.0))   # warm-start com sigma anterior
        for _ in range(50):
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            bs = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            vega_ = S * norm.pdf(d1) * np.sqrt(T)
            diff = bs - price
            if abs(diff) < 1e-6 or vega_ < 1e-12:
                break
            sigma -= diff / vega_
            sigma  = np.clip(sigma, 0.01, 3.0)

        method = getattr(rec, "method_name", "?")
        if not hasattr(self, "_diag_count"):
            self._diag_count = 0
        if self._diag_count < 5:
            print(f"  [diag:{method}] {date}  S={S:.2f}  K_req={K:.2f}  "
                  f"K_hit={strike_hit:.2f}  model_price={price:.4f}  IV={sigma*100:.2f}%")
        elif self._diag_count == 5:
            print(f"  [diag:{method}] ... (suprimindo demais linhas)")
        self._diag_count += 1

        return float(sigma)


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
    raw = np.load(npz_path, allow_pickle=True)
    seen = set()
    out = []
    for d in raw["fetch_date"]:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# Stores 

from dataclasses import dataclass as _dc


@_dc
class _SurfaceRecord:
    date: str
    method_name: str
    strikes: np.ndarray
    model_prices: np.ndarray
    maturities: np.ndarray


class SPXSurfaceStore:
    def __init__(
        self,
        npz_path,
        surface_model_path,
        method_name="surface_model",
        target_maturity=0.25,
        r=0.0,
        exclude_periods: Optional[List[Tuple[str, str]]] = None,
    ):
        self.method_name = method_name
        self.target_maturity = target_maturity
        self.r = r
        self.exclude_periods = exclude_periods
        self.records = []
        self._load(npz_path, surface_model_path)

    def _is_excluded(self, date_str: str) -> bool:
        if not self.exclude_periods:
            return False
        d = pd.Timestamp(date_str)
        return any(
            pd.Timestamp(s) <= d <= pd.Timestamp(e)
            for s, e in self.exclude_periods
        )

    def _load(self, npz_path, surface_model_path):
        import importlib.util
        from pathlib import Path

        _sm_dir = Path(surface_model_path).parent.parent / "models"
        _sm_file = _sm_dir / "surface_model.py"
        if not _sm_file.exists():
            _sm_file = Path("surface_model.py")
        if not _sm_file.exists():
            raise FileNotFoundError(f"surface_model.py nao encontrado em {_sm_dir}")

        spec = importlib.util.spec_from_file_location("surface_model", str(_sm_file))
        sm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sm_mod)
        sm = sm_mod.SurfaceModel()
        sm.load(surface_model_path)

        raw = np.load(npz_path, allow_pickle=True)
        moneyness_grid  = raw["strikes_norm"].astype(np.float64)
        maturity_grid = raw["maturities"].astype(np.float64)
        mat_idx = int(np.argmin(np.abs(maturity_grid - self.target_maturity)))
        prices_full = _fetch_spx_prices(str(raw["hist_start"]), float(raw["S0"]))
        scores_full = sm.pc_history.astype(np.float32)
        dates_unique = _unique_dates_from_npz(npz_path)
        n_align = min(len(dates_unique), len(scores_full))
        dates = dates_unique[-n_align:]
        scores = scores_full[-n_align:]

        prices_map = {}
        for d in dates:
            try:
                prices_map[d] = float(prices_full.loc[pd.Timestamp(d)])
            except KeyError:
                pass

        P, M = len(maturity_grid), len(moneyness_grid)
        skipped = 0

        for date, score in zip(dates, scores):
            if self._is_excluded(date):
                skipped += 1
                continue
            S_t = prices_map.get(date)
            if S_t is None:
                continue
            surf_flat = sm.pca.inverse_transform(score[np.newaxis])
            surf = surf_flat.reshape(P, M)
            strikes_abs = moneyness_grid * S_t
            iv_row = surf[mat_idx]
            T = float(maturity_grid[mat_idx])
            model_prices = np.array([
                bs_price(S_t, float(Ks), T, float(iv), self.r)
                for Ks, iv in zip(strikes_abs, iv_row)
            ], dtype=np.float64)
            self.records.append(_SurfaceRecord(
                date=date, method_name=self.method_name,
                strikes=strikes_abs, model_prices=model_prices,
                maturities=maturity_grid,
            ))

        if skipped:
            print(f" [exclude_periods] SPXSurfaceStore: {skipped} excluded periods")
        if not self.records:
            raise RuntimeError
        print(f"  SPXSurfaceStore: {len(self.records)} registros "
              f"({self.records[0].date} -> {self.records[-1].date})  "
              f"metodo={self.method_name}")


class SPXHestonStore(SPXSurfaceStore):
    def __init__(
        self,
        heston_pkl_path,
        surface_model_path,
        npz_path,
        method_name="heston",
        target_maturity=0.25,
        r=0.0,
        exclude_periods: Optional[List[Tuple[str, str]]] = None,
    ):
        self.method_name = method_name
        self.target_maturity = target_maturity
        self.r = r
        self.exclude_periods = exclude_periods
        self.records = []
        self._load_heston(heston_pkl_path, surface_model_path, npz_path)

    def _load_heston(self, heston_pkl_path, surface_model_path, npz_path):
        import pickle
        import importlib.util
        from pathlib import Path

        _sm_dir = Path(surface_model_path).parent.parent / "models"
        _sm_file = _sm_dir / "surface_model.py"
        if not _sm_file.exists():
            _sm_file = Path("surface_model.py")

        spec = importlib.util.spec_from_file_location("surface_model", str(_sm_file))
        sm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sm_mod)
        sm = sm_mod.SurfaceModel()
        sm.load(surface_model_path)

        with open(heston_pkl_path, "rb") as f:
            state = pickle.load(f)

        _h_file = _sm_dir / "heston.py"
        if not _h_file.exists():
            _h_file = Path("heston.py")
        spec2 = importlib.util.spec_from_file_location("heston_mod", str(_h_file))
        h_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(h_mod)

        cal = h_mod.HestonCalibrator.__new__(h_mod.HestonCalibrator)
        cal.surface_model = sm
        for k, v in state.items():
            setattr(cal, k, v)
        cal.rng = np.random.default_rng(42)
        cal._seed = 42
        cal._X_source = None
        cal._X_target = None
        if not hasattr(cal, "_regime_params"):
            cal._regime_params = None
        if not hasattr(cal, "_regime_edges"):
            cal._regime_edges  = None

        raw = np.load(npz_path, allow_pickle=True)
        moneyness_grid = raw["strikes_norm"].astype(np.float64)
        maturity_grid = raw["maturities"].astype(np.float64)
        mat_idx = int(np.argmin(np.abs(maturity_grid - self.target_maturity)))

        _hp = raw["heston_params"].astype(np.float64)
        _v0_fb, _kappa_fb, _theta_fb, _rho_fb = _hp
        _sigma_fb = getattr(cal, "sigma", 0.30)

        prices_full = _fetch_spx_prices(str(raw["hist_start"]), float(raw["S0"]))
        scores_full = sm.pc_history.astype(np.float32)
        dates_unique = _unique_dates_from_npz(npz_path)
        n_align = min(len(dates_unique), len(scores_full))
        dates = dates_unique[-n_align:]
        scores = scores_full[-n_align:]

        prices_map = {}
        for d in dates:
            try:
                prices_map[d] = float(prices_full.loc[pd.Timestamp(d)])
            except KeyError:
                pass

        pc_stds_k = getattr(cal, "pc_stds_k",
                            np.ones(getattr(cal, "K", scores.shape[1])))
        K_pca = getattr(cal, "K", scores.shape[1])
        P, M = len(maturity_grid), len(moneyness_grid)
        skipped = 0

        for date, score_full in zip(dates, scores):
            if self._is_excluded(date):
                skipped += 1
                continue
            S_t = prices_map.get(date)
            if S_t is None:
                continue

            score_k = score_full[:K_pca] / pc_stds_k
            strikes_abs = moneyness_grid * S_t
            T = float(maturity_grid[mat_idx])
            iv_row = None
            used_fallback = False

            try:
                params = cal._get_regime_params(score_k)
                kappa, theta, sigma, rho = params
                v0_est = max(float(cal._scores_to_surface(score_k)[0, M // 2] ** 2), 1e-4)
                iv_surf = h_mod.heston_iv_surface(
                    S_t, strikes_abs, maturity_grid, self.r,
                    v0_est, kappa, theta, sigma, rho, N_fft=64,
                )
                iv_row = iv_surf[mat_idx]
            except Exception:
                pass

            # NOTE: fallback para PCA (surface_model) foi removido intencionalmente.
            # Se o Heston falhar, só tentamos com os params fixos do NPZ.
            # Usar PCA aqui tornaria o Heston idêntico ao surface_model.
            if iv_row is None:
                try:
                    iv_surf = h_mod.heston_iv_surface(
                        S_t, strikes_abs, maturity_grid, self.r,
                        _v0_fb, _kappa_fb, _theta_fb, _sigma_fb, _rho_fb, N_fft=64,
                    )
                    iv_row = iv_surf[mat_idx]
                    # marca que veio do fallback fixo para diagnóstico
                    iv_row = np.where(np.isfinite(iv_row) & (iv_row > 0), iv_row, np.nan)
                    if not np.any(np.isfinite(iv_row)):
                        iv_row = None
                    else:
                        used_fallback = True
                except Exception:
                    pass

            if iv_row is None:
                continue

            model_prices = np.array([
                bs_price(S_t, float(Ks), T, float(iv), self.r)
                for Ks, iv in zip(strikes_abs, iv_row)
            ], dtype=np.float64)
            rec = _SurfaceRecord(
                date=date, method_name=self.method_name,
                strikes=strikes_abs, model_prices=model_prices,
                maturities=maturity_grid,
            )
            rec._fallback = used_fallback
            self.records.append(rec)

        if skipped:
            print(f"  [exclude_periods] SPXHestonStore: {skipped} registros excluidos")
        n_fallback = sum(1 for r in self.records if getattr(r, "_fallback", False))
        n_heston = len(self.records) - n_fallback
        if not self.records:
            raise RuntimeError("SPXHestonStore: nenhum registro gerado.")
        print(f"  SPXHestonStore: {len(self.records)} registros "
              f"({self.records[0].date} -> {self.records[-1].date})  "
              f"metodo={self.method_name}  "
              f"[heston_real={n_heston}  fallback_fixo={n_fallback}]")


# Main 

if __name__ == "__main__":
    from pathlib import Path
    EXCLUDE = COVID_PERIOD
   
    excl_tag = (
        "no_covid" if EXCLUDE == COVID_PERIOD
        else ("full"   if not EXCLUDE
              else "custom")
    )

    print("pnl_simulator.py — SPX real data\n")

    BASE = Path(r"C:\volatility-options")
    NPZ_PATH = str(BASE / "data" / "live_spx_data_extended.npz")
    SM_PATH = str(BASE / "data" / "surface_model.npz")
    HESTON_PATH = str(BASE / "data" / "heston_surface.pkl")
    RESULTS_DIR = str(BASE / "results" / "backtest")

    TARGET_MAT  = 0.25
    R           = 0.0
    MONEYNESS_K = 1.0

    stores = []

    try:
        sm_store = SPXSurfaceStore(
            npz_path=NPZ_PATH, surface_model_path=SM_PATH,
            method_name="surface_model", target_maturity=TARGET_MAT, r=R,
            exclude_periods=EXCLUDE,
        )
        stores.append(sm_store)
    except Exception as e:
        print(f"  [WARN] SPXSurfaceStore falhou: {e}")

    try:
        h_store = SPXHestonStore(
            heston_pkl_path=HESTON_PATH, surface_model_path=SM_PATH,
            npz_path=NPZ_PATH,
            method_name="heston", target_maturity=TARGET_MAT, r=R,
            exclude_periods=EXCLUDE,
        )
        stores.append(h_store)
    except Exception as e:
        print(f"  [WARN] SPXHestonStore falhou: {e}")

    if not stores:
        raise RuntimeError

    raw = np.load(NPZ_PATH, allow_pickle=True)
    price_series = _fetch_spx_prices(str(raw["hist_start"]), float(raw["S0"]))

    pv = np.array(price_series, dtype=float).flatten()
    print(f"\n  Price series (pre-filtro): {len(pv)} days"
          f"S_min={pv.min():.1f}  S_max={pv.max():.1f}")

    log_rets = np.diff(np.log(pv))
    atm_vol_entry = float(np.std(log_rets[-30:]) * np.sqrt(252))
    print(f"  ATM vol entry (30d realizada): {atm_vol_entry*100:.1f}%")
    print(f"  MONEYNESS_K = {MONEYNESS_K} × {MONEYNESS_K}")

    results = []

    for store in stores:
        sim = PnLSimulator(
            store=store,
            price_series=price_series,
            r=R, use_model_iv=True,
            exclude_periods=EXCLUDE,
        )
        try:
            res = sim.run(
                T_entry=TARGET_MAT,
                method_name=store.method_name,
                atm_vol_entry=atm_vol_entry,
                moneyness_k=MONEYNESS_K,
            )
            res.print_summary()
            results.append(res)
            fname = f"pnl_{store.method_name}_{excl_tag}.png"
            res.plot(save_path=str(Path(RESULTS_DIR) / fname))
        except Exception as e:
            print(f"Simulation {store.method_name} failed: {e}")

    if len(results) > 1:
        print("\n Method Comparison")
        df_cmp = pd.DataFrame([r.summary() for r in results]).set_index("method")
        print(df_cmp.to_string())
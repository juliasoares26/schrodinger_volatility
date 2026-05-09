import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "models"))

# Shared utils 
try:
    from utils import (
        COVID_PERIOD,
        is_excluded as _is_excluded_fn,
        fetch_spx_prices as _fetch_spx_prices,
        unique_dates_from_npz as _unique_dates_from_npz,
        load_surface_model as _load_surface_model,
    )
except ImportError:
    COVID_PERIOD = [("2020-02-20", "2020-06-30")]
    def _is_excluded_fn(date_str, exclude_periods):
        if not exclude_periods: return False
        d = pd.Timestamp(date_str)
        return any(pd.Timestamp(s) <= d <= pd.Timestamp(e) for s, e in exclude_periods)

# Alpha Engineering
def build_regime_features(price_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": price_series})
    returns = df["price"].pct_change()
    df["vol_20"] = returns.rolling(20).std()
    df["mom_20"] = df["price"] / df["price"].shift(20) - 1
    df["regime"] = (df["vol_20"] > df["vol_20"].median()).astype(int)
    return df.fillna(0.0)


@dataclass
class EvaluationReport:
    method_tag: str
    n_test: int
    horizon: int
    sharpe_strategy: float
    sharpe_buy_hold: float
    total_trades: int
    win_rate: float
    avg_size: float = 0.0

    def print(self) -> None:
        print(f"\n EvaluationReport: {self.method_tag}")
        print(f"  Test Samples   : {self.n_test}  Horizonte: {self.horizon}d")
        print(f"  Strategy Sharpe (Ann) : {self.sharpe_strategy:.4f}")
        print(f"  Buy & Hold Sharpe (Ann): {self.sharpe_buy_hold:.4f}")
        print(f"  Total Trades   : {self.total_trades}  ({self.total_trades/self.n_test*100:.1f}%)")
        print(f"  Win Rate       : {self.win_rate*100:.1f}%")
        print(f"  Avg Position   : {self.avg_size:.3f}")


# SPXSnapBuilder

class SPXSnapBuilder:
    def __init__(self, npz_path, surface_model_path, horizon=10, exclude_periods=None):
        self.npz_path = npz_path
        self.surface_model_path = surface_model_path
        self.horizon = horizon
        self.exclude_periods = exclude_periods

    def build(self):
        sm = _load_surface_model(self.surface_model_path)
        scores_full = sm.pc_history.astype(np.float64)
        raw = np.load(self.npz_path, allow_pickle=True)
        prices_full = _fetch_spx_prices(str(raw["hist_start"]), float(raw["S0"]))
        dates_unique = _unique_dates_from_npz(self.npz_path)

        reg_feats = build_regime_features(prices_full)
        n_align = min(len(dates_unique), len(scores_full))
        dates, scores = dates_unique[-n_align:], scores_full[-n_align:]
        prices_map = {
            d: float(prices_full.loc[pd.Timestamp(d)])
            for d in dates if pd.Timestamp(d) in prices_full.index
        }

        X1_rows, X2_rows = [], []
        for i, date in enumerate(dates):
            if i + self.horizon >= len(dates):
                continue
            if _is_excluded_fn(date, self.exclude_periods):
                continue
            S_t = prices_map.get(date)
            S_f = prices_map.get(dates[i + self.horizon])
            if S_t is None or S_f is None:
                continue
            ret_h = np.log(S_f / S_t)
            try:
                extra = reg_feats.loc[pd.Timestamp(date), ["vol_20", "mom_20", "regime"]].values
                X1_rows.append(np.concatenate([scores[i], extra]))
                X2_rows.append([ret_h])
            except KeyError:
                continue

        X1 = np.array(X1_rows)
        X2 = np.array(X2_rows)
        print(f"  SPXSnapBuilder: {len(X1)} samples "
              f"d1={X1.shape[1]}  d2=1  horizon={self.horizon}d")
        return X1, X2


# PredictionEvaluator 

class PredictionEvaluator:

    def __init__(self, X1, X2, horizon):
        from brenier import make_brenier_estimator
        self._make_est = make_brenier_estimator
        self.X1, self.X2 = X1, X2
        self.horizon = horizon

    # helpers 
    def _fit_model(self, X1_tr, X2_tr, X1_cal, X2_cal, label="model"):
        """Treina + calibra conformal um estimador Brenier d2=1."""
        n  = len(X1_tr)
        d1 = X1_tr.shape[1]
        model = self._make_est(n_samples=n, d1=d1, d2=1, method="adaptive")
        print(f"\n  [{label}] fit  n={n}  d1={d1}  d2=1")
        model.fit(X1_tr, X2_tr)
        model.calibrate_conformal(X1_cal, X2_cal, coverage=0.90)
        return model

    def _get_signal(self, model, x1_row, n_samples=300):
        """
        Extrai (mu, sigma, skew, q_low, q_high) de um único ponto.
        Retorna tudo no espaço do retorno (escalar d2=1).
        """
        _, samples = model.predict(x1_row, return_distribution=True, n_samples=n_samples)
        samples = samples.flatten()          # (n_samples,) — retorno univariado
        mu    = float(np.mean(samples))
        sigma = float(np.std(samples)) + 1e-8
        skew  = float(np.mean(((samples - mu) / sigma) ** 3))

        _, low, high = model.predict_interval(x1_row)
        q_low  = float(np.atleast_1d(low.flatten())[0])
        q_high = float(np.atleast_1d(high.flatten())[0])

        return mu, sigma, skew, q_low, q_high

    # main evaluation 

    def evaluate_trading(self, train_frac=0.7, n_samples=300,
                         score_thresh=0.3, conf_thresh=0.005):
        n_train = int(len(self.X1) * train_frac)
        X1_tr_full = self.X1[:n_train]
        X2_tr_full = self.X2[:n_train]
        X1_te = self.X1[n_train:]
        X2_te = self.X2[n_train:]

        # 20% do treino reservado para calibração conformal — fora do fit
        n_cal  = int(len(X1_tr_full) * 0.2)
        X1_tr  = X1_tr_full[:-n_cal]
        X2_tr  = X2_tr_full[:-n_cal]
        X1_cal = X1_tr_full[-n_cal:]
        X2_cal = X2_tr_full[-n_cal:]

        regime_tr = X1_tr[:, -1]

        model_all = self._fit_model(X1_tr, X2_tr, X1_cal, X2_cal,
                                    label="model_all")

        mask_lv_tr  = regime_tr == 0
        mask_lv_cal = X1_cal[:, -1] == 0
        if mask_lv_tr.sum() >= 30 and mask_lv_cal.sum() >= 10:
            model_lv = self._fit_model(
                X1_tr[mask_lv_tr],  X2_tr[mask_lv_tr],
                X1_cal[mask_lv_cal], X2_cal[mask_lv_cal],
                label="model_lowvol",
            )
            use_specialist = True
        else:
            print("  [model_lowvol] Insufficient samples — using model_all as fallback")
            model_lv = model_all
            use_specialist = False

        # ── Loop de avaliação ─────────────────────────────────────────────────
        strategy_returns = []
        sizes_used = []
        trades = 0
        wins   = 0

        print(f"\n  Evaluating {len(X1_te)} test samples...")

        for i in range(len(X1_te)):
            x1     = X1_te[i:i+1]
            regime = int(X1_te[i, -1])

            mu_all, sigma_all, skew_all, q_low_all, q_high_all = \
                self._get_signal(model_all, x1, n_samples)
            score_all = mu_all / sigma_all

            if use_specialist and regime == 0:
                mu_lv, sigma_lv, skew_lv, q_low_lv, q_high_lv = \
                    self._get_signal(model_lv, x1, n_samples)
                score_lv = mu_lv / sigma_lv
                score = 0.6 * score_all + 0.4 * score_lv
                q_low = 0.6 * q_low_all  + 0.4 * q_low_lv
                q_high = 0.6 * q_high_all + 0.4 * q_high_lv
                skew = 0.6 * skew_all   + 0.4 * skew_lv
            else:
                score = score_all
                q_low = q_low_all
                q_high = q_high_all
                skew = skew_all

            signal = 0
            if   score >  score_thresh and q_low  >  conf_thresh:
                signal =  1
            elif score < -score_thresh and q_high < -conf_thresh:
                signal = -1

            if signal == 0:
                strategy_returns.append(0.0)
                continue

            # Position sizing
            size = float(np.clip(abs(score), 0.0, 1.0))

            vol_scale = 0.4 if regime == 1 else 1.0
            size *= vol_scale

            size *= 1.2 if (signal * skew > 0) else 0.8

            size = float(np.clip(size, 0.0, 1.0))

            pnl = size * signal * float(X2_te[i, 0])
            strategy_returns.append(pnl)
            sizes_used.append(size)
            trades += 1
            if pnl > 0:
                wins += 1

        # Metrics 
        strat_arr = np.array(strategy_returns)
        bh_arr = X2_te[:, 0]

        def ann_sharpe(r):
            return float(np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252 / self.horizon))

        avg_size = float(np.mean(sizes_used)) if sizes_used else 0.0

        return EvaluationReport(
            method_tag = "Brenier-d2=1-DualModel",
            n_test = len(X1_te),
            horizon = self.horizon,
            sharpe_strategy = ann_sharpe(strat_arr),
            sharpe_buy_hold = ann_sharpe(bh_arr),
            total_trades = trades,
            win_rate = wins / (trades + 1e-8),
            avg_size = avg_size,
        )


# Entry point

if __name__ == "__main__":
    BASE = Path(r"C:\volatility-options")
    builder = SPXSnapBuilder(
        str(BASE / "data" / "live_spx_data_extended.npz"),
        str(BASE / "data" / "surface_model.npz"),
        horizon=10,
        exclude_periods=COVID_PERIOD,
    )
    X1, X2 = builder.build()
    report = PredictionEvaluator(X1, X2, horizon=10).evaluate_trading()
    report.print()
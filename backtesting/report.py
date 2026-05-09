import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")

_COLORS = [
    "steelblue", "darkorange", "seagreen", "crimson",
    "mediumpurple", "saddlebrown", "deeppink", "teal",
]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


# BacktestReport

class BacktestReport:

    def __init__(
        self,
        results: List,
        output_dir: str = "results/backtest",
        title: str = "Schrödinger Volatility — Backtest Report",
    ):
        if not results:
            raise ValueError("Forneça ao menos um SimulationResult em `results`.")
        self.results = results
        self.output_dir = Path(output_dir)
        self.title  = title
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Entry point

    def generate(self) -> None:
        """Gera todos os artefatos do relatório."""
        print(f"\n Generating Backtest Report")
        print(f"Output dir: {self.output_dir}")
        print(f"Methods: {[r.method_name for r in self.results]}")

        self._print_summary_table()
        self._plot_cumulative_pnl()
        self._plot_daily_pnl()
        self._plot_sigma_ts()
        self._plot_drawdown()
        self._plot_full_report()

        print("Report complete \n")

    # Console summary 

    def _print_summary_table(self) -> None:
        rows = [r.summary() for r in self.results]
        df   = pd.DataFrame(rows).set_index("method")
        print("\n P&L Summary")
        print(df.to_string(float_format=lambda x: f"{x:+.4f}"))
        path = self.output_dir / "pnl_summary.csv"
        df.to_csv(path)
        print(f"  Saved: {path.name}")

    # Individual plots 
    def _plot_cumulative_pnl(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 5))
        for i, sim in enumerate(self.results):
            df = sim.to_dataframe()
            ax.plot(
                df["date"], df["cumulative_pnl"],
                label=f"{sim.method_name}  (Sharpe={sim.sharpe:.2f})",
                color=_color(i), linewidth=2,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
        ax.set_ylabel("Cumulative P&L")
        ax.set_title("Delta-Hedged Cumulative P&L", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _rotate_ticks(ax)
        plt.tight_layout()
        self._save(fig, "cumulative_pnl.png")

    def _plot_daily_pnl(self) -> None:
        n   = len(self.results)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, sim in enumerate(self.results):
            df = sim.to_dataframe()
            axes[i].bar(df["date"], df["daily_pnl"], color=_color(i), alpha=0.6)
            axes[i].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
            axes[i].set_ylabel("Daily P&L")
            axes[i].set_title(f"Daily P&L — {sim.method_name}", fontweight="bold")
            axes[i].grid(True, alpha=0.3)
            _rotate_ticks(axes[i])
        plt.tight_layout()
        self._save(fig, "daily_pnl.png")

    def _plot_sigma_ts(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 4))
        for i, sim in enumerate(self.results):
            df = sim.to_dataframe()
            ax.plot(
                df["date"], df["sigma_model"] * 100,
                label=sim.method_name, color=_color(i), linewidth=1.5, alpha=0.85,
            )
        ax.set_ylabel("Model IV (%)")
        ax.set_title("Model Implied Volatility — Time Series", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _rotate_ticks(ax)
        plt.tight_layout()
        self._save(fig, "sigma_ts.png")

    def _plot_drawdown(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 4))
        for i, sim in enumerate(self.results):
            df  = sim.to_dataframe()
            cum = df["cumulative_pnl"].values
            run = np.maximum.accumulate(cum)
            dd  = run - cum
            ax.fill_between(df["date"], -dd, 0, alpha=0.35, color=_color(i),
                            label=f"{sim.method_name}  (MaxDD={sim.max_drawdown:.4f})")
        ax.set_ylabel("Drawdown")
        ax.set_title("Drawdown from Peak", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _rotate_ticks(ax)
        plt.tight_layout()
        self._save(fig, "drawdown.png")

    # Full combined report 

    def _plot_full_report(self) -> None:
        fig = plt.figure(figsize=(18, 22))
        gs  = gridspec.GridSpec(4, 1, hspace=0.45, height_ratios=[2, 2, 2, 1.5])

        ax0 = fig.add_subplot(gs[0])
        for i, sim in enumerate(self.results):
            df = sim.to_dataframe()
            ax0.plot(
                df["date"], df["cumulative_pnl"],
                label=f"{sim.method_name}  (Sharpe={sim.sharpe:.2f})",
                color=_color(i), linewidth=2,
            )
        ax0.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
        ax0.set_ylabel("Cumulative P&L")
        ax0.set_title("Delta-Hedged Cumulative P&L", fontweight="bold")
        ax0.legend(fontsize=9)
        ax0.grid(True, alpha=0.3)
        _rotate_ticks(ax0)

        # daily P&L sobrepostos
        ax1 = fig.add_subplot(gs[1])
        for i, sim in enumerate(self.results):
            df = sim.to_dataframe()
            ax1.bar(df["date"], df["daily_pnl"],
                    color=_color(i), alpha=0.5, label=sim.method_name)
        ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
        ax1.set_ylabel("Daily P&L")
        ax1.set_title("Daily P&L", fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        _rotate_ticks(ax1)

        # model IV
        ax2 = fig.add_subplot(gs[2])
        for i, sim in enumerate(self.results):
            df = sim.to_dataframe()
            ax2.plot(
                df["date"], df["sigma_model"] * 100,
                label=sim.method_name, color=_color(i), linewidth=1.5, alpha=0.85,
            )
        ax2.set_ylabel("Model IV (%)")
        ax2.set_title("Model Implied Volatility", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        _rotate_ticks(ax2)

        ax3 = fig.add_subplot(gs[3])
        ax3.axis("off")
        lines = [self.title, "=" * 72, ""]
        lines.append(f"  {'Method':<22} {'CumPnL':>10} {'Sharpe':>8} {'MaxDD':>10} {'N_days':>8}")
        lines.append("  " + "-" * 62)
        for sim in self.results:
            s = sim.summary()
            lines.append(
                f"  {s['method']:<22} {s['cumulative_pnl']:>+10.4f}"
                f" {s['sharpe']:>8.3f} {s['max_drawdown']:>10.4f} {s['n_days']:>8}"
            )
        lines.append("")
        lines.append(f"  K={self.results[0].K:.2f}   T_entry={self.results[0].T_entry:.2f}Y   r={self.results[0].r:.4f}")

        ax3.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax3.transAxes,
            fontsize=9, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )

        plt.suptitle(self.title, fontsize=14, fontweight="bold", y=1.01)
        self._save(fig, "full_report.png")

    # Helper

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.output_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")


def _rotate_ticks(ax, rotation: int = 25) -> None:
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)


# Smoke test 

if __name__ == "__main__":
    import tempfile
    print("report.py — smoke test\n")

    try:
        from pnl_simulator import SimulationResult, DailyPnL
    except ImportError:
        raise ImportError("pnl_simulator.py not found")

    rng = np.random.default_rng(0)
    n = 120

    def _fake_result(name: str, drift: float) -> SimulationResult:
        res     = SimulationResult(method_name=name, K=4500.0, T_entry=0.25, r=0.0)
        cum_pnl = 0.0
        for i in range(n):
            daily = float(rng.normal(drift, 0.002))
            cum_pnl += daily
            res.daily.append(DailyPnL(
                date=str(pd.Timestamp("2023-01-02") + pd.Timedelta(days=i)),
                S=4500.0 + i, K=4500.0, T_remaining=max(0.25 - i / 252, 1e-4),
                sigma_model=0.18 + 0.01 * rng.standard_normal(),
                option_price=50.0, delta=0.5,
                hedge_pnl=daily * 0.6, option_delta=daily * 0.4,
                financing=0.0, daily_pnl=daily, cumulative_pnl=cum_pnl,
            ))
        return res

    results = [
        _fake_result("bridge", drift=+0.0005),
        _fake_result("heston", drift=-0.0002),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        report = BacktestReport(results=results, output_dir=tmp, title="Smoke Test")
        report.generate()
        files = sorted(Path(tmp).iterdir())
        print(f"  Arquivos gerados: {[f.name for f in files]}")

    print("\nSmoke test passed")
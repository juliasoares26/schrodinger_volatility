import numpy as np
import pandas as pd
from datetime import date, datetime
import warnings

warnings.filterwarnings("ignore")

from yfinance_loader import (
    get_risk_free_rate,
    get_dividend_yield,
    implied_vol,
    compute_forward,
)

_IB_AVAILABLE = False
try:
    import ib_insync  # noqa: F401
    _IB_AVAILABLE = True
except ImportError:
    pass


class IBKRLoader:

    def __init__(
        self,
        host:      str = "127.0.0.1",
        port:      int = 7497,
        client_id: int = 1,
        timeout:   int = 15,
    ):
        if not _IB_AVAILABLE:
            raise ImportError(
                "ib_insync is not installed. Run: pip install ib_insync\n"
                "Also ensure TWS or IB Gateway is running"
            )
        from ib_insync import IB
        self._ib      = IB()
        self._host    = host
        self._port    = port
        self._client  = client_id
        self._timeout = timeout
        self._connected = False

    # Connection management

    def connect(self) -> None:
        if not self._connected:
            self._ib.connect(self._host, self._port, clientId=self._client,
                             timeout=self._timeout)
            self._connected = True
            print(f"IBKR connected: {self._host}:{self._port} (clientId={self._client})")

    def disconnect(self) -> None:
        if self._connected:
            self._ib.disconnect()
            self._connected = False

    # SPX option chain

    def load_spx_options(
        self,
        min_dte: int   = 7,
        max_dte: int   = 400,
        min_open_interest: int   = 100,
        r: float = None,
        q: float = None,
    ) -> tuple:
        from ib_insync import Stock, Index, Option

        self.connect()
        ib = self._ib

        if r is None:
            r = get_risk_free_rate()
        if q is None:
            q = get_dividend_yield()

        fetch_date = date.today()

        # SPX spot via index contract
        spx_contract = Index("SPX", "CBOE", "USD")
        ib.qualifyContracts(spx_contract)
        [ticker] = ib.reqTickers(spx_contract)
        S0 = float(ticker.marketPrice() or ticker.close or 0)
        if S0 <= 0:
            raise RuntimeError("Could not get SPX spot price from IB")

        print(f"  IBKR SPX spot: {S0:.2f}  r={r*100:.2f}%  q={q*100:.2f}%")

        # Option chain parameters
        chains = ib.reqSecDefOptParams("SPX", "", "IND", ib.qualifyContracts(spx_contract)[0].conId)
        if not chains:
            raise RuntimeError("No option chain parameters returned by IB")

        chain   = next((c for c in chains if c.exchange == "CBOE"), chains[0])
        expiries = sorted(chain.expirations)

        records = []
        for exp_str in expiries:
            exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
            dte      = (exp_date - fetch_date).days
            if not (min_dte <= dte <= max_dte):
                continue
            T = dte / 365.0

            # Request market data for each strike
            strikes_filtered = [
                s for s in sorted(chain.strikes)
                if S0 * 0.75 <= s <= S0 * 1.30
            ]

            for opt_type in ("C", "P"):
                opt_label = "call" if opt_type == "C" else "put"
                contracts = [
                    Option("SPX", exp_str, K, opt_type, "CBOE")
                    for K in strikes_filtered
                ]
                ib.qualifyContracts(*contracts)
                tickers = ib.reqTickers(*contracts)

                for ticker_obj, K in zip(tickers, strikes_filtered):
                    bid = float(ticker_obj.bid or 0)
                    ask = float(ticker_obj.ask or 0)
                    oi  = float(getattr(ticker_obj, "openInterest", 0) or 0)
                    if bid <= 0 or ask <= bid:
                        continue
                    if oi < min_open_interest:
                        continue

                    mid = (bid + ask) / 2.0
                    iv  = implied_vol(S0, K, T, mid, r, q, opt_label)
                    if iv is None or np.isnan(iv):
                        continue

                    records.append({
                        "fetch_date": fetch_date,
                        "expiration": exp_date,
                        "dte": dte,
                        "T": T,
                        "strike": K,
                        "moneyness": K / S0,
                        "log_moneyness": np.log(K / S0),
                        "option_type": opt_label,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "spread": ask - bid,
                        "volume": 0.0,
                        "open_interest": oi,
                        "iv": iv,
                        "S0": S0,
                        "r": r,
                        "q": q,
                        "forward": compute_forward(S0, r, q, T),
                    })

            n_exp = sum(1 for rec in records if rec["expiration"] == exp_date)
            print(f"{exp_str} (DTE={dte}): {n_exp} options")

        df_out = pd.DataFrame(records)
        print(f"\n  IBKR total raw records: {len(df_out)}")
        return df_out, S0, r, q, fetch_date

    # SPX historical bar data

    def load_spx_history(
        self,
        duration_str: str = "5 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
    ) -> pd.Series:
        from ib_insync import Index

        self.connect()
        spx_contract = Index("SPX", "CBOE", "USD")
        self._ib.qualifyContracts(spx_contract)

        bars = self._ib.reqHistoricalData(
            spx_contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
        )
        if not bars:
            raise RuntimeError("No historical data returned by IB")

        df = pd.DataFrame([{"date": b.date, "close": b.close} for b in bars])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        prices = df.set_index("date")["close"].sort_index()
        print(f" IBKR loaded {len(prices)} trading days")
        return prices
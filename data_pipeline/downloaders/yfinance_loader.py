import numpy as np
import pandas as pd
from datetime import datetime, date
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# Risk-free rate / dividend yield helper

def get_risk_free_rate() -> float:
    """3-month Treasury yield as risk-free proxy. Falls back to ~5.3%"""
    try:
        import yfinance as yf
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.053


def get_dividend_yield(ticker: str = "SPY") -> float:
    """
    Continuous dividend yield for SPX (uses SPY as proxy). Falls back to 1.5%

    yfinance occasionally returns dividendYield already as a decimal (e.g. 0.014) but some versions / tickers return it as a percentage-like float (e.g. 1.14)
    We guard against the latter by capping at a plausible upper bound of 10%
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info

        # Prefer trailingAnnualDividendYield — more reliable for ETFs
        for field in ("trailingAnnualDividendYield", "dividendYield"):
            val = info.get(field)
            if val is not None:
                val = float(val)
                # If yfinance returned it as a whole-number percentage (e.g. 1.4
                # meaning 1.4%), convert to decimal.  Threshold: anything > 0.15
                # (15%) is almost certainly mis-scaled for the SPX.
                if val > 0.15:
                    val /= 100.0
                # Final sanity clamp: SPX yield is realistically 0.5%–5%
                val = float(np.clip(val, 0.005, 0.05))
                return val
    except Exception:
        pass
    return 0.015

# Black-Scholes helper

def compute_forward(S0: float, r: float, q: float, T: float) -> float:
    """F = S0 * exp((r - q) * T)"""
    return S0 * np.exp((r - q) * T)


def black_scholes_price(
    S: float, K: float, T: float, sigma: float,
    r: float = 0.0, q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Black-Scholes call or put price"""
    if T <= 0 or sigma <= 1e-6:
        if option_type == "call":
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    F  = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def implied_vol(
    S: float, K: float, T: float, price: float,
    r: float = 0.0, q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson implied volatility. Returns NaN if not solvable"""
    F    = S * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(disc * (F - K), 0.0)
    else:
        intrinsic = max(disc * (K - F), 0.0)

    if price <= intrinsic + 1e-8:
        return np.nan

    sigma = 0.25
    for _ in range(max_iter):
        bs = black_scholes_price(S, K, T, sigma, r, q, option_type)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        diff = bs - price
        if abs(diff) < tol:
            break
        if vega < 1e-12:
            return np.nan
        sigma -= diff / vega
        sigma  = np.clip(sigma, 1e-4, 5.0)

    return float(sigma) if 1e-4 < sigma < 4.9 else np.nan

# SPX option chain loade

def load_spx_yfinance(
    min_dte: int = 7,
    max_dte: int = 400,
    min_volume: int = 10,
    min_open_interest: int = 100,
) -> tuple:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    print("\n Loading SPX options via yfinance...")
    spx  = yf.Ticker("^SPX")
    hist = spx.history(period="2d")
    if hist.empty:
        raise ValueError("Could not fetch SPX spot from yfinance")

    S0 = float(hist["Close"].iloc[-1])
    fetch_date = hist.index[-1].date()
    r = get_risk_free_rate()
    q = get_dividend_yield()
    print(f"SPX spot: {S0:.2f}  |  date: {fetch_date}")
    print(f"r={r*100:.2f}%  q={q*100:.2f}%")

    records = []
    for exp_str in spx.options:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - fetch_date).days
        if not (min_dte <= dte <= max_dte):
            continue
        T = dte / 365.0

        try:
            chain = spx.option_chain(exp_str)
        except Exception as e:
            print(f"  Skipping {exp_str}: {e}")
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            if df.empty:
                continue
            for _, row in df.iterrows():
                K = float(row.get("strike", np.nan))
                bid = float(row.get("bid",    np.nan))
                ask = float(row.get("ask",    np.nan))
                vol = float(row.get("volume", 0) or 0)
                oi  = float(row.get("openInterest", 0) or 0)
                if np.isnan(K) or np.isnan(bid) or np.isnan(ask):
                    continue
                if bid <= 0 or ask <= bid:
                    continue
                if vol < min_volume and oi < min_open_interest:
                    continue

                mid = (bid + ask) / 2.0
                iv  = implied_vol(S0, K, T, mid, r, q, opt_type)
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
                    "option_type": opt_type,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread": ask - bid,
                    "volume": vol,
                    "open_interest": oi,
                    "iv": iv,
                    "S0": S0,
                    "r": r,
                    "q": q,
                    "forward": compute_forward(S0, r, q, T),
                })

        n_exp = sum(1 for rec in records if rec["expiration"] == exp_date)
        print(f"  {exp_str} (DTE={dte}): {n_exp} options")

    df_out = pd.DataFrame(records)
    print(f"\n  Total raw records: {len(df_out)}")
    return df_out, S0, r, q, fetch_date

# SPX historical price loade

def load_spx_history(
    start: str = "2020-01-01",
    end: str = None,
    freq: str = "1d",
) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    print(f"\n Loading SPX history {start} → {end}")
    hist = yf.Ticker("^SPX").history(start=start, end=end, interval=freq)
    if hist.empty:
        raise ValueError("No SPX history returned by yfinance")

    prices= hist["Close"].copy()
    prices.index = pd.to_datetime(prices.index).date
    prices = prices.sort_index()
    print(f"  Loaded {len(prices)} trading days  [{prices.min():.2f}, {prices.max():.2f}]")
    return prices

# Smoke tes

if __name__ == "__main__":
    r = get_risk_free_rate()
    q = get_dividend_yield()
    print(f"Risk-free rate: {r*100:.3f}%")
    print(f"Dividend yield: {q*100:.3f}%")

    # IV round-trip
    S0, K, T, sigma = 5000.0, 5000.0, 0.25, 0.20
    price = black_scholes_price(S0, K, T, sigma, r, q)
    iv_rt = implied_vol(S0, K, T, price, r, q)
    assert abs(iv_rt - sigma) < 1e-5, "IV round-trip failed"
    print(f"IV round-trip OK: {sigma:.4f} → {iv_rt:.4f}")

    try:
        prices = load_spx_history(start="2023-01-01")
        print(f"History OK: {len(prices)} days")
    except Exception as e:
        print(f"History skipped (no network): {e}")
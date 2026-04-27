"""
Data fetcher module — downloads and caches market data from yfinance.

Handles:
  - OHLCV daily price data for any ticker
  - VIX (market fear gauge)
  - 10-Year Treasury yield (risk-free rate proxy)
  - Current options chains (calls/puts with implied volatility)
  - Local CSV caching to avoid redundant API calls
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_START_DATE, DATA_END_DATE, CACHE_DIR, VIX_TICKER, TREASURY_TICKER


def _ensure_cache_dir():
    """Create the cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str, suffix: str = "") -> str:
    """Generate a cache file path for a ticker."""
    _ensure_cache_dir()
    clean = ticker.replace("^", "").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{clean}{suffix}.csv")


def _is_cache_fresh(path: str, max_age_hours: int = 12) -> bool:
    """Check if a cached file exists and is fresh enough."""
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime) < timedelta(hours=max_age_hours)


def fetch_price_data(
    ticker: str,
    start: str = DATA_START_DATE,
    end: str = DATA_END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a ticker.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Adj Close, Volume
    indexed by Date.
    """
    cache = _cache_path(ticker, "_prices")
    if use_cache and _is_cache_fresh(cache):
        print(f"  [cache] Loading {ticker} price data from {cache}")
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df

    print(f"  [fetch] Downloading {ticker} price data ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if present (yfinance sometimes returns multi-level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache)
    print(f"  [cache] Saved {len(df)} rows to {cache}")
    return df


def fetch_vix(
    start: str = DATA_START_DATE,
    end: str = DATA_END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download VIX (CBOE Volatility Index) daily data.

    Returns DataFrame with 'Close' column as VIX level.
    """
    return fetch_price_data(VIX_TICKER, start, end, use_cache)


def fetch_treasury_yield(
    start: str = DATA_START_DATE,
    end: str = DATA_END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download 10-Year Treasury Yield daily data.

    Returns DataFrame with 'Close' column as yield.
    """
    return fetch_price_data(TREASURY_TICKER, start, end, use_cache)


def fetch_options_chain(ticker: str) -> dict:
    """
    Fetch the current options chain for a ticker.

    Returns a dict with:
      - 'expirations': list of expiration date strings
      - 'calls': DataFrame of call options for the nearest expiration
      - 'puts': DataFrame of put options for the nearest expiration
      - 'all_chains': dict mapping expiration → (calls_df, puts_df)
    """
    print(f"  [fetch] Downloading options chain for {ticker}...")
    tk = yf.Ticker(ticker)

    expirations = tk.options
    if not expirations:
        print(f"  [warn] No options data available for {ticker}")
        return {
            "expirations": [],
            "calls": pd.DataFrame(),
            "puts": pd.DataFrame(),
            "all_chains": {},
        }

    # Get chains for up to 4 nearest expirations
    all_chains = {}
    for exp in expirations[:4]:
        try:
            chain = tk.option_chain(exp)
            all_chains[exp] = {
                "calls": chain.calls,
                "puts": chain.puts,
            }
        except Exception as e:
            print(f"  [warn] Failed to fetch chain for {exp}: {e}")

    # Nearest expiration as default
    nearest = expirations[0]
    nearest_chain = all_chains.get(nearest, {"calls": pd.DataFrame(), "puts": pd.DataFrame()})

    return {
        "expirations": list(expirations),
        "calls": nearest_chain["calls"],
        "puts": nearest_chain["puts"],
        "all_chains": all_chains,
    }


def fetch_all_data(ticker: str, use_cache: bool = True) -> dict:
    """
    Fetch all data needed for the pipeline for a single ticker.

    Returns a dict with:
      - 'prices': OHLCV DataFrame
      - 'vix': VIX DataFrame
      - 'treasury': Treasury yield DataFrame
      - 'options': Options chain dict
    """
    print(f"\n{'='*60}")
    print(f"Fetching data for {ticker}")
    print(f"{'='*60}")

    prices = fetch_price_data(ticker, use_cache=use_cache)
    vix = fetch_vix(use_cache=use_cache)
    treasury = fetch_treasury_yield(use_cache=use_cache)

    # Options chain is always live (not cached — snapshot data)
    try:
        options = fetch_options_chain(ticker)
    except Exception as e:
        print(f"  [warn] Options chain fetch failed: {e}")
        options = {
            "expirations": [],
            "calls": pd.DataFrame(),
            "puts": pd.DataFrame(),
            "all_chains": {},
        }

    print(f"\nData summary for {ticker}:")
    print(f"  Price data:  {len(prices)} trading days ({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"  VIX data:    {len(vix)} trading days")
    print(f"  Treasury:    {len(treasury)} trading days")
    print(f"  Options:     {len(options['expirations'])} expirations available")

    return {
        "prices": prices,
        "vix": vix,
        "treasury": treasury,
        "options": options,
    }


if __name__ == "__main__":
    # Quick test
    data = fetch_all_data("SPY")
    print("\nPrice data tail:")
    print(data["prices"].tail())

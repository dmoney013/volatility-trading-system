"""
Futures Data Utilities — fetch fresh futures and ETF data.

Uses yfinance to download:
  - Futures price data (ES=F, NQ=F, etc.) for price quotes and GARCH analysis
  - ETF price data (SPY, QQQ, etc.) for LSTM feature computation
  - VIX data for market context features

Always fetches the most up-to-date data available (no caching).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict

from futures.config import ALL_FUTURES, FUTURES_UNIVERSE


def fetch_futures_prices(etf_symbol: str,
                         years: int = 4,
                         use_micro: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data for a futures contract.

    Args:
        etf_symbol: The ETF key (e.g., 'SPY') — maps to futures symbol internally
        years: Years of history to fetch
        use_micro: If True, use micro futures symbol (MES=F instead of ES=F)

    Returns:
        DataFrame with OHLCV columns, or None on failure
    """
    if etf_symbol not in ALL_FUTURES:
        print(f"  ❌ {etf_symbol}: Not in futures universe")
        return None

    spec = ALL_FUTURES[etf_symbol]
    symbol = spec['micro'] if use_micro else spec['future']

    try:
        start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        data = yf.download(symbol, start=start, progress=False)

        if data is None or len(data) == 0:
            print(f"  ❌ {symbol}: No data returned")
            return None

        # Flatten multi-level columns from newer yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Drop rows with NaN close
        data = data.dropna(subset=['Close'])

        print(f"  📈 {symbol} ({spec['name']}): {len(data)} rows "
              f"({data.index[0].strftime('%Y-%m-%d')} → "
              f"{data.index[-1].strftime('%Y-%m-%d')})")
        return data

    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return None


def fetch_etf_prices(etf_symbol: str, years: int = 4) -> Optional[pd.DataFrame]:
    """
    Fetch fresh ETF price data for LSTM feature computation.
    Always downloads fresh — no cache — to ensure most up-to-date data.

    Args:
        etf_symbol: ETF ticker (SPY, QQQ, IWM, DIA, ARKK, BITO)
        years: Years of history

    Returns:
        DataFrame with OHLCV columns
    """
    try:
        start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        data = yf.download(etf_symbol, start=start, progress=False)

        if data is None or len(data) == 0:
            print(f"  ❌ {etf_symbol} ETF: No data returned")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data = data.dropna(subset=['Close'])

        print(f"  📊 {etf_symbol} ETF: {len(data)} rows "
              f"({data.index[0].strftime('%Y-%m-%d')} → "
              f"{data.index[-1].strftime('%Y-%m-%d')})")
        return data

    except Exception as e:
        print(f"  ❌ {etf_symbol} ETF: {e}")
        return None


def fetch_vix(years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch VIX data for LSTM market context features."""
    try:
        start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        vix = yf.download("^VIX", start=start, progress=False)
        if vix is not None and isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
        return vix
    except Exception:
        return None


def compute_basis(etf_prices: pd.DataFrame,
                  futures_prices: pd.DataFrame) -> pd.Series:
    """
    Compute the futures basis (futures - spot) aligned by date.

    Positive basis = contango (normal), negative = backwardation.
    """
    # Align by date
    aligned = pd.DataFrame({
        'spot': etf_prices['Close'],
        'futures': futures_prices['Close']
    }).dropna()

    basis = aligned['futures'] - aligned['spot']
    basis_pct = basis / aligned['spot'] * 100  # As percentage of spot

    return basis_pct


def get_latest_quotes(etf_symbol: str) -> Dict:
    """
    Get the most recent price quotes for both the ETF and futures contract.

    Returns dict with spot, futures, basis, and contract specs.
    """
    spec = ALL_FUTURES[etf_symbol]

    # Fetch latest 5 days to get most recent close
    etf = yf.download(etf_symbol, period='5d', progress=False)
    fut = yf.download(spec['future'], period='5d', progress=False)
    micro = yf.download(spec['micro'], period='5d', progress=False)

    if isinstance(etf.columns, pd.MultiIndex):
        etf.columns = etf.columns.droplevel(1)
    if isinstance(fut.columns, pd.MultiIndex):
        fut.columns = fut.columns.droplevel(1)
    if isinstance(micro.columns, pd.MultiIndex):
        micro.columns = micro.columns.droplevel(1)

    spot = float(etf['Close'].iloc[-1]) if etf is not None and len(etf) > 0 else None
    fut_price = float(fut['Close'].iloc[-1]) if fut is not None and len(fut) > 0 else None
    micro_price = float(micro['Close'].iloc[-1]) if micro is not None and len(micro) > 0 else None

    basis = None
    if spot and fut_price:
        basis = round((fut_price - spot) / spot * 100, 3)

    return {
        'etf_symbol': etf_symbol,
        'future_symbol': spec['future'],
        'micro_symbol': spec['micro'],
        'name': spec['name'],
        'spot_price': spot,
        'futures_price': fut_price,
        'micro_price': micro_price,
        'basis_pct': basis,
        'point_value': spec['point_value'],
        'micro_point_value': spec['micro_pv'],
        'margin': spec['margin'],
        'micro_margin': spec['micro_margin'],
    }

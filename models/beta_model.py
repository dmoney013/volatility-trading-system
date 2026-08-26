"""
Beta Model — Rolling beta, R², and best-fit index/ETF selection.

Computes the sensitivity of each stock to market indices and sector ETFs
to determine which benchmark best explains the stock's movement.

Used by the Hybrid Scanner to translate index-level directional predictions
into stock-level directional predictions via: expected_move = β × index_move

Key metrics:
    β (beta):  Sensitivity — how much the stock moves per 1% index move
    R²:        Fit quality — what % of stock variance is explained by the index
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════
#  Benchmark Definitions
# ═══════════════════════════════════════════════════════════════════

# Major indices — we train LSTMs on these
INDEX_BENCHMARKS = {
    'SPY':  'S&P 500',
    'QQQ':  'Nasdaq 100',
    'IWM':  'Russell 2000',
    'DIA':  'Dow Jones',
    'ARKK': 'Innovation/Growth',
    'BITO': 'Bitcoin/Crypto',
}

# Sector ETFs — used for beta/R² computation only
SECTOR_BENCHMARKS = {
    'XLK':  'Technology',
    'XLF':  'Financials',
    'XLE':  'Energy',
    'XLV':  'Healthcare',
    'XLY':  'Consumer Discretionary',
    'XLI':  'Industrials',
    'XLC':  'Communication Services',
    'XLB':  'Materials',
    'XLRE': 'Real Estate',
    'XLU':  'Utilities',
    'XLP':  'Consumer Staples',
}

# All benchmarks combined
ALL_BENCHMARKS = {**INDEX_BENCHMARKS, **SECTOR_BENCHMARKS}

# Trainable indices (we build LSTM models for these)
TRAINABLE_INDICES = list(INDEX_BENCHMARKS.keys())


# ═══════════════════════════════════════════════════════════════════
#  Beta & R² Computation
# ═══════════════════════════════════════════════════════════════════

def compute_rolling_beta(stock_returns: pd.Series,
                         index_returns: pd.Series,
                         window: int = 60) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling beta and R² between a stock and an index.

    β = Cov(stock, index) / Var(index)
    R² = Corr(stock, index)²

    Args:
        stock_returns: Daily log returns of the stock
        index_returns: Daily log returns of the index
        window: Rolling window in trading days (default 60 = ~3 months)

    Returns:
        (beta_series, r_squared_series)
    """
    # Align the series
    common = stock_returns.index.intersection(index_returns.index)
    stock_ret = stock_returns.loc[common]
    index_ret = index_returns.loc[common]

    # Rolling covariance and variance
    cov = stock_ret.rolling(window).cov(index_ret)
    var = index_ret.rolling(window).var()

    beta = cov / var.replace(0, np.nan)

    # Rolling R²
    corr = stock_ret.rolling(window).corr(index_ret)
    r_squared = corr ** 2

    return beta.dropna(), r_squared.dropna()


def compute_current_beta(stock_prices: pd.DataFrame,
                         index_prices: pd.DataFrame,
                         window: int = 60) -> Dict:
    """
    Compute current beta and R² (most recent values).

    Returns:
        Dict with beta, r_squared, and recent values
    """
    stock_ret = np.log(stock_prices['Close'] / stock_prices['Close'].shift(1)).dropna()
    index_ret = np.log(index_prices['Close'] / index_prices['Close'].shift(1)).dropna()

    beta_series, r2_series = compute_rolling_beta(stock_ret, index_ret, window)

    if len(beta_series) == 0 or len(r2_series) == 0:
        return {'beta': 0.0, 'r_squared': 0.0, 'valid': False}

    return {
        'beta': round(beta_series.iloc[-1], 4),
        'r_squared': round(r2_series.iloc[-1], 4),
        'beta_mean_60d': round(beta_series.iloc[-60:].mean(), 4) if len(beta_series) >= 60 else round(beta_series.mean(), 4),
        'r2_mean_60d': round(r2_series.iloc[-60:].mean(), 4) if len(r2_series) >= 60 else round(r2_series.mean(), 4),
        'valid': True,
    }


# ═══════════════════════════════════════════════════════════════════
#  Best-Fit Index Selection
# ═══════════════════════════════════════════════════════════════════

_benchmark_cache = {}

def _fetch_benchmark(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch and cache benchmark price data."""
    if symbol in _benchmark_cache:
        return _benchmark_cache[symbol]

    try:
        data = yf.download(symbol, period="4y", progress=False)
        if data is not None and len(data) > 200:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            _benchmark_cache[symbol] = data
            return data
    except Exception:
        pass
    return None


def find_best_fit_index(stock_prices: pd.DataFrame,
                        benchmarks: Dict[str, str] = None,
                        min_r2: float = 0.15,
                        window: int = 60) -> Dict:
    """
    Find the benchmark (index or sector ETF) that best explains
    the stock's movement (highest R²).

    Args:
        stock_prices: DataFrame with stock OHLCV
        benchmarks: Dict of {symbol: name} to test against
        min_r2: Minimum R² to consider a valid fit
        window: Rolling window for beta computation

    Returns:
        Dict with best_benchmark, beta, r_squared, and all results
    """
    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    stock_ret = np.log(
        stock_prices['Close'] / stock_prices['Close'].shift(1)
    ).dropna()

    results = {}
    best_r2 = -1
    best_symbol = None

    for symbol, name in benchmarks.items():
        bench_data = _fetch_benchmark(symbol)
        if bench_data is None:
            continue

        bench_ret = np.log(
            bench_data['Close'] / bench_data['Close'].shift(1)
        ).dropna()

        beta_series, r2_series = compute_rolling_beta(
            stock_ret, bench_ret, window)

        if len(beta_series) == 0:
            continue

        current_beta = float(beta_series.iloc[-1])
        current_r2 = float(r2_series.iloc[-1])

        results[symbol] = {
            'name': name,
            'beta': round(current_beta, 4),
            'r_squared': round(current_r2, 4),
            'is_trainable': symbol in TRAINABLE_INDICES,
        }

        if current_r2 > best_r2:
            best_r2 = current_r2
            best_symbol = symbol

    # If the best fit is a sector ETF (not trainable), find the best
    # trainable index as fallback
    best_trainable = None
    best_trainable_r2 = -1
    for sym in TRAINABLE_INDICES:
        if sym in results and float(results[sym]['r_squared']) > best_trainable_r2:
            best_trainable_r2 = float(results[sym]['r_squared'])
            best_trainable = sym

    return {
        'best_benchmark': best_symbol,
        'best_benchmark_name': results[best_symbol]['name'] if best_symbol else 'N/A',
        'best_beta': results[best_symbol]['beta'] if best_symbol else 0.0,
        'best_r2': round(best_r2, 4),
        'best_trainable_index': best_trainable,
        'best_trainable_r2': round(best_trainable_r2, 4) if best_trainable else 0.0,
        'trainable_beta': results[best_trainable]['beta'] if best_trainable else 0.0,
        'meets_min_r2': best_r2 >= min_r2,
        'all_results': results,
    }


# ═══════════════════════════════════════════════════════════════════
#  Batch Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_universe(tickers: list, window: int = 60) -> pd.DataFrame:
    """
    Compute best-fit index for every ticker in the universe.
    Returns a summary DataFrame.
    """
    from data.fetcher import fetch_price_data

    rows = []
    for ticker in tickers:
        print(f"  📊 {ticker}...", end=" ", flush=True)

        try:
            prices = fetch_price_data(ticker)
            if prices is None or len(prices) < 200:
                print("insufficient data")
                continue

            fit = find_best_fit_index(prices, window=window)
            rows.append({
                'ticker': ticker,
                'best_benchmark': fit['best_benchmark'],
                'best_name': fit['best_benchmark_name'],
                'best_beta': fit['best_beta'],
                'best_r2': fit['best_r2'],
                'best_trainable': fit['best_trainable_index'],
                'trainable_r2': fit['best_trainable_r2'],
                'trainable_beta': fit['trainable_beta'],
                'valid': fit['meets_min_r2'],
            })
            print(f"→ {fit['best_benchmark']} (β={fit['best_beta']:.2f}, "
                  f"R²={fit['best_r2']:.2f}) | "
                  f"Best index: {fit['best_trainable_index']} "
                  f"(R²={fit['best_trainable_r2']:.2f})")
        except Exception as e:
            print(f"error: {e}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from signals.scanner import SCAN_UNIVERSE

    print(f"\n🔬 Beta Analysis — {len(SCAN_UNIVERSE)} tickers × "
          f"{len(ALL_BENCHMARKS)} benchmarks\n")

    df = analyze_universe(SCAN_UNIVERSE)

    print(f"\n{'='*80}")
    print(f"  BEST-FIT BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Ticker':<8} {'Best Benchmark':<20} {'β':<8} {'R²':<8} "
          f"{'Best Index':<10} {'Idx R²':<8} {'Valid':<6}")
    print(f"  {'─'*75}")

    for _, r in df.iterrows():
        print(f"  {r['ticker']:<8} {r['best_benchmark']:<6} "
              f"({r['best_name']:<12}) {r['best_beta']:<8.2f} "
              f"{r['best_r2']:<8.3f} {r['best_trainable']:<10} "
              f"{r['trainable_r2']:<8.3f} {'✅' if r['valid'] else '❌'}")

    # Summary stats
    print(f"\n  Benchmarks used:")
    for bench in df['best_benchmark'].value_counts().items():
        print(f"    {bench[0]}: {bench[1]} tickers")

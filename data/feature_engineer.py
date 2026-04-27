"""
Feature engineering module — constructs ~20 features for the Transformer.

Features are organized into categories:
  - Price / Returns
  - Realized Volatility (rolling windows)
  - Volume dynamics
  - Technical indicators (RSI, MACD, Bollinger, ATR)
  - Market regime (VIX)
  - Options-derived (put/call ratio, implied vol)
  - Macro (Treasury yield)
  - Momentum

All features are z-score normalized using rolling windows to avoid lookahead bias.
"""
import numpy as np
import pandas as pd
import ta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_DAYS, FEATURE_NAMES


def compute_log_returns(prices: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from Close prices."""
    return np.log(prices["Close"] / prices["Close"].shift(1))


def compute_realized_vol(log_returns: pd.Series, window: int) -> pd.Series:
    """Compute annualized rolling realized volatility."""
    return log_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)


def build_features(
    prices: pd.DataFrame,
    vix: pd.DataFrame,
    treasury: pd.DataFrame,
    options: dict,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw data.

    Args:
        prices: OHLCV DataFrame for the target ticker
        vix: VIX DataFrame (Close column)
        treasury: Treasury yield DataFrame (Close column)
        options: Options chain dict from fetcher

    Returns:
        DataFrame with columns matching FEATURE_NAMES, indexed by date.
        NaN rows at the start (from rolling calculations) are dropped.
    """
    df = pd.DataFrame(index=prices.index)

    # ─── Price / Returns ────────────────────────────────────────────
    log_ret = compute_log_returns(prices)
    df["Log Return"] = log_ret
    df["Abs Return"] = log_ret.abs()
    df["Return Sign"] = np.sign(log_ret)

    # ─── Realized Volatility ────────────────────────────────────────
    df["RV 5-Day"] = compute_realized_vol(log_ret, 5)
    df["RV 10-Day"] = compute_realized_vol(log_ret, 10)
    df["RV 21-Day"] = compute_realized_vol(log_ret, 21)

    # ─── Volume ─────────────────────────────────────────────────────
    vol_ma20 = prices["Volume"].rolling(20).mean()
    df["Volume Ratio"] = prices["Volume"] / vol_ma20
    df["Log Volume"] = np.log1p(prices["Volume"])

    # ─── Technical Indicators ───────────────────────────────────────
    df["RSI(14)"] = ta.momentum.RSIIndicator(
        close=prices["Close"], window=14
    ).rsi()

    macd = ta.trend.MACD(close=prices["Close"])
    df["MACD Signal"] = macd.macd_signal()

    boll = ta.volatility.BollingerBands(close=prices["Close"], window=20)
    boll_upper = boll.bollinger_hband()
    boll_lower = boll.bollinger_lband()
    boll_mid = boll.bollinger_mavg()
    # Bollinger width = (upper - lower) / middle
    df["Bollinger Width"] = (boll_upper - boll_lower) / boll_mid

    df["ATR(14)"] = ta.volatility.AverageTrueRange(
        high=prices["High"], low=prices["Low"], close=prices["Close"], window=14
    ).average_true_range()

    # ─── Market Regime (VIX) ────────────────────────────────────────
    # Align VIX to prices index
    vix_aligned = vix["Close"].reindex(prices.index, method="ffill")
    df["VIX Level"] = vix_aligned
    df["VIX Change %"] = vix_aligned.pct_change()

    # ─── Options-Derived ────────────────────────────────────────────
    # These are snapshot features — we use the latest options data
    # For historical training, we'll forward-fill placeholder values
    _add_options_features(df, options, prices)

    # ─── Macro ──────────────────────────────────────────────────────
    treasury_aligned = treasury["Close"].reindex(prices.index, method="ffill")
    df["10Y Yield"] = treasury_aligned
    df["Yield Change"] = treasury_aligned.diff()

    # ─── Momentum ───────────────────────────────────────────────────
    df["Momentum 5D"] = prices["Close"].pct_change(5)
    df["Momentum 21D"] = prices["Close"].pct_change(21)

    # ─── Drop NaN rows from rolling calculations ────────────────────
    df = df.dropna()

    # Ensure column order matches FEATURE_NAMES
    df = df[FEATURE_NAMES]

    return df


def _add_options_features(
    df: pd.DataFrame,
    options: dict,
    prices: pd.DataFrame,
):
    """
    Add options-derived features.

    For historical periods we use proxy features since we don't have
    historical options data. For recent data, real options metrics
    are used where available.
    """
    calls = options.get("calls", pd.DataFrame())
    puts = options.get("puts", pd.DataFrame())

    if not calls.empty and not puts.empty:
        # Current snapshot values
        put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
        call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
        pc_ratio = put_vol / max(call_vol, 1)

        oi_calls = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
        oi_puts = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
        total_oi = oi_calls + oi_puts

        avg_iv = 0.0
        if "impliedVolatility" in calls.columns:
            # Weighted average IV by volume for ATM-ish strikes
            spot = prices["Close"].iloc[-1]
            near_calls = calls[
                (calls["strike"] >= spot * 0.95) & (calls["strike"] <= spot * 1.05)
            ]
            if not near_calls.empty and near_calls["impliedVolatility"].notna().any():
                avg_iv = near_calls["impliedVolatility"].mean()
            else:
                avg_iv = calls["impliedVolatility"].mean()

        # Fill the entire series with historical proxies, then override last row
        # Proxy: use VIX-scaled estimate for put/call ratio and IV
        df["Put/Call Ratio"] = 0.7  # market average
        df["Open Interest Change"] = 0.0
        df["Avg Implied Vol"] = df.get("VIX Level", pd.Series(0.2, index=df.index)) / 100.0

        # Override the last available row with real data
        if len(df) > 0:
            df.loc[df.index[-1], "Put/Call Ratio"] = pc_ratio
            df.loc[df.index[-1], "Open Interest Change"] = total_oi
            df.loc[df.index[-1], "Avg Implied Vol"] = avg_iv
    else:
        # No options data — use proxies throughout
        df["Put/Call Ratio"] = 0.7
        df["Open Interest Change"] = 0.0
        if "VIX Level" in df.columns:
            df["Avg Implied Vol"] = df["VIX Level"] / 100.0
        else:
            df["Avg Implied Vol"] = 0.2


def normalize_features(
    df: pd.DataFrame,
    rolling_window: int = 63,  # ~3 months for z-score
) -> pd.DataFrame:
    """
    Z-score normalize features using a rolling window.

    This avoids lookahead bias — each data point is normalized
    only using past data.
    """
    rolling_mean = df.rolling(window=rolling_window, min_periods=21).mean()
    rolling_std = df.rolling(window=rolling_window, min_periods=21).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, 1e-8)

    normalized = (df - rolling_mean) / rolling_std
    normalized = normalized.dropna()

    # Clip extreme values to [-5, 5] for training stability
    normalized = normalized.clip(-5, 5)

    return normalized


def build_target(
    prices: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """
    Build the prediction target: next-day realized volatility.

    Computed as absolute log return (single-day proxy for realized vol).
    """
    log_ret = compute_log_returns(prices)
    # Forward-shifted so we predict tomorrow's vol from today's features
    target = log_ret.abs().shift(-horizon)
    target.name = "target_vol"
    return target


if __name__ == "__main__":
    from fetcher import fetch_all_data

    data = fetch_all_data("SPY")
    features = build_features(
        data["prices"], data["vix"], data["treasury"], data["options"]
    )
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
    print(f"\nFeature statistics:")
    print(features.describe().round(4))

    normed = normalize_features(features)
    print(f"\nNormalized shape: {normed.shape}")
    print(normed.tail())

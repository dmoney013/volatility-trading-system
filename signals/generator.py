"""
Signal Generator — combines GARCH volatility forecasts with Transformer
insights to produce actionable options trading signals.

Signal Logic:
  - Compare market Implied Volatility (IV) against GARCH-forecasted
    Realized Volatility (RV)
  - If IV >> RV: options overpriced → SELL VOL (sell straddles/strangles)
  - If IV << RV: options underpriced → BUY VOL (buy straddles/strangles)
  - Annotate with top Transformer-identified volatility drivers
"""
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VOL_SPREAD_THRESHOLD, SIGNAL_STRENGTH_MAX


class SignalDirection:
    BUY_VOL = "BUY VOL 📈"
    SELL_VOL = "SELL VOL 📉"
    NEUTRAL = "NEUTRAL ⚖️"


def compute_iv_from_options(options: dict) -> float:
    """
    Compute aggregate implied volatility from options chain.

    Uses volume-weighted average of ATM-ish options.
    """
    calls = options.get("calls", pd.DataFrame())
    puts = options.get("puts", pd.DataFrame())

    if calls.empty and puts.empty:
        return None

    ivs = []
    weights = []

    for chain in [calls, puts]:
        if not chain.empty and "impliedVolatility" in chain.columns:
            valid = chain[chain["impliedVolatility"].notna() & (chain["impliedVolatility"] > 0)]
            if not valid.empty:
                vol = valid.get("volume", pd.Series(1, index=valid.index)).fillna(1)
                ivs.extend(valid["impliedVolatility"].values)
                weights.extend(vol.values)

    if not ivs:
        return None

    ivs = np.array(ivs)
    weights = np.array(weights, dtype=float)
    weights = np.maximum(weights, 1)  # Ensure positive weights

    return float(np.average(ivs, weights=weights))


def generate_signals(
    ticker: str,
    garch_forecast_vol: float,
    garch_current_vol: float,
    options: dict,
    feature_importance: pd.DataFrame = None,
    prices: pd.DataFrame = None,
) -> dict:
    """
    Generate trading signals for a ticker.

    Args:
        ticker: Stock symbol
        garch_forecast_vol: GARCH 1-day-ahead annualized vol forecast (decimal)
        garch_current_vol: GARCH current annualized conditional vol (decimal)
        options: Options chain dict from fetcher
        feature_importance: Transformer feature importance DataFrame
        prices: Price DataFrame for context

    Returns:
        Signal dictionary with direction, strength, spread, and context
    """
    # Get market implied volatility
    market_iv = compute_iv_from_options(options)

    if market_iv is None:
        # Fall back to using VIX-based estimate if no options data
        return {
            "ticker": ticker,
            "signal": SignalDirection.NEUTRAL,
            "strength": 0,
            "iv": None,
            "rv_forecast": garch_forecast_vol,
            "rv_current": garch_current_vol,
            "spread": None,
            "spread_pct": None,
            "top_drivers": _get_top_drivers(feature_importance),
            "strategy": "Insufficient options data for signal generation",
            "rationale": "No implied volatility data available from options chain.",
        }

    # ─── Core Signal: IV vs RV Spread ───────────────────────────
    spread = market_iv - garch_forecast_vol
    spread_pct = spread / max(garch_forecast_vol, 0.001) * 100  # % mispricing

    # Signal direction
    if spread > VOL_SPREAD_THRESHOLD:
        direction = SignalDirection.SELL_VOL
        strategy = _sell_vol_strategy(spread_pct, prices)
        rationale = (
            f"Market IV ({market_iv*100:.1f}%) exceeds GARCH forecast RV "
            f"({garch_forecast_vol*100:.1f}%) by {spread*100:.1f}pp. "
            f"Options appear overpriced — volatility premium is elevated."
        )
    elif spread < -VOL_SPREAD_THRESHOLD:
        direction = SignalDirection.BUY_VOL
        strategy = _buy_vol_strategy(spread_pct, prices)
        rationale = (
            f"GARCH forecast RV ({garch_forecast_vol*100:.1f}%) exceeds "
            f"market IV ({market_iv*100:.1f}%) by {abs(spread)*100:.1f}pp. "
            f"Options appear underpriced — consider buying volatility."
        )
    else:
        direction = SignalDirection.NEUTRAL
        strategy = "No trade — IV and RV are in equilibrium."
        rationale = (
            f"IV ({market_iv*100:.1f}%) ≈ RV ({garch_forecast_vol*100:.1f}%). "
            f"Spread of {spread*100:.1f}pp is within threshold (±{VOL_SPREAD_THRESHOLD*100:.0f}pp)."
        )

    # Signal strength (0-100)
    strength = min(
        int(abs(spread_pct) / 2),  # Scale: 50% mispricing → strength 25
        SIGNAL_STRENGTH_MAX,
    )

    # ─── Volatility regime context ──────────────────────────────
    vol_regime = _classify_vol_regime(garch_current_vol)

    return {
        "ticker": ticker,
        "signal": direction,
        "strength": strength,
        "iv": market_iv,
        "rv_forecast": garch_forecast_vol,
        "rv_current": garch_current_vol,
        "spread": spread,
        "spread_pct": spread_pct,
        "vol_regime": vol_regime,
        "top_drivers": _get_top_drivers(feature_importance),
        "strategy": strategy,
        "rationale": rationale,
    }


def _get_top_drivers(feature_importance: pd.DataFrame, n: int = 3) -> list:
    """Get top N volatility drivers from Transformer importance."""
    if feature_importance is None or feature_importance.empty:
        return ["(Transformer not trained)"]

    top = feature_importance.head(n)
    return [
        f"{row['Feature']} ({row['Importance']*100:.1f}%)"
        for _, row in top.iterrows()
    ]


def _classify_vol_regime(current_vol: float) -> str:
    """Classify current volatility regime."""
    if current_vol > 0.35:
        return "🔴 EXTREME — Vol > 35%"
    elif current_vol > 0.25:
        return "🟠 HIGH — Vol 25-35%"
    elif current_vol > 0.15:
        return "🟡 MODERATE — Vol 15-25%"
    else:
        return "🟢 LOW — Vol < 15%"


def _sell_vol_strategy(spread_pct: float, prices: pd.DataFrame) -> str:
    """Suggest a sell-vol strategy based on spread magnitude."""
    if abs(spread_pct) > 30:
        return (
            "Iron Condor or Short Strangle — High conviction sell-vol. "
            "Consider wide strikes (1-2 std dev) to manage tail risk."
        )
    elif abs(spread_pct) > 15:
        return (
            "Short Straddle or Credit Spread — Moderate sell-vol opportunity. "
            "ATM or slightly OTM strikes recommended."
        )
    else:
        return (
            "Covered Call or Cash-Secured Put — Conservative sell-vol approach. "
            "Mild IV premium suggests cautious positioning."
        )


def _buy_vol_strategy(spread_pct: float, prices: pd.DataFrame) -> str:
    """Suggest a buy-vol strategy based on spread magnitude."""
    if abs(spread_pct) > 30:
        return (
            "Long Straddle or Long Strangle — Strong buy-vol signal. "
            "ATM or near-ATM strikes for maximum gamma exposure."
        )
    elif abs(spread_pct) > 15:
        return (
            "Debit Spread or Calendar Spread — Moderate buy-vol opportunity. "
            "Consider front-month options for higher gamma."
        )
    else:
        return (
            "Long Call or Long Put — Mild buy-vol signal. "
            "Directional bias may improve risk/reward."
        )


def format_signal_report(signal: dict) -> str:
    """Format a signal dictionary into a readable report."""
    lines = [
        f"\n{'═'*60}",
        f"  TRADING SIGNAL: {signal['ticker']}",
        f"{'═'*60}",
        f"",
        f"  Direction:     {signal['signal']}",
        f"  Strength:      {'█' * (signal['strength'] // 5)}{'░' * (20 - signal['strength'] // 5)} {signal['strength']}/100",
        f"  Vol Regime:    {signal.get('vol_regime', 'N/A')}",
        f"",
        f"  Market IV:     {signal['iv']*100:.2f}%" if signal["iv"] else "  Market IV:     N/A",
        f"  GARCH RV (1d): {signal['rv_forecast']*100:.2f}%",
        f"  Current RV:    {signal['rv_current']*100:.2f}%",
        f"  IV-RV Spread:  {signal['spread']*100:+.2f}pp" if signal["spread"] is not None else "  IV-RV Spread:  N/A",
        f"",
        f"  Rationale:",
        f"    {signal['rationale']}",
        f"",
        f"  Strategy:",
        f"    {signal['strategy']}",
        f"",
        f"  Top Vol Drivers (Transformer):",
    ]

    for driver in signal.get("top_drivers", []):
        lines.append(f"    • {driver}")

    lines.append(f"{'═'*60}")
    return "\n".join(lines)

"""
Quote Validator — validates scan opportunities before execution.

Strategy:
    Webull's Market Data API does NOT support option quotes (404).
    Instead, we use:
        1. Webull stock snapshot → real-time underlying price
        2. yfinance option chain → fresh bid/ask/volume (15-min delayed)

    If the underlying moved significantly since scan time,
    the option premiums from the scan are stale → reject.

Checks:
    1. Spot drift: has the underlying moved > 2% since scan?
    2. Spread width: is bid/ask spread > 15% of mid price?
    3. Volume: do both legs have volume ≥ 10?
    4. Premium sanity: are fresh yfinance premiums within 20% of scan estimate?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import yfinance as yf

log = logging.getLogger("quote_validator")

# ─── Validation Thresholds ───────────────────────────────────────
MAX_SPOT_DRIFT_PCT = 0.02       # Reject if underlying moved > 2% since scan
MAX_SPREAD_PCT = 0.15           # Reject if bid/ask spread > 15% of mid
MAX_PREMIUM_INFLATION = 0.20    # Reject if fresh premium > 20% above scan
MIN_MARGIN_AFTER_VALIDATION = 1.00  # Reject if margin shrinks below $1.00
MIN_VOLUME = 10                 # Minimum volume per leg


def _get_webull_spot(ticker):
    """
    Fetch real-time stock price from Webull's stock snapshot API.
    This endpoint DOES work (unlike the option quote endpoint).
    Returns spot price or None.
    """
    try:
        from broker.webull_client import _call_api
        resp = _call_api("GET", "/openapi/market-data/stock/snapshot",
                         query_params={"symbols": ticker, "category": "US_STOCK"})
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                q = data[0]
            elif isinstance(data, dict):
                q = data.get("data", data)
                if isinstance(q, list) and len(q) > 0:
                    q = q[0]
            else:
                return None

            # Try multiple field names for the price
            price = (q.get('last', None) or q.get('lastPrice', None) or
                     q.get('close', None) or q.get('price', None))
            if price:
                return float(price)
    except Exception as e:
        log.warning(f"   [webull] Stock snapshot failed: {e}")
    return None


def validate_opportunity(scan_result, strategy='straddle'):
    """
    Validate a scan opportunity before execution.

    Uses:
        - Webull stock snapshot for real-time underlying price
        - yfinance for fresh option bid/ask/volume

    Args:
        scan_result: dict from scanner with ticker, strike(s), expiry, premium, margin, etc.
        strategy: 'straddle' or 'strangle'

    Returns:
        dict with:
            'valid': bool — True if opportunity still viable
            'warnings': list of str — issues found (non-fatal)
            'rejections': list of str — deal-breakers (fatal)
            'yahoo': dict — original scan data
            'webull': dict — real-time data (spot price)
            'comparison': dict — side-by-side metrics
    """
    ticker = scan_result['ticker']
    expiry = scan_result['expiry']

    warnings = []
    rejections = []

    scan_spot = scan_result.get('spot', 0)
    scan_premium = scan_result.get('premium', 0)
    scan_margin = scan_result.get('margin', scan_result.get('best_margin', 0))

    # ─── Check 1: Spot Drift (Webull real-time vs scan) ──────────
    wb_spot = _get_webull_spot(ticker)
    if wb_spot and scan_spot > 0:
        drift_pct = abs(wb_spot - scan_spot) / scan_spot
        if drift_pct > MAX_SPOT_DRIFT_PCT:
            rejections.append(
                f"Spot drifted {drift_pct:.1%}: scan ${scan_spot:.2f} → "
                f"Webull ${wb_spot:.2f} (>{MAX_SPOT_DRIFT_PCT:.0%} threshold)")
        elif drift_pct > 0.01:
            warnings.append(
                f"Spot moved {drift_pct:.1%}: ${scan_spot:.2f} → ${wb_spot:.2f}")
        else:
            log.info(f"   ✅ Spot stable: ${scan_spot:.2f} → ${wb_spot:.2f} ({drift_pct:.2%})")
    elif not wb_spot:
        warnings.append("Webull stock snapshot unavailable — skipping spot check")

    # ─── Check 2: Fresh yfinance option data ─────────────────────
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)

        if strategy == 'straddle':
            strike = scan_result.get('strike', scan_result.get('call_strike'))
            c = chain.calls[chain.calls['strike'] == float(strike)]
            p = chain.puts[chain.puts['strike'] == float(strike)]
        else:
            call_strike = scan_result.get('call_strike', scan_result.get('strike'))
            put_strike = scan_result.get('put_strike', scan_result.get('strike'))
            c = chain.calls[chain.calls['strike'] == float(call_strike)]
            p = chain.puts[chain.puts['strike'] == float(put_strike)]

        if c.empty or p.empty:
            rejections.append("Option strike not found in fresh chain")
        else:
            c, p = c.iloc[0], p.iloc[0]
            import pandas as pd

            # Volume check
            c_vol = int(c['volume']) if not pd.isna(c.get('volume', 0)) else 0
            p_vol = int(p['volume']) if not pd.isna(p.get('volume', 0)) else 0
            if c_vol < MIN_VOLUME or p_vol < MIN_VOLUME:
                rejections.append(
                    f"Low volume: call={c_vol}, put={p_vol} (min {MIN_VOLUME})")

            # Spread width check
            for label, opt in [('Call', c), ('Put', p)]:
                bid = float(opt.get('bid', 0) or 0)
                ask = float(opt.get('ask', 0) or 0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    spread_pct = (ask - bid) / mid if mid > 0 else 0
                    if spread_pct > MAX_SPREAD_PCT:
                        rejections.append(
                            f"{label} spread too wide: ${bid:.2f}-${ask:.2f} "
                            f"({spread_pct:.0%} > {MAX_SPREAD_PCT:.0%})")
                    elif spread_pct > 0.10:
                        warnings.append(f"{label} spread {spread_pct:.0%}: ${bid:.2f}-${ask:.2f}")

            # Premium inflation check
            c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (c['bid'] + c['ask']) / 2
            p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (p['bid'] + p['ask']) / 2
            fresh_premium = c_price + p_price

            if scan_premium > 0 and fresh_premium > 0:
                inflation = (fresh_premium - scan_premium) / scan_premium
                if inflation > MAX_PREMIUM_INFLATION:
                    rejections.append(
                        f"Premium inflated {inflation:+.0%}: "
                        f"${scan_premium:.2f} → ${fresh_premium:.2f}")
                elif inflation > 0.10:
                    warnings.append(
                        f"Premium shifted {inflation:+.0%}: "
                        f"${scan_premium:.2f} → ${fresh_premium:.2f}")

                # Margin erosion
                margin_delta = fresh_premium - scan_premium
                fresh_margin = scan_margin - margin_delta
                if fresh_margin < MIN_MARGIN_AFTER_VALIDATION:
                    rejections.append(
                        f"Margin eroded: ${scan_margin:.2f} → ${fresh_margin:.2f} "
                        f"(below ${MIN_MARGIN_AFTER_VALIDATION:.2f} min)")

    except Exception as e:
        warnings.append(f"Fresh yfinance check failed: {e}")

    # ─── Build result ────────────────────────────────────────────
    valid = len(rejections) == 0

    result = {
        'valid': valid,
        'warnings': warnings,
        'rejections': rejections,
        'yahoo': scan_result,
        'webull': {'spot': wb_spot} if wb_spot else None,
        'comparison': {
            'scan_spot': scan_spot,
            'webull_spot': wb_spot,
        },
    }

    # Log summary
    status = "✅ VALID" if valid else "❌ REJECTED"
    log.info(f"   {status}: {ticker}")
    for w in warnings:
        log.info(f"   ⚠️  {w}")
    for r in rejections:
        log.info(f"   ❌ {r}")

    return result


def print_validation(result):
    """Pretty-print a validation result."""
    ticker = result['yahoo']['ticker']
    print(f"\n  {'─'*55}")
    print(f"  {ticker} Quote Validation")
    print(f"  {'─'*55}")

    if result['valid']:
        print(f"  ✅ OPPORTUNITY CONFIRMED")
    else:
        print(f"  ❌ OPPORTUNITY REJECTED")
    for w in result['warnings']:
        print(f"  ⚠️  {w}")
    for r in result['rejections']:
        print(f"  ❌ {r}")
    print()

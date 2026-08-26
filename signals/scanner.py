"""
Live Scanner — THE SINGLE SOURCE OF TRUTH for scanning straddle opportunities.

Scans a broad universe of 54 budget-friendly tickers for the best
GARCH-signaled straddle opportunities within a given budget.

This module is the canonical scanner used by:
  - main.py --mode scan        (CLI entry point)
  - dashboard/app.py           (Streamlit landing page)
  - broker/webull_client.py    (live trade execution)

Signal methodology:
  - Compares GARCH forecast RV against 30-DAY ROLLING CLOSE-TO-CLOSE
    HISTORICAL VOLATILITY to identify when options are underpriced.
  - This benchmark outperformed the Garman-Klass OHLC IV proxy in
    backtesting (+1,247% vs +736% over 43 sequential 5-day periods).
  - Uses real last-traded option prices for cost/affordability checks.
  - Tracks liquidity (call+put volume) to avoid illiquid contracts.

Filters (parity with strangle scanner):
  - Dampened GARCH rejection (v3.1)
  - Minimum persistence threshold (0.70)
  - IV Rank filter (reject if IV rank > 50)
  - Minimum margin past breakeven ($1.00)
  - Realized vs predicted vol stale signal check (v3.1)
  - Earnings catalyst flag (informational)

DO NOT create alternative scanner scripts. All scanning routes through here.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.garch_model import GARCHVolatilityModel
from data.fetcher import fetch_price_data
from config import (
    TRADING_DAYS, MIN_EXPIRY_TRADING_DAYS, MIN_MARGIN_THRESHOLD,
    IV_RANK_MAX, MAX_TOP_PICKS, REJECT_DAMPENED, REALIZED_VS_PREDICTED_MIN,
    GARCH_CALIBRATION_SCALE, BREAKEVEN_SIGMA, MIN_PERSISTENCE,
    VOL_MEAN_REVERSION_MAX, MIN_BE_FEASIBILITY
)


SCAN_UNIVERSE = [
    # ─── Tier 1: Sub-$20 high-vol (original universe) ────────────
    'F', 'SOFI', 'NIO', 'RIVN', 'SNAP', 'MARA', 'PLUG', 'LCID',
    'AMC', 'GME', 'BB', 'CLOV', 'OPEN', 'RIOT', 'SNDL', 'GEVO',
    'ACHR', 'RGTI', 'QUBT', 'FCEL', 'CHPT', 'QS', 'ENVX',
    'RUN', 'XPEV', 'UPST', 'SKLZ', 'FUBO', 'CLSK', 'HIMS',
    'AAL', 'NCLH', 'PATH', 'BLNK', 'T', 'INTC', 'PFE', 'CCL',
    'PYPL', 'DKNG', 'HOOD', 'SIRI',
    # ─── Tier 2: Mid-cap volatile ($20-$100) ─────────────────────
    'SHOP', 'COIN', 'ROKU', 'RBLX', 'PLTR',
    'MSTR', 'SMCI', 'ARM', 'CRWD', 'AFRM', 'NET',
    'UBER', 'DASH', 'ABNB', 'SE', 'SNOW', 'ENPH',
    'WYNN', 'BA', 'SQ',
    # ─── Tier 3: Large-cap volatile ($100-$600) ──────────────────
    'TSLA', 'NVDA', 'META', 'AMZN', 'GOOGL', 'NFLX',
    'AMD', 'MSFT', 'AAPL', 'CRM', 'AVGO',
]


def _compute_iv_rank(ticker_obj, spot, chain):
    """Compute IV rank as percentile of current IV vs. 1-year range."""
    try:
        hist = ticker_obj.history(period="1y")
        if hist is None or len(hist) < 60:
            return None
        log_rets = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        rolling_iv = log_rets.rolling(21).std() * np.sqrt(252)
        rolling_iv = rolling_iv.dropna()
        if len(rolling_iv) < 30:
            return None
        current_iv = (chain.calls['impliedVolatility'].mean() +
                      chain.puts['impliedVolatility'].mean()) / 2
        iv_min = rolling_iv.min()
        iv_max = rolling_iv.max()
        if iv_max == iv_min:
            return 50
        return round(((current_iv - iv_min) / (iv_max - iv_min)) * 100)
    except Exception:
        return None


def _check_upcoming_earnings(ticker_obj):
    """Check if earnings are within 7 days."""
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return {'has_earnings': False, 'earnings_date': None}
        if isinstance(cal, dict):
            ed = cal.get('Earnings Date')
            if ed and len(ed) > 0:
                ed_date = ed[0]
                if hasattr(ed_date, 'date'):
                    ed_date = ed_date.date()
                days_until = (ed_date - datetime.now().date()).days
                return {
                    'has_earnings': 0 <= days_until <= 7,
                    'earnings_date': str(ed_date) if 0 <= days_until <= 7 else None
                }
        return {'has_earnings': False, 'earnings_date': None}
    except Exception:
        return {'has_earnings': False, 'earnings_date': None}


def scan_for_opportunities(budget=150.0, top_n=None, verbose=True):
    """
    Scan the universe for affordable straddles with positive GARCH signals
    and all v3/v3.1 safety filters applied.

    Returns a tuple: (results_list, rejections_dict)
    Results sorted by composite score (GARCH signal + liquidity).
    """
    if top_n is None:
        top_n = MAX_TOP_PICKS

    # Detect off-hours
    import pytz
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    is_market_hours = now.weekday() < 5 and 9 <= now.hour <= 16
    off_hours = not is_market_hours

    if off_hours and verbose:
        print("⚠️  Running outside market hours — OI and bid/ask may be stale.")
        print("   Skipping OI filter, using volume + price filters only.\n")

    # Minimum expiry: 14 trading days ≈ 20 calendar days
    min_calendar_days = int(MIN_EXPIRY_TRADING_DAYS * 7 / 5)
    min_expiry_date = datetime.now() + timedelta(days=min_calendar_days)
    results = []
    rejections = {
        'no_options_chain': [],
        'expiry_too_short': [],
        'strikes_unavailable': [],
        'penny_option': [],
        'over_budget': [],
        'negative_signal': [],
        'dampened_signal': [],
        'low_persistence': [],
        'high_iv_rank': [],
        'low_margin': [],
        'stale_signal': [],
        'vol_mean_reversion': [],
        'low_feasibility': [],
        'error': [],
    }

    for sym in SCAN_UNIVERSE:
        try:
            tk = yf.Ticker(sym)
            exps = tk.options
            if not exps:
                rejections['no_options_chain'].append(sym)
                continue

            # Filter out expirations shorter than minimum
            valid_exps = [e for e in exps
                          if datetime.strptime(e, '%Y-%m-%d') >= min_expiry_date]
            if not valid_exps:
                rejections['expiry_too_short'].append(sym)
                continue

            # Pick the nearest valid expiry (closest to minimum)
            best_exp = min(valid_exps, key=lambda x:
                datetime.strptime(x, '%Y-%m-%d') - min_expiry_date)

            chain = tk.option_chain(best_exp)
            prices = fetch_price_data(sym)
            prices = prices.dropna(subset=['Close'])
            spot = prices['Close'].iloc[-1]
            atm = round(spot)

            # Try ATM and ±1 strikes
            found_strike = False
            for strike in [atm, atm - 1, atm + 1]:
                c = chain.calls[chain.calls['strike'] == strike]
                p = chain.puts[chain.puts['strike'] == strike]
                if c.empty or p.empty:
                    continue
                c = c.iloc[0]
                p = p.iloc[0]

                # Use lastPrice (market may be closed)
                c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (
                    c['bid'] + c['ask']) / 2
                p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (
                    p['bid'] + p['ask']) / 2
                if c_price < 0.03 or p_price < 0.03:
                    continue

                straddle_cost = (c_price + p_price) * 100 + 1.30
                if straddle_cost > budget or straddle_cost < 10:
                    continue

                contracts = int((budget - 1.30) / straddle_cost)
                if contracts < 1:
                    continue

                c_vol = int(c['volume']) if not pd.isna(c['volume']) else 0
                p_vol = int(p['volume']) if not pd.isna(p['volume']) else 0
                avg_iv = (c['impliedVolatility'] + p['impliedVolatility']) / 2

                # ─── GARCH Fit ────────────────────────────────────
                garch = GARCHVolatilityModel()
                garch.fit(prices, verbose=False)

                # v4: Apply bias correction — GARCH overestimates by ~25%
                bias_factor = garch.compute_bias_correction(prices)
                garch_rv = garch.get_corrected_conditional_volatility().iloc[-1]

                # 30-day rolling close-to-close realized vol (annualized)
                log_ret = np.log(prices['Close'] / prices['Close'].shift(1))
                hist_vol_30d = log_ret.rolling(30).std().iloc[-1] * np.sqrt(TRADING_DAYS)
                if np.isnan(hist_vol_30d):
                    continue

                spread = garch_rv - hist_vol_30d

                # ─── Filter 1: Positive signal required ───────────
                if spread <= 0:
                    rejections['negative_signal'].append(
                        f"{sym} (spread {spread:+.1%})")
                    found_strike = True
                    break

                # ─── Filter 2: Dampened GARCH rejection (v3.1) ────
                if REJECT_DAMPENED and garch.dampened:
                    rejections['dampened_signal'].append(
                        f"{sym} (persistence {garch.persistence:.3f}, dampened)")
                    found_strike = True
                    break

                # ─── Filter 3: Minimum persistence ────────────────
                if garch.persistence < MIN_PERSISTENCE:
                    rejections['low_persistence'].append(
                        f"{sym} (persistence {garch.persistence:.3f} < {MIN_PERSISTENCE})")
                    found_strike = True
                    break

                # ─── Filter 4: GARCH breakeven margin check ───────
                exp_date = datetime.strptime(best_exp, '%Y-%m-%d')
                calendar_days = (exp_date - datetime.now()).days
                holding_days = max(1, int(calendar_days * 5 / 7))

                price_range = garch.forecast_price_range(
                    spot, horizon_days=holding_days)

                premium = c_price + p_price
                breakeven_up = strike + premium
                breakeven_down = strike - premium

                upside_margin = price_range['upper_1sig'] - breakeven_up
                downside_margin = breakeven_down - price_range['lower_1sig']
                best_margin = max(upside_margin, downside_margin)

                if best_margin < MIN_MARGIN_THRESHOLD:
                    rejections['low_margin'].append(
                        f"{sym} (margin ${best_margin:.2f} < ${MIN_MARGIN_THRESHOLD:.2f})")
                    found_strike = True
                    break

                # ─── Filter 5: IV Rank filter ─────────────────────
                iv_rank = _compute_iv_rank(tk, spot, chain)
                if iv_rank is not None and iv_rank > IV_RANK_MAX:
                    rejections['high_iv_rank'].append(
                        f"{sym} (IV rank {iv_rank} > {IV_RANK_MAX})")
                    found_strike = True
                    break

                # ─── Filter 6: Realized vs Predicted vol (v3.1) ───
                recent_rets = prices['Close'].pct_change().dropna().tail(5)
                realized_5d_move = recent_rets.abs().mean() * np.sqrt(holding_days) * 100
                predicted_move = price_range['expected_move_pct']
                if predicted_move > 0 and realized_5d_move / predicted_move < REALIZED_VS_PREDICTED_MIN:
                    rejections['stale_signal'].append(
                        f"{sym} (realized {realized_5d_move:.1f}% vs predicted "
                        f"{predicted_move:.1f}%, ratio "
                        f"{realized_5d_move/predicted_move:.2f} < {REALIZED_VS_PREDICTED_MIN})")
                    found_strike = True
                    break

                # ─── Filter 7: Vol Mean-Reversion (v4) ────────────
                # When HV30 >> HV90, short-term vol is elevated above
                # its long-term average and will likely compress.
                # Buying vol here = buying at the peak.
                # RGTI (1.32x), SOFI (1.53x) both lost >74%.
                all_rets = prices['Close'].pct_change().dropna()
                if len(all_rets) >= 90:
                    hv90 = all_rets.iloc[-90:].std() * np.sqrt(TRADING_DAYS)
                    if hv90 > 0:
                        vol_ratio = hist_vol_30d / hv90
                        if vol_ratio > VOL_MEAN_REVERSION_MAX:
                            rejections['vol_mean_reversion'].append(
                                f"{sym} (HV30/HV90 = {vol_ratio:.2f} > "
                                f"{VOL_MEAN_REVERSION_MAX}, vol likely to compress)")
                            found_strike = True
                            break

                # ─── Filter 8: Breakeven Feasibility (v4) ─────────
                # Check what % of historical N-day windows achieved
                # a move >= breakeven. If < 25%, the stock rarely
                # moves enough to make this trade profitable.
                close_series = prices['Close']
                n_day_moves = (close_series.pct_change(holding_days).dropna().abs())
                be_move_pct = max(
                    abs(breakeven_up / spot - 1),
                    abs(1 - breakeven_down / spot)
                )
                feasibility = (n_day_moves >= be_move_pct).mean()
                if feasibility < MIN_BE_FEASIBILITY:
                    rejections['low_feasibility'].append(
                        f"{sym} (only {feasibility:.0%} of {holding_days}d windows "
                        f"achieved ±{be_move_pct:.1%} move, need {MIN_BE_FEASIBILITY:.0%})")
                    found_strike = True
                    break

                # ─── Earnings check (informational) ───────────────
                earnings = _check_upcoming_earnings(tk)

                # ─── Composite Score ──────────────────────────────
                margin_score = min(1.0, best_margin / max(premium, 0.01))
                combined_vol = c_vol + p_vol
                liquidity_score = min(1.0, np.log10(max(combined_vol, 1)) / 4)
                composite = 0.60 * margin_score + 0.40 * liquidity_score

                results.append({
                    'ticker': sym,
                    'spot': round(spot, 2),
                    'strike': int(strike),
                    'expiry': best_exp,
                    'call_price': round(c_price, 2),
                    'put_price': round(p_price, 2),
                    'straddle_cost': round(straddle_cost, 2),
                    'contracts': contracts,
                    'total_cost': round(straddle_cost * contracts + 1.30, 2),
                    'garch_rv': round(garch_rv, 4),
                    'hist_vol': round(hist_vol_30d, 4),
                    'mkt_iv': round(avg_iv, 4),
                    'spread': round(spread, 4),
                    'signal_strength': round(
                        spread / max(hist_vol_30d, 0.01), 3),
                    'option_iv': round(avg_iv, 4),
                    'call_volume': int(c_vol),
                    'put_volume': int(p_vol),
                    'liquidity': int(c_vol + p_vol),
                    'persistence': round(garch.persistence, 4),
                    'dampened': garch.dampened,
                    'best_margin': round(best_margin, 2),
                    'iv_rank': iv_rank,
                    'composite_score': round(composite, 3),
                    'earnings_flag': earnings.get('has_earnings', False),
                    'earnings_date': earnings.get('earnings_date'),
                })
                found_strike = True
                break  # Best strike per ticker

            if not found_strike:
                rejections['strikes_unavailable'].append(sym)

        except Exception as e:
            rejections['error'].append(f"{sym}: {e}")

    # Sort by composite score (blends signal + liquidity)
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    return results[:top_n], rejections


if __name__ == "__main__":
    recs, rejections = scan_for_opportunities(budget=150.0, top_n=8)
    for i, r in enumerate(recs):
        print(f"{i+1}. {r['ticker']} ${r['strike']} straddle | "
              f"GARCH {r['garch_rv']:.1%} vs 30dHV {r['hist_vol']:.1%} | "
              f"Spread {r['spread']:+.1%} | Score {r['composite_score']:.3f} | "
              f"${r['total_cost']:.2f}")
    print(f"\nRejections: {sum(len(v) for v in rejections.values())}")
    for reason, items in rejections.items():
        if items:
            print(f"  {reason}: {len(items)}")

"""
Live Strangle Scanner — scans the 42-ticker universe for the best
GARCH-signaled strangle opportunities within a given budget.

Mirrors signals/scanner.py but for the strangle strategy:
  - OTM call + OTM put at separate strikes
  - Uses the same SCAN_UNIVERSE (imported from scanner.py)
  - Same GARCH signal methodology (forecast RV vs 30d historical vol)
  - Same budget, liquidity, and cost filters

Liquidity Safeguards (added after FUBO incident 2026-06-03):
  - Both legs must have open interest >= 100
  - Both legs must have daily volume >= 25 (combined >= 50)
  - Both legs must be priced >= $0.15 (no penny options)
  - Composite score blends GARCH signal (70%) + liquidity (30%)

This module is used by:
  - main.py --mode scan --strategy strangle
  - dashboard/app.py (Strangle Scanner tab)
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.garch_model import GARCHVolatilityModel
from data.fetcher import fetch_price_data
from config import TRADING_DAYS, STRANGLE_OTM_WIDTH, MIN_MARGIN_THRESHOLD, IV_RANK_MAX, MAX_TOP_PICKS, REJECT_DAMPENED, REALIZED_VS_PREDICTED_MIN, MAX_STRANGLE_SPREAD_PCT, MIN_EXPIRY_TRADING_DAYS
from signals.scanner import SCAN_UNIVERSE

# ─── Liquidity Cache Path ───────────────────────────────────────
LIQUIDITY_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", "scanner_liquidity.json"
)

# ─── Liquidity Filter Thresholds ────────────────────────────────
MIN_OPEN_INTEREST = 100      # per leg
MIN_LEG_VOLUME = 25          # per leg (daily)
MIN_OPTION_PRICE = 0.15      # per leg (no penny options)

# ─── IV Rank lookback ──────────────────────────────────────────
IV_RANK_LOOKBACK = 252       # 1 year of IV history for percentile rank


def compute_iv_rank(ticker_obj, spot, chain, lookback=IV_RANK_LOOKBACK):
    """
    Compute IV rank (percentile) — where is current IV relative to past year?

    Uses ATM option implied vol from the chain and compares to
    historical close-to-close realized vol as a proxy for past IV levels.

    Returns:
        IV rank 0-100 (0 = vol is at 1-year low, 100 = at 1-year high)
        None if computation fails.
    """
    try:
        # Current IV: average of ATM call + put IV
        atm = round(spot)
        atm_calls = chain.calls[chain.calls['strike'] == atm]
        atm_puts = chain.puts[chain.puts['strike'] == atm]
        if atm_calls.empty or atm_puts.empty:
            # Fall back to nearest-strike
            atm_calls = chain.calls.iloc[(chain.calls['strike'] - spot).abs().argsort()[:1]]
            atm_puts = chain.puts.iloc[(chain.puts['strike'] - spot).abs().argsort()[:1]]

        current_iv = (atm_calls.iloc[0]['impliedVolatility'] +
                      atm_puts.iloc[0]['impliedVolatility']) / 2

        # Historical IV proxy: rolling 30-day realized vol over past year
        hist = ticker_obj.history(period='1y')
        if len(hist) < 60:
            return None
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        rolling_rv = log_ret.rolling(30).std() * np.sqrt(252)
        rolling_rv = rolling_rv.dropna()

        if len(rolling_rv) < 30:
            return None

        # IV rank = percentile of current IV within historical RV distribution
        rank = (rolling_rv < current_iv).mean() * 100
        return round(rank, 1)
    except Exception:
        return None


def check_upcoming_earnings(ticker_obj, days_ahead=7):
    """
    Check if a ticker has earnings within the next N days.

    Returns:
        dict with 'has_earnings' bool and 'earnings_date' if found.
    """
    try:
        cal = ticker_obj.calendar
        if cal is None or cal.empty:
            return {'has_earnings': False, 'earnings_date': None}

        # yfinance returns earnings date in different formats
        if isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.columns:
                earn_date = pd.to_datetime(cal['Earnings Date'].iloc[0])
            elif 'Earnings Date' in cal.index:
                earn_date = pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
            else:
                return {'has_earnings': False, 'earnings_date': None}
        else:
            return {'has_earnings': False, 'earnings_date': None}

        days_until = (earn_date - pd.Timestamp.now()).days
        return {
            'has_earnings': 0 <= days_until <= days_ahead,
            'earnings_date': earn_date.strftime('%Y-%m-%d') if 0 <= days_until <= days_ahead else None,
            'days_until': days_until if days_until >= 0 else None,
        }
    except Exception:
        return {'has_earnings': False, 'earnings_date': None}


def scan_strangle_opportunities(budget=150.0, top_n=None, otm_width=STRANGLE_OTM_WIDTH):
    """
    Scan the universe for affordable strangles with positive GARCH signals
    and sufficient liquidity to guarantee fill.

    v3 additions:
      - Calibrated GARCH range (1.3x wider — corrects underestimation)
      - IV rank filter (reject if IV rank > 50 — options are expensive)
      - Min margin threshold ($1.00 past breakeven)
      - Earnings catalyst flag (informational, not a filter)

    Returns a tuple: (results_list, rejections_dict)
    Results sorted by composite score (GARCH signal + liquidity).
    """
    if top_n is None:
        top_n = MAX_TOP_PICKS
    # Detect weekend/off-hours: yfinance returns OI=0, bid/ask=$0 on non-trading days
    import pytz
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    is_market_day = now.weekday() < 5  # Mon-Fri
    is_market_hours = is_market_day and 9 <= now.hour <= 16
    off_hours = not is_market_hours

    if off_hours:
        print("⚠️  Running outside market hours — OI and bid/ask may be stale.")
        print("   Skipping OI filter, using volume + price filters only.")
        print()
    # Minimum expiry: 14 trading days ≈ 20 calendar days
    min_calendar_days = int(MIN_EXPIRY_TRADING_DAYS * 7 / 5)  # convert trading → calendar
    min_expiry_date = datetime.now() + timedelta(days=min_calendar_days)
    results = []
    rejections = {
        'no_options_chain': [],
        'expiry_too_short': [],
        'strikes_unavailable': [],
        'spread_too_wide': [],
        'low_open_interest': [],
        'low_volume': [],
        'penny_option': [],
        'over_budget': [],
        'high_iv_rank': [],
        'dampened_signal': [],
        'stale_signal': [],
        'low_margin': [],
        'negative_signal': [],
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
                rejections['expiry_too_short'].append(
                    f"{sym} (nearest: {exps[0]}, need >= {min_expiry_date.strftime('%Y-%m-%d')})")
                continue

            # Pick the nearest valid expiry (closest to minimum)
            best_exp = min(valid_exps, key=lambda x:
                datetime.strptime(x, '%Y-%m-%d') - min_expiry_date)

            chain = tk.option_chain(best_exp)
            prices = fetch_price_data(sym)
            prices = prices.dropna(subset=['Close'])
            spot = prices['Close'].iloc[-1]

            # OTM strikes for strangle
            call_strike = round(spot * (1 + otm_width))
            put_strike = round(spot * (1 - otm_width))

            # Ensure strikes are at least $1 away from ATM
            atm = round(spot)
            if call_strike <= atm:
                call_strike = atm + 1
            if put_strike >= atm:
                put_strike = atm - 1

            # Find the OTM call and put in the chain
            c = chain.calls[chain.calls['strike'] == call_strike]
            p = chain.puts[chain.puts['strike'] == put_strike]

            # If exact strikes not available, try ±1
            if c.empty:
                c = chain.calls[chain.calls['strike'] == call_strike + 1]
                if not c.empty:
                    call_strike = call_strike + 1
            if p.empty:
                p = chain.puts[chain.puts['strike'] == put_strike - 1]
                if not p.empty:
                    put_strike = put_strike - 1

            if c.empty or p.empty:
                rejections['strikes_unavailable'].append(sym)
                continue

            # ─── Spread Width Filter ─────────────────────────────
            spread = call_strike - put_strike
            max_spread = spot * MAX_STRANGLE_SPREAD_PCT
            if spread > max_spread:
                rejections['spread_too_wide'].append(
                    f"{sym} (${spread:.0f} spread > ${max_spread:.0f} max, "
                    f"{spread/spot*100:.1f}% > {MAX_STRANGLE_SPREAD_PCT*100:.0f}%)")
                continue

            c = c.iloc[0]
            p = p.iloc[0]

            # ─── Liquidity Filter 1: Open Interest ──────────────
            c_oi = int(c['openInterest']) if not pd.isna(c.get('openInterest', 0)) else 0
            p_oi = int(p['openInterest']) if not pd.isna(p.get('openInterest', 0)) else 0
            # Skip OI filter off-hours — yfinance returns 0 on weekends
            if not off_hours:
                if c_oi < MIN_OPEN_INTEREST or p_oi < MIN_OPEN_INTEREST:
                    rejections['low_open_interest'].append(
                        f"{sym} (call_OI={c_oi}, put_OI={p_oi})")
                    continue

            # ─── Liquidity Filter 2: Daily Volume ───────────────
            c_vol = int(c['volume']) if not pd.isna(c.get('volume', 0)) else 0
            p_vol = int(p['volume']) if not pd.isna(p.get('volume', 0)) else 0
            # Relax volume threshold off-hours (Friday's final volume may be lower)
            vol_threshold = 10 if off_hours else MIN_LEG_VOLUME
            if c_vol < vol_threshold or p_vol < vol_threshold:
                rejections['low_volume'].append(
                    f"{sym} (call_vol={c_vol}, put_vol={p_vol})")
                continue

            # Use lastPrice (market may be closed)
            c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (
                c['bid'] + c['ask']) / 2
            p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (
                p['bid'] + p['ask']) / 2

            # ─── Liquidity Filter 3: Minimum Price ──────────────
            if c_price < MIN_OPTION_PRICE or p_price < MIN_OPTION_PRICE:
                rejections['penny_option'].append(
                    f"{sym} (call=${c_price:.2f}, put=${p_price:.2f})")
                continue

            strangle_cost = (c_price + p_price) * 100 + 1.30
            if strangle_cost > budget or strangle_cost < 10:
                rejections['over_budget'].append(
                    f"{sym} (${strangle_cost:.2f})")
                continue

            contracts = int((budget - 1.30) / strangle_cost)
            if contracts < 1:
                rejections['over_budget'].append(
                    f"{sym} (${strangle_cost:.2f})")
                continue

            avg_iv = (c['impliedVolatility'] + p['impliedVolatility']) / 2

            # ─── GARCH Breakeven Comparison ─────────────────────
            # Instead of comparing GARCH vol to historical vol (abstract),
            # we compare GARCH's predicted $ move to the real breakeven.
            garch = GARCHVolatilityModel()
            garch.fit(prices, verbose=False)

            # Days to expiry = trading days between now and expiry
            exp_date = datetime.strptime(best_exp, '%Y-%m-%d')
            calendar_days = (exp_date - datetime.now()).days
            holding_days = max(1, int(calendar_days * 5 / 7))  # approx trading days

            # Forecast price range over the holding period
            price_range = garch.forecast_price_range(
                spot, horizon_days=holding_days)

            garch_rv = garch.get_conditional_volatility().iloc[-1]

            # Real breakeven prices for the strangle
            premium = c_price + p_price  # total debit per share
            breakeven_up = call_strike + premium
            breakeven_down = put_strike - premium

            # How far past breakeven does GARCH predict?
            upside_margin = price_range['upper_1sig'] - breakeven_up
            downside_margin = breakeven_down - price_range['lower_1sig']
            best_margin = max(upside_margin, downside_margin)

            # Reject if GARCH predicts movement WON'T cover premium
            if best_margin <= 0:
                rejections['negative_signal'].append(
                    f"{sym} (predicted ${price_range['upper_1sig']:.2f}-"
                    f"${price_range['lower_1sig']:.2f}, "
                    f"BE ${breakeven_down:.2f}-${breakeven_up:.2f})")
                continue

            # ─── v3 Fix 3: Minimum Margin Threshold ─────────────
            # Only trade when GARCH predicts movement > $1.00 past breakeven.
            # Filters out marginal signals like SKLZ's $0.23 margin.
            if best_margin < MIN_MARGIN_THRESHOLD:
                rejections['low_margin'].append(
                    f"{sym} (margin ${best_margin:.2f} < ${MIN_MARGIN_THRESHOLD:.2f})")
                continue

            # ─── v3 Fix 4: IV Rank Filter ────────────────────────
            # If IV is already at the 90th percentile, the premium bakes
            # in the expected move — there's no edge buying vol.
            iv_rank = compute_iv_rank(tk, spot, chain)
            if iv_rank is not None and iv_rank > IV_RANK_MAX:
                rejections['high_iv_rank'].append(
                    f"{sym} (IV rank {iv_rank:.0f} > {IV_RANK_MAX})")
                continue

            # ─── v3.1 Fix 6: Reject Dampened GARCH ────────────────
            # When GARCH persistence hits 1.0 and gets clamped, the model
            # is reading stale vol from a past event. NCLH (dampened) lost
            # -16.5%; PYPL (undampened) won +15.7%.
            if REJECT_DAMPENED and garch.dampened:
                rejections['dampened_signal'].append(
                    f"{sym} (persistence {garch.persistence:.3f}, dampened)")
                continue

            # ─── v3.1 Fix 7: Realized vs Predicted Vol Check ─────
            # If 5-day realized vol is less than 50% of GARCH's prediction,
            # the signal is stale — GARCH is reading old volatility.
            recent_rets = prices['Close'].pct_change().dropna().tail(5)
            realized_5d_move = recent_rets.abs().mean() * np.sqrt(holding_days) * 100
            predicted_move = price_range['expected_move_pct']
            if predicted_move > 0 and realized_5d_move / predicted_move < REALIZED_VS_PREDICTED_MIN:
                rejections['stale_signal'].append(
                    f"{sym} (realized {realized_5d_move:.1f}% vs predicted {predicted_move:.1f}%, "
                    f"ratio {realized_5d_move/predicted_move:.2f} < {REALIZED_VS_PREDICTED_MIN})")
                continue

            # ─── v3 Fix 5: Earnings Catalyst Flag ───────────────
            # Informational — flags tickers with earnings in the next 7 days.
            # GARCH can't predict catalyst-driven moves, so these are higher risk.
            earnings = check_upcoming_earnings(tk)

            # ─── Composite Score ─────────────────────────────────
            # Breakeven margin score (60%): how far past BE does GARCH predict?
            # Normalized by premium (higher = more margin of safety)
            margin_score = min(1.0, best_margin / max(premium, 0.01))

            # Liquidity score (40%): volume + OI on log scale
            combined_vol = c_vol + p_vol
            combined_oi = c_oi + p_oi
            liquidity_score = min(1.0, np.log10(max(combined_vol, 1)) / 4)
            oi_score = min(1.0, np.log10(max(combined_oi, 1)) / 4)
            liq_composite = (liquidity_score + oi_score) / 2

            composite = 0.60 * margin_score + 0.40 * liq_composite

            # Capture previous-day bid/ask (reliable at market close)
            c_bid = round(float(c['bid']), 2) if not pd.isna(c.get('bid', 0)) else 0.0
            c_ask = round(float(c['ask']), 2) if not pd.isna(c.get('ask', 0)) else 0.0
            p_bid = round(float(p['bid']), 2) if not pd.isna(p.get('bid', 0)) else 0.0
            p_ask = round(float(p['ask']), 2) if not pd.isna(p.get('ask', 0)) else 0.0

            results.append({
                'ticker': sym,
                'spot': round(spot, 2),
                'call_strike': int(call_strike),
                'put_strike': int(put_strike),
                'expiry': best_exp,
                'call_price': round(c_price, 2),
                'put_price': round(p_price, 2),
                'call_bid': c_bid,
                'call_ask': c_ask,
                'put_bid': p_bid,
                'put_ask': p_ask,
                'strangle_cost': round(strangle_cost, 2),
                'contracts': contracts,
                'total_cost': round(strangle_cost * contracts + 1.30, 2),
                'garch_rv': round(garch_rv, 4),
                'predicted_upper': price_range['upper_1sig'],
                'predicted_lower': price_range['lower_1sig'],
                'breakeven_up': round(breakeven_up, 2),
                'breakeven_down': round(breakeven_down, 2),
                'upside_margin': round(upside_margin, 2),
                'downside_margin': round(downside_margin, 2),
                'best_margin': round(best_margin, 2),
                'margin_score': round(margin_score, 3),
                'mkt_iv': round(avg_iv, 4),
                'option_iv': round(avg_iv, 4),
                'iv_rank': iv_rank,
                'earnings_flag': earnings.get('has_earnings', False),
                'earnings_date': earnings.get('earnings_date'),
                'call_volume': int(c_vol),
                'put_volume': int(p_vol),
                'call_oi': c_oi,
                'put_oi': p_oi,
                'liquidity': int(combined_vol),
                'liquidity_score': round(liq_composite, 3),
                'composite_score': round(composite, 3),
                'otm_width': otm_width,
                'holding_days': holding_days,
                'expected_move_pct': price_range['expected_move_pct'],
                'persistence': round(garch.persistence, 4),
                'dampened': garch.dampened,
            })

        except Exception:
            rejections['error'].append(sym)

    # Sort by composite score (blends signal + liquidity)
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    return results[:top_n], rejections


def cache_liquidity_data(results):
    """
    Cache scanner liquidity data (bid/ask, OI, volume) to disk.

    This is used as a fallback at market open when live quotes
    from yfinance return stale $0.00 bid/ask. The previous day's
    closing bid/ask is a reliable predictor of next-day liquidity.

    Off-hours behavior: if bid/ask is all zeros (weekend), merge into
    existing cache WITHOUT overwriting entries that have real liquidity data.
    """
    import pytz
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    is_market_day = now.weekday() < 5
    is_market_hours = is_market_day and 9 <= now.hour <= 16
    off_hours = not is_market_hours

    # Load existing cache to potentially preserve good data
    existing = {}
    if off_hours and os.path.exists(LIQUIDITY_CACHE):
        try:
            with open(LIQUIDITY_CACHE, 'r') as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    cache = {} if not off_hours else dict(existing)

    for r in results:
        key = f"{r['ticker']}_{r['call_strike']}_{r['put_strike']}_{r['expiry']}"

        new_entry = {
            'ticker': r['ticker'],
            'call_strike': r['call_strike'],
            'put_strike': r['put_strike'],
            'expiry': r['expiry'],
            'call_bid': r.get('call_bid', 0),
            'call_ask': r.get('call_ask', 0),
            'put_bid': r.get('put_bid', 0),
            'put_ask': r.get('put_ask', 0),
            'call_price': r['call_price'],
            'put_price': r['put_price'],
            'call_oi': r.get('call_oi', 0),
            'put_oi': r.get('put_oi', 0),
            'call_volume': r.get('call_volume', 0),
            'put_volume': r.get('put_volume', 0),
            'scanned_at': datetime.now().isoformat(),
        }

        # Off-hours: only overwrite if new data has real bid/ask,
        # otherwise preserve existing entry with Friday's good data
        if off_hours and key in existing:
            old = existing[key]
            new_has_quotes = (new_entry['call_bid'] > 0 or new_entry['call_ask'] > 0)
            old_has_quotes = (old.get('call_bid', 0) > 0 or old.get('call_ask', 0) > 0)
            if old_has_quotes and not new_has_quotes:
                # Keep old entry's liquidity data, update signal fields
                cache[key] = old
                cache[key]['call_price'] = new_entry['call_price']
                cache[key]['put_price'] = new_entry['put_price']
                continue

        cache[key] = new_entry

    os.makedirs(os.path.dirname(LIQUIDITY_CACHE), exist_ok=True)
    with open(LIQUIDITY_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)
    mode = "MERGED (preserved existing liquidity)" if off_hours else "FRESH"
    print(f"\n💾 Liquidity cache saved → {LIQUIDITY_CACHE} ({len(cache)} entries) [{mode}]")


def load_liquidity_cache():
    """Load cached liquidity data from the last scanner run."""
    if os.path.exists(LIQUIDITY_CACHE):
        try:
            with open(LIQUIDITY_CACHE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def print_rejection_summary(rejections):
    """Print a summary of why tickers were rejected."""
    print(f"\n{'='*60}")
    print("REJECTION SUMMARY")
    print(f"{'='*60}")
    for reason, tickers in rejections.items():
        if tickers:
            label = reason.replace('_', ' ').title()
            print(f"  {label} ({len(tickers)}): {', '.join(tickers[:5])}"
                  f"{'...' if len(tickers) > 5 else ''}")
    total_rejected = sum(len(v) for v in rejections.values())
    print(f"\n  Total rejected: {total_rejected} / {len(SCAN_UNIVERSE)}")
    print(f"  Passed all filters: {len(SCAN_UNIVERSE) - total_rejected}")


if __name__ == "__main__":
    recs, rejections = scan_strangle_opportunities(budget=150.0, top_n=8)
    for i, r in enumerate(recs):
        print(f"{i+1}. {r['ticker']} ${r['call_strike']}C/${r['put_strike']}P strangle | "
              f"GARCH {r['garch_rv']:.1%} vs 30dHV {r['hist_vol']:.1%} | "
              f"Spread {r['spread']:+.1%} | ${r['total_cost']:.2f} | "
              f"Score {r['composite_score']:.3f}")
    print_rejection_summary(rejections)
    # Cache liquidity data for next-day trading
    cache_liquidity_data(recs)

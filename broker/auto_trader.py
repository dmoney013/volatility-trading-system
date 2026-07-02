"""
Auto-Trader Daemon — fully autonomous open + close for a single straddle.

This script:
  1. Waits for market open (9:30 AM ET)
  2. Opens a long straddle using fresh quotes (no stale scanner prices)
  3. Monitors the position and auto-closes at +12% take-profit or -30% stop-loss
  4. Uses fresh bid prices at sell time (fixes the stale last_price bug)

Usage:
    python broker/auto_trader.py                    # Live execution
    python broker/auto_trader.py --dry-run          # Simulate without real orders
    nohup python broker/auto_trader.py &            # Background daemon

The trade parameters (ticker, strike, expiry) are set via command-line args
or loaded from cache/active_trade.json if resuming after a restart.
"""
import sys
import os
import time
import json
import argparse
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import broker.webull_client as wb
from config import TAKE_PROFIT_PCT, STOP_LOSS_PCT
from broker.webull_client import (
    get_accounts, place_straddle, close_straddle, get_positions,
)
from broker.position_tracker import fetch_live_positions
from broker.auto_close import (
    TARGET_RETURN_PCT,
    is_market_hours, is_market_open_rush,
    _log_closed_trade,
)

# ─── Configuration ───────────────────────────────────────────────
# Default active trade file — overridden by --trade-file CLI arg
# to support multiple simultaneous daemons.
ACTIVE_TRADE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cache", "active_trade.json"
)
LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cache", "auto_trader.log"
)
POLL_INTERVAL_MARKET = 30       # seconds between checks during market hours
POLL_INTERVAL_OFF = 300         # seconds between checks outside market hours
POLL_INTERVAL_RUSH = 10         # aggressive polling first 5 min after open

# ─── Logging ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

log = logging.getLogger("auto_trader")
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(ch)

fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(fh)


# ─── Market Timing ──────────────────────────────────────────────

def _get_et_now():
    """Get current time in US/Eastern."""
    import pytz
    return datetime.now(pytz.timezone("US/Eastern"))


def wait_for_market_open():
    """
    Block until US market opens (9:30 AM ET). Returns immediately if already open.

    IMPORTANT: Uses short sleep intervals (max 15s) and always re-checks the
    real clock after waking. This is critical because macOS laptop sleep causes
    time.sleep() to freeze for the entire hibernation duration — a 5-minute
    sleep can actually last hours if the lid is closed. By sleeping in tiny
    chunks and checking the clock each time, we catch market open within
    seconds of waking from hibernation.
    """
    from datetime import timedelta as td
    last_log_min = -1  # Track last logged minute to avoid spam

    while True:
        now = _get_et_now()

        # Check if it's a weekday
        if now.weekday() >= 5:
            next_monday = now + td(days=(7 - now.weekday()))
            next_open = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
            wait_secs = (next_open - now).total_seconds()
            log.info(f"Weekend detected. Next market open in {wait_secs/3600:.1f} hours.")
            # Sleep max 15 minutes on weekends, re-check clock each time
            time.sleep(min(wait_secs, 900))
            continue

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now >= market_open and now < market_close:
            log.info("🟢 Market is OPEN. Proceeding.")
            return

        if now < market_open:
            remaining = (market_open - now).total_seconds()
            current_min = int(remaining / 60)

            # Log progress (but don't spam — only log when minute changes)
            if current_min != last_log_min:
                if remaining > 120:
                    log.info(f"  ⏳ {current_min} minutes until market open...")
                else:
                    log.info(f"  ⏳ {remaining:.0f} seconds until market open...")
                last_log_min = current_min

            # CRITICAL: Always sleep in short chunks (max 15s).
            # macOS hibernation freezes time.sleep() — a 300s sleep can
            # actually block for hours. 15s ensures we re-check the clock
            # almost immediately after waking from sleep.
            time.sleep(min(remaining, 15))
            continue

        # After 4 PM — market closed for today
        log.info("🔴 Market closed for today. Waiting for tomorrow's open...")
        # Short sleeps even here — laptop may sleep overnight and we
        # need to catch the next morning's open promptly
        time.sleep(min(900, 15))


# ─── Dynamic Strike Scanner ────────────────────────────────────

def scan_optimal_strangle(ticker, budget=150.0, otm_width=0.05):
    """
    Dynamically scan for the best strangle strikes at market open.

    Uses the live spot price to pick OTM call/put strikes, finds the
    nearest available expiry (~14 days out), and validates against budget.

    Returns dict with call_strike, put_strike, expiry, cost or None.
    """
    import yfinance as yf
    from config import STRANGLE_OTM_WIDTH

    otm_width = otm_width or STRANGLE_OTM_WIDTH

    try:
        tk = yf.Ticker(ticker)
        # Use 5d not 1d — at exactly 9:30 AM the first candle hasn't formed
        # yet, so period='1d' returns empty data and yfinance flags "possibly delisted"
        spot = tk.history(period="5d")['Close'].iloc[-1]
        log.info(f"📡 Live spot for {ticker}: ${spot:.2f}")

        exps = tk.options
        if not exps:
            log.error(f"No options chains for {ticker}")
            return None

        # Pick expiry closest to 14 days out
        from datetime import timedelta
        target_date = datetime.now() + timedelta(days=14)
        best_exp = min(exps, key=lambda x: abs(
            datetime.strptime(x, '%Y-%m-%d') - target_date))
        log.info(f"   Expiry selected: {best_exp}")

        chain = tk.option_chain(best_exp)

        # Calculate ideal OTM strikes
        ideal_call = round(spot * (1 + otm_width))
        ideal_put = round(spot * (1 - otm_width))

        # Ensure at least $1 away from ATM
        atm = round(spot)
        if ideal_call <= atm:
            ideal_call = atm + 1
        if ideal_put >= atm:
            ideal_put = atm - 1

        log.info(f"   Ideal strikes: ${ideal_call}C / ${ideal_put}P "
                 f"(ATM ${atm}, OTM width {otm_width:.0%})")

        # Find available strikes in chain (try exact, then ±1)
        c = chain.calls[chain.calls['strike'] == ideal_call]
        if c.empty:
            c = chain.calls[chain.calls['strike'] == ideal_call + 1]
            if not c.empty:
                ideal_call = ideal_call + 1
        if c.empty:
            c = chain.calls[chain.calls['strike'] == ideal_call - 1]
            if not c.empty:
                ideal_call = ideal_call - 1

        p = chain.puts[chain.puts['strike'] == ideal_put]
        if p.empty:
            p = chain.puts[chain.puts['strike'] == ideal_put - 1]
            if not p.empty:
                ideal_put = ideal_put - 1
        if p.empty:
            p = chain.puts[chain.puts['strike'] == ideal_put + 1]
            if not p.empty:
                ideal_put = ideal_put + 1

        if c.empty or p.empty:
            log.error(f"Could not find matching strikes for {ticker}")
            return None

        c = c.iloc[0]
        p = p.iloc[0]

        c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (c['bid'] + c['ask']) / 2
        p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (p['bid'] + p['ask']) / 2

        total_cost = (c_price + p_price) * 100 + 1.30

        if total_cost > budget:
            log.error(f"Strangle cost ${total_cost:.2f} exceeds budget ${budget:.2f}")
            return None

        log.info(f"   ✅ Optimal strangle: ${ideal_call}C (${c_price:.2f}) / "
                 f"${ideal_put}P (${p_price:.2f}) = ${total_cost:.2f}")

        return {
            'call_strike': int(ideal_call),
            'put_strike': int(ideal_put),
            'expiry': best_exp,
            'call_price': round(c_price, 2),
            'put_price': round(p_price, 2),
            'total_cost': round(total_cost, 2),
            'spot': round(spot, 2),
        }

    except Exception as e:
        log.error(f"Failed to scan strangle for {ticker}: {e}")
        return None


# ─── Fresh Quote Fetching ───────────────────────────────────────

def fetch_fresh_option_quotes(ticker, strike, expiry):
    """
    Fetch live call and put quotes from Webull (real-time) or yfinance (fallback).

    Returns dict with call_price, put_price, call_bid, put_bid, call_ask, put_ask,
    call_volume, put_volume. Returns None on failure.
    """
    # ─── Try Webull first (real-time, accurate bid/ask) ─────────
    try:
        from broker.webull_client import get_straddle_quotes
        if ensure_token():
            log.info("   📡 Trying Webull real-time quotes...")
            wb_quotes = get_straddle_quotes(ticker, strike, expiry)
            if wb_quotes:
                log.info(f"   ✅ Webull quotes received (source: live)")
                return {
                    'call_price': wb_quotes['call_price'],
                    'put_price': wb_quotes['put_price'],
                    'call_bid': wb_quotes['call_bid'],
                    'put_bid': wb_quotes['put_bid'],
                    'call_ask': wb_quotes['call_ask'],
                    'put_ask': wb_quotes['put_ask'],
                    'call_volume': wb_quotes['call_volume'],
                    'put_volume': wb_quotes['put_volume'],
                    'call_iv': 0.0,  # Webull snapshot may not include IV
                    'put_iv': 0.0,
                    'source': 'webull',
                }
            else:
                log.warning("   ⚠️  Webull quotes unavailable, falling back to yfinance")
    except Exception as e:
        log.warning(f"   ⚠️  Webull quote error: {e}, falling back to yfinance")

    # ─── Fallback: yfinance ─────────────────────────────────────
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)

        call_row = chain.calls[chain.calls['strike'] == strike]
        put_row = chain.puts[chain.puts['strike'] == strike]

        if call_row.empty or put_row.empty:
            log.error(f"No options found for {ticker} ${strike} exp {expiry}")
            return None

        c = call_row.iloc[0]
        p = put_row.iloc[0]

        # Use lastPrice if available, fall back to mid
        c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (c['bid'] + c['ask']) / 2
        p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (p['bid'] + p['ask']) / 2

        return {
            'call_price': round(c_price, 2),
            'put_price': round(p_price, 2),
            'call_bid': round(float(c['bid']), 2),
            'put_bid': round(float(p['bid']), 2),
            'call_ask': round(float(c['ask']), 2),
            'put_ask': round(float(p['ask']), 2),
            'call_volume': int(c['volume']) if not __import__('pandas').isna(c['volume']) else 0,
            'put_volume': int(p['volume']) if not __import__('pandas').isna(p['volume']) else 0,
            'call_iv': round(float(c['impliedVolatility']), 4),
            'put_iv': round(float(p['impliedVolatility']), 4),
            'source': 'yfinance',
        }
    except Exception as e:
        log.error(f"Failed to fetch quotes for {ticker}: {e}")
        return None


def fetch_fresh_strangle_quotes(ticker, call_strike, put_strike, expiry):
    """
    Fetch live call and put quotes for a STRANGLE (different strikes).

    Tries Webull real-time API first (accurate bid/ask at market open),
    falls back to yfinance if Webull fails.
    """
    # ─── Try Webull first (real-time, accurate bid/ask) ─────────
    try:
        from broker.webull_client import get_strangle_quotes
        if ensure_token():
            log.info("   📡 Trying Webull real-time quotes...")
            wb_quotes = get_strangle_quotes(ticker, call_strike, put_strike, expiry)
            if wb_quotes:
                log.info(f"   ✅ Webull quotes received (source: live)")
                return wb_quotes
            else:
                log.warning("   ⚠️  Webull quotes unavailable, falling back to yfinance")
    except Exception as e:
        log.warning(f"   ⚠️  Webull quote error: {e}, falling back to yfinance")

    # ─── Fallback: yfinance (delayed, may have stale bid/ask) ───
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)

        call_row = chain.calls[chain.calls['strike'] == call_strike]
        put_row = chain.puts[chain.puts['strike'] == put_strike]

        if call_row.empty:
            log.error(f"No call found for {ticker} ${call_strike} exp {expiry}")
            return None
        if put_row.empty:
            log.error(f"No put found for {ticker} ${put_strike} exp {expiry}")
            return None

        c = call_row.iloc[0]
        p = put_row.iloc[0]

        c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (c['bid'] + c['ask']) / 2
        p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (p['bid'] + p['ask']) / 2

        return {
            'call_strike': call_strike,
            'put_strike': put_strike,
            'call_price': round(c_price, 2),
            'put_price': round(p_price, 2),
            'call_bid': round(float(c['bid']), 2),
            'put_bid': round(float(p['bid']), 2),
            'call_ask': round(float(c['ask']), 2),
            'put_ask': round(float(p['ask']), 2),
            'call_volume': int(c['volume']) if not __import__('pandas').isna(c['volume']) else 0,
            'put_volume': int(p['volume']) if not __import__('pandas').isna(p['volume']) else 0,
            'source': 'yfinance',
        }
    except Exception as e:
        log.error(f"Failed to fetch strangle quotes for {ticker}: {e}")
        return None


# ─── Active Trade Persistence ──────────────────────────────────

def _save_active_trade(trade_info):
    """Save active trade details to disk for recovery after restart."""
    os.makedirs(os.path.dirname(ACTIVE_TRADE_FILE), exist_ok=True)
    with open(ACTIVE_TRADE_FILE, "w") as f:
        json.dump(trade_info, f, indent=2)
    log.info(f"💾 Active trade saved to {ACTIVE_TRADE_FILE}")


def _load_active_trade():
    """Load active trade from disk (for resuming after restart)."""
    if os.path.exists(ACTIVE_TRADE_FILE):
        try:
            with open(ACTIVE_TRADE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _clear_active_trade():
    """Remove the active trade file after position is closed."""
    if os.path.exists(ACTIVE_TRADE_FILE):
        os.remove(ACTIVE_TRADE_FILE)
        log.info("🗑️  Active trade file cleared.")


# ─── Ensure Token ──────────────────────────────────────────────

def ensure_token():
    """Load access token from .env if not set."""
    if not wb.ACCESS_TOKEN:
        from dotenv import load_dotenv
        load_dotenv(wb._ENV_PATH)
        wb.ACCESS_TOKEN = os.getenv("WEBULL_ACCESS_TOKEN", "")
    if not wb.ACCESS_TOKEN:
        log.error("❌ No WEBULL_ACCESS_TOKEN found in .env")
        return False
    return True


def get_account_id():
    """Get the margin account ID (or first available)."""
    accounts = get_accounts()
    if not accounts:
        return None
    for a in accounts:
        if a.get("account_type") == "MARGIN" and a.get("account_class") == "INDIVIDUAL_MARGIN":
            return a["account_id"]
    return accounts[0]["account_id"]


# ─── Phase 1: Open the Straddle ────────────────────────────────

def open_straddle(ticker, strike, expiry, budget=150.0, dry_run=False):
    """
    Open a long straddle at market open using fresh quotes.

    Returns dict with trade details or None on failure.
    """
    log.info(f"\n{'='*60}")
    log.info(f"📈 OPENING STRADDLE: {ticker} ${strike} exp {expiry}")
    log.info(f"   Budget: ${budget:.2f} | Mode: {'DRY RUN' if dry_run else '🔴 LIVE'}")
    log.info(f"{'='*60}")

    # Fetch fresh quotes
    log.info("📡 Fetching fresh option quotes...")
    quotes = fetch_fresh_option_quotes(ticker, float(strike), expiry)
    if not quotes:
        log.error("❌ Could not fetch option quotes. Aborting.")
        return None

    call_price = quotes['call_price']
    put_price = quotes['put_price']
    call_bid, call_ask = quotes['call_bid'], quotes['call_ask']
    put_bid, put_ask = quotes['put_bid'], quotes['put_ask']
    source = quotes.get('source', 'unknown')

    log.info(f"   Call: ${call_price:.2f} (bid ${call_bid:.2f} / ask ${call_ask:.2f}) [{source}]")
    log.info(f"   Put:  ${put_price:.2f} (bid ${put_bid:.2f} / ask ${put_ask:.2f}) [{source}]")

    contracts = max(1, int((budget - 1.30) / ((call_price + put_price) * 100)))
    total_cost = (call_price + put_price) * 100 * contracts + 1.30

    log.info(f"   ✅ Both legs have active markets.")
    log.info(f"   Straddle: ${(call_price + put_price):.2f}/share × {contracts} contract(s)")
    log.info(f"   Total cost: ${total_cost:.2f}")

    if total_cost > budget:
        log.error(f"❌ Cost ${total_cost:.2f} exceeds budget ${budget:.2f}. Aborting.")
        return None

    if dry_run:
        log.info(f"🧪 DRY RUN — would buy {contracts}x {ticker} ${strike} straddle for ${total_cost:.2f}")
        trade_info = {
            "ticker": ticker,
            "strike": float(strike),
            "expiry": expiry,
            "contracts": contracts,
            "call_price": call_price,
            "put_price": put_price,
            "total_cost": total_cost,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "mode": "dry_run",
        }
        _save_active_trade(trade_info)
        return trade_info

    # Live execution
    if not ensure_token():
        return None

    account_id = get_account_id()
    if not account_id:
        log.error("❌ Could not retrieve account. Aborting.")
        return None

    log.info(f"💰 PLACING LIVE STRADDLE ORDER (COMBO)...")
    from broker.webull_client import place_combo_straddle

    result = place_combo_straddle(
        account_id=account_id,
        symbol=ticker,
        strike=f"{float(strike):.2f}",
        expiry=expiry,
        side="BUY",
        quantity=contracts,
        call_limit=f"{call_price:.2f}",
        put_limit=f"{put_price:.2f}",
    )

    if not result["success"]:
        log.error(f"❌ Combo order failed: {result.get('error', 'unknown')}")
        return None

    if result["success"]:
        log.info(f"✅ STRADDLE OPENED! {ticker} ${strike} × {contracts} for ${total_cost:.2f}")
        trade_info = {
            "ticker": ticker,
            "strike": float(strike),
            "expiry": expiry,
            "contracts": contracts,
            "call_price": call_price,
            "put_price": put_price,
            "total_cost": total_cost,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "mode": "live",
            "call_order": str(result.get("call_order", {})),
            "put_order": str(result.get("put_order", {})),
        }
        _save_active_trade(trade_info)
        return trade_info
    else:
        log.error(f"❌ ORDER FAILED: {result}")
        return None


def open_strangle(ticker, call_strike, put_strike, expiry, budget=150.0, dry_run=False):
    """
    Open a long strangle at market open — OTM call + OTM put at different strikes.
    """
    log.info(f"\n{'='*60}")
    log.info(f"📈 OPENING STRANGLE: {ticker} ${call_strike}C / ${put_strike}P exp {expiry}")
    log.info(f"   Budget: ${budget:.2f} | Mode: {'DRY RUN' if dry_run else '🔴 LIVE'}")
    log.info(f"{'='*60}")

    # ─── Quote fetch with previous-day liquidity fallback ──────────
    log.info("📡 Fetching strangle quotes...")
    quotes = fetch_fresh_strangle_quotes(ticker, float(call_strike), float(put_strike), expiry)
    if not quotes:
        log.error("❌ Could not fetch strangle quotes. Aborting.")
        return None

    call_price = quotes['call_price']
    put_price = quotes['put_price']
    call_bid, call_ask = quotes['call_bid'], quotes['call_ask']
    put_bid, put_ask = quotes['put_bid'], quotes['put_ask']
    source = quotes.get('source', 'unknown')

    log.info(f"   Call ${call_strike}: ${call_price:.2f} (bid ${call_bid:.2f} / ask ${call_ask:.2f}) [{source}]")
    log.info(f"   Put  ${put_strike}: ${put_price:.2f} (bid ${put_bid:.2f} / ask ${put_ask:.2f}) [{source}]")

    contracts = max(1, int((budget - 1.30) / ((call_price + put_price) * 100)))
    total_cost = (call_price + put_price) * 100 * contracts + 1.30

    log.info(f"   Strangle: ${(call_price + put_price):.2f}/share × {contracts} contract(s)")
    log.info(f"   Total cost: ${total_cost:.2f}")

    if total_cost > budget:
        log.error(f"❌ Cost ${total_cost:.2f} exceeds budget ${budget:.2f}. Aborting.")
        return None

    if dry_run:
        log.info(f"🧪 DRY RUN — would buy {contracts}x {ticker} ${call_strike}C/${put_strike}P strangle for ${total_cost:.2f}")
        trade_info = {
            "ticker": ticker,
            "strategy": "strangle",
            "call_strike": float(call_strike),
            "put_strike": float(put_strike),
            "strike": float(call_strike),  # backward compat for monitor
            "expiry": expiry,
            "contracts": contracts,
            "call_price": call_price,
            "put_price": put_price,
            "total_cost": total_cost,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "mode": "dry_run",
        }
        _save_active_trade(trade_info)
        return trade_info

    # Live execution
    if not ensure_token():
        return None

    account_id = get_account_id()
    if not account_id:
        log.error("❌ Could not retrieve account. Aborting.")
        return None

    log.info(f"💰 PLACING LIVE STRANGLE ORDER (COMBO)...")
    from broker.webull_client import place_combo_strangle

    combo_result = place_combo_strangle(
        account_id, ticker,
        call_strike=f"{float(call_strike):.2f}",
        put_strike=f"{float(put_strike):.2f}",
        expiry=expiry, side="BUY", quantity=contracts,
        call_limit=f"{call_price:.2f}",
        put_limit=f"{put_price:.2f}",
    )

    if not combo_result["success"]:
        log.error(f"❌ Combo order failed: {combo_result.get('error', 'unknown')}")
        return None

    log.info(f"   ✅ COMBO order accepted (both legs atomic).")

    log.info(f"✅ STRANGLE OPENED! {ticker} ${call_strike}C/${put_strike}P × {contracts} for ${total_cost:.2f}")
    trade_info = {
        "ticker": ticker,
        "strategy": "strangle",
        "call_strike": float(call_strike),
        "put_strike": float(put_strike),
        "strike": float(call_strike),  # backward compat for monitor
        "expiry": expiry,
        "contracts": contracts,
        "call_price": call_price,
        "put_price": put_price,
        "total_cost": total_cost,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "call_order": str(combo_result),
        "put_order": str(combo_result),
    }
    _save_active_trade(trade_info)
    return trade_info


# ─── Phase 2: Monitor & Auto-Close ─────────────────────────────

def refresh_and_close(account_id, straddle_data, ticker, strike, expiry):
    """
    Close a straddle using FRESH quotes (not stale last_price).

    This fixes the bug where close_straddle used stale last_price values,
    which caused Webull rejections or bad fills.
    """
    log.info(f"🔄 Fetching fresh quotes for sell order...")
    quotes = fetch_fresh_option_quotes(ticker, float(strike), expiry)

    if quotes:
        # Use bid price for sells (what we'd actually get filled at)
        # Fall back to lastPrice, then to a penny above zero
        call_sell = quotes['call_bid'] if quotes['call_bid'] > 0.01 else quotes['call_price']
        put_sell = quotes['put_bid'] if quotes['put_bid'] > 0.01 else quotes['put_price']

        log.info(f"   Sell call @ ${call_sell:.2f} (bid), put @ ${put_sell:.2f} (bid)")

        # Update straddle_data with fresh prices
        if straddle_data.get("call"):
            straddle_data["call"]["last_price"] = call_sell
        if straddle_data.get("put"):
            straddle_data["put"]["last_price"] = put_sell
    else:
        log.warning("⚠️  Could not refresh quotes. Using position's last_price as fallback.")

    return close_straddle(account_id, straddle_data)


def monitor_and_close(ticker, strike, expiry, dry_run=False, target_pct=None, no_stop_loss=False):
    """
    Monitor an open straddle position and auto-close when thresholds are hit.

    Runs until the position is closed or the script is stopped.
    """
    tp = target_pct if target_pct is not None else TARGET_RETURN_PCT
    sl_label = "DISABLED" if no_stop_loss else f"{STOP_LOSS_PCT}%"
    log.info(f"\n{'='*60}")
    log.info(f"👁️  AUTO-CLOSE MONITOR STARTED")
    log.info(f"   Watching: {ticker} ${strike} exp {expiry}")
    log.info(f"   Take-profit: +{tp}% | Stop-loss: {sl_label}")
    log.info(f"{'='*60}")

    consecutive_errors = 0
    max_errors = 50

    while True:
        try:
            market_open = is_market_hours()
            rush = is_market_open_rush()

            # Determine poll interval
            if rush:
                interval = POLL_INTERVAL_RUSH
            elif market_open:
                interval = POLL_INTERVAL_MARKET
            else:
                interval = POLL_INTERVAL_OFF

            status = "🟢 MARKET OPEN" if market_open else "🔴 MARKET CLOSED"
            if rush:
                status += " (OPEN RUSH)"

            log.info(f"\n--- Monitor @ {datetime.now().strftime('%H:%M:%S')} | {status} ---")

            if not ensure_token():
                log.error("No token. Retrying in 60s...")
                time.sleep(60)
                continue

            # Fetch live positions
            pos_data = None
            for attempt in range(3):
                try:
                    pos_data = fetch_live_positions()
                    break
                except Exception as e:
                    log.warning(f"Position fetch attempt {attempt+1}/3 failed: {e}")
                    time.sleep(5)

            if not pos_data or not pos_data.get("straddles"):
                log.info("  No open straddles found.")
                # Check if our trade was supposed to be open
                active = _load_active_trade()
                if active and active.get("mode") == "live":
                    log.warning("  ⚠️  Active trade file exists but no positions found. "
                                "Orders may still be filling. Waiting...")
                time.sleep(interval)
                consecutive_errors = 0
                continue

            # Find our specific straddle
            target_key = f"{ticker}_{float(strike):.2f}_{expiry}"
            # Also try integer strike format
            target_key_int = f"{ticker}_{int(float(strike))}_{expiry}"
            # And with .0 format
            target_key_dot = f"{ticker}_{float(strike)}_{expiry}"

            straddle = None
            matched_key = None
            for key, s in pos_data["straddles"].items():
                if s["symbol"] == ticker and s["expiry"] == expiry:
                    # Match on either strike (strangles key by put strike)
                    straddle = s
                    matched_key = key
                    break

            if not straddle:
                log.info(f"  Position for {ticker} exp {expiry} not found in straddles.")
                log.info(f"  Available keys: {list(pos_data['straddles'].keys())}")
                time.sleep(interval)
                consecutive_errors = 0
                continue

            # Report P&L
            pnl_pct = straddle["pnl_pct"]
            total_pnl = straddle["total_pnl"]
            log.info(
                f"  {ticker} ${strike} straddle: "
                f"P&L ${total_pnl:+.2f} ({pnl_pct:+.2f}%) | "
                f"TP: +{tp}% | SL: {STOP_LOSS_PCT}%"
            )

            # Check thresholds
            exit_reason = None
            if pnl_pct >= tp:
                exit_reason = "take_profit"
                log.info(f"🎯 TAKE-PROFIT HIT! {ticker} at +{pnl_pct:.2f}%")
            elif not no_stop_loss and pnl_pct <= STOP_LOSS_PCT:
                exit_reason = "stop_loss"
                log.warning(f"🛑 STOP-LOSS HIT! {ticker} at {pnl_pct:.2f}%")

            if exit_reason is None:
                consecutive_errors = 0
                time.sleep(interval)
                continue

            # Threshold hit — close the position
            if dry_run:
                log.info(f"🧪 DRY RUN — would close {ticker} straddle ({exit_reason}) "
                         f"at P&L {pnl_pct:+.2f}%")
                time.sleep(interval)
                continue

            if not market_open:
                log.info(
                    f"⏳ {ticker} triggered {exit_reason} ({pnl_pct:+.2f}%) "
                    f"but market is CLOSED. Will re-check when market opens."
                )
                time.sleep(interval)
                continue

            # Market is open — close with fresh quotes
            account_id = get_account_id()
            if not account_id:
                log.error("Cannot get account ID for close. Retrying...")
                time.sleep(30)
                continue

            log.info(f"📤 AUTO-CLOSING {ticker} ({exit_reason}) with FRESH quotes...")
            close_result = refresh_and_close(
                account_id, straddle, ticker, strike, expiry
            )

            if close_result["success"]:
                log.info(f"✅ CLOSED! {ticker} P&L: ${total_pnl:+.2f} ({pnl_pct:+.2f}%)")
                _log_closed_trade(matched_key, straddle, close_result, pnl_pct,
                                  exit_reason=exit_reason)
                _clear_active_trade()
                return close_result
            else:
                log.error(f"❌ Close failed: {close_result}. Will retry next poll.")

            consecutive_errors = 0
            time.sleep(interval)

        except KeyboardInterrupt:
            log.info("\n⏹  Monitor stopped by user.")
            return None
        except Exception as e:
            consecutive_errors += 1
            log.error(f"Error (#{consecutive_errors}): {e}")
            if consecutive_errors >= max_errors:
                log.critical(f"Too many errors ({max_errors}). Stopping monitor.")
                return None
            time.sleep(60)


# ─── Pending Close Handler ──────────────────────────────────────


# ─── Main Entry Point ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Straddle Trader — opens at market open, auto-closes at +12%",
    )
    parser.add_argument("--ticker", "-t", default="QS", help="Ticker symbol (default: QS)")
    parser.add_argument("--strike", "-k", default="7", help="Strike price for straddle (default: 7)")
    parser.add_argument("--call-strike", default=None, help="Call strike for strangle (enables strangle mode)")
    parser.add_argument("--put-strike", default=None, help="Put strike for strangle (enables strangle mode)")
    parser.add_argument("--expiry", "-e", default="2026-06-12", help="Expiry date (default: 2026-06-12)")
    parser.add_argument("--budget", "-b", type=float, default=150.0, help="Max budget (default: $150)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without placing real orders")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Skip opening, just monitor existing position for auto-close")
    parser.add_argument("--auto-scan", action="store_true",
                        help="Dynamically pick optimal OTM strikes at market open (strangle only)")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated ranked tickers from night-before scan (e.g. SKLZ,QS,XPEV). "
                             "Skips full scanner, uses these tickers directly.")
    parser.add_argument("--target-pct", type=float, default=None,
                        help="Override take-profit %% (default: 12%%). Example: --target-pct 56.25")
    parser.add_argument("--no-stop-loss", action="store_true",
                        help="Disable stop-loss auto-close (only close on take-profit)")
    parser.add_argument("--trade-file", type=str, default=None,
                        help="Custom path for active trade state file (enables multiple daemons)")

    args = parser.parse_args()

    # Override global ACTIVE_TRADE_FILE if --trade-file is specified
    if args.trade_file:
        global ACTIVE_TRADE_FILE
        ACTIVE_TRADE_FILE = args.trade_file
        log.info(f"   Trade file: {ACTIVE_TRADE_FILE}")

    # Detect strangle mode
    is_strangle = args.call_strike is not None and args.put_strike is not None
    strategy_name = "STRANGLE" if is_strangle else "STRADDLE"

    log.info("=" * 60)
    log.info(f"🤖 AUTONOMOUS {strategy_name} TRADER")
    log.info(f"   Ticker:  {args.ticker}")
    if is_strangle:
        log.info(f"   Call:    ${args.call_strike}")
        log.info(f"   Put:     ${args.put_strike}")
    else:
        log.info(f"   Strike:  ${args.strike}")
    log.info(f"   Expiry:  {args.expiry}")
    log.info(f"   Budget:  ${args.budget:.2f}")
    log.info(f"   Mode:    {'🧪 DRY RUN' if args.dry_run else '🔴 LIVE'}")
    log.info(f"   TP/SL:   +{TARGET_RETURN_PCT}% / {STOP_LOSS_PCT}%")
    log.info("=" * 60)

    # Check for existing active trade (resume after restart)
    active = _load_active_trade()
    already_open = False

    if active and not args.monitor_only:
        log.info(f"📋 Found existing active trade: {active['ticker']} ${active['strike']} "
                 f"opened {active['opened_at'][:16]}")
        log.info("  Skipping open phase, going straight to monitor.")
        already_open = True
        args.ticker = active['ticker']
        args.strike = str(active['strike'])
        args.expiry = active['expiry']

    if args.monitor_only:
        log.info("📋 Monitor-only mode. Skipping open phase.")
        already_open = True

    if not already_open:
        # ─── Phase 1: Wait for market open and place the trade ───
        log.info("\n" + "─" * 40)
        log.info("PHASE 1: WAITING FOR MARKET OPEN")
        log.info("─" * 40)

        wait_for_market_open()

        # Open position at 9:30 AM ET (market open)
        log.info("⏳ Waiting until 9:30 AM ET...")
        import pytz
        et = pytz.timezone("US/Eastern")
        target_930 = datetime.now(et).replace(hour=9, minute=30, second=0, microsecond=0)
        while datetime.now(et) < target_930:
            remaining = (target_930 - datetime.now(et)).total_seconds()
            if remaining > 60:
                log.info(f"   ⏳ {int(remaining)}s until 9:30 AM ET...")
            time.sleep(min(15, remaining))
        log.info("🟢 9:30 AM — market open.")

        # Auto-scan: iterate through ranked opportunities
        if args.auto_scan:
            # Use pre-selected targets if provided, otherwise run full scanner
            if args.targets:
                target_tickers = [t.strip() for t in args.targets.split(',')]
                MAX_COMBO_ATTEMPTS = len(target_tickers)
                log.info(f"\n🔍 AUTO-SCAN: Using pre-selected targets (up to {MAX_COMBO_ATTEMPTS} attempts)...")
                for i, t in enumerate(target_tickers, 1):
                    log.info(f"      #{i}: {t}")
                # Pre-selected targets use strangle mode
                scan_mode = "strangle"
            else:
                MAX_COMBO_ATTEMPTS = 3
                log.info(f"\n🔍 AUTO-SCAN: Running v4 straddle + strangle scanners...")

                from signals.strangle_scanner import scan_strangle_opportunities
                from signals.scanner import scan_for_opportunities

                # Run both scanners
                strangle_results, strangle_rej = scan_strangle_opportunities(budget=args.budget)
                straddle_results = scan_for_opportunities(budget=args.budget)

                # Normalize and merge results with strategy tag
                all_candidates = []
                for r in (strangle_results or []):
                    all_candidates.append({
                        'ticker': r['ticker'], 'score': r['composite_score'],
                        'strategy': 'strangle', 'data': r,
                        'desc': f"{r['ticker']} ${r['call_strike']}C/${r['put_strike']}P "
                                f"exp {r['expiry']} (score {r['composite_score']:.3f})"
                    })
                for r in (straddle_results or []):
                    score = r.get('composite_score', r.get('score', r.get('garch_spread', 0)))
                    all_candidates.append({
                        'ticker': r['ticker'], 'score': score,
                        'strategy': 'straddle', 'data': r,
                        'desc': f"{r['ticker']} ${r.get('strike', '?')} straddle "
                                f"exp {r.get('expiry', '?')} (score {score:.3f})"
                    })

                all_candidates.sort(key=lambda x: x['score'], reverse=True)

                if not all_candidates:
                    log.error("❌ Both scanners found no valid opportunities. Exiting.")
                    sys.exit(1)

                log.info(f"   📊 Combined: {len(all_candidates)} opportunities "
                         f"(trying top {min(MAX_COMBO_ATTEMPTS, len(all_candidates))}):")
                for i, c in enumerate(all_candidates[:MAX_COMBO_ATTEMPTS]):
                    log.info(f"      #{i+1}: [{c['strategy'].upper()}] {c['desc']}")

                target_tickers = [c['ticker'] for c in all_candidates[:MAX_COMBO_ATTEMPTS]]
                # Track which strategy to use per ticker
                scan_mode = {c['ticker']: c for c in all_candidates[:MAX_COMBO_ATTEMPTS]}

            trade = None
            for attempt, ticker in enumerate(target_tickers, 1):
                log.info(f"\n{'─'*40}")
                log.info(f"   🎯 ATTEMPT {attempt}/{MAX_COMBO_ATTEMPTS}: {ticker}")
                log.info(f"{'─'*40}")

                # Determine strategy for this ticker
                if isinstance(scan_mode, str):
                    strategy = scan_mode
                else:
                    strategy = scan_mode[ticker]['strategy']

                if strategy == "strangle":
                    # Fresh price scan for this ticker
                    scan = scan_optimal_strangle(ticker, budget=args.budget)
                    if not scan:
                        log.warning(f"   ⚠️  Fresh scan failed for {ticker}. Skipping to next.")
                        continue

                    log.info(f"   📍 Live strikes: {ticker} "
                             f"${scan['call_strike']}C/${scan['put_strike']}P "
                             f"exp {scan['expiry']} (spot ${scan['spot']})")

                    args.ticker = ticker
                    args.call_strike = str(scan['call_strike'])
                    args.put_strike = str(scan['put_strike'])
                    args.expiry = scan['expiry']
                    is_strangle = True
                    strategy_name = "STRANGLE"

                    trade = open_strangle(
                        ticker=args.ticker,
                        call_strike=args.call_strike,
                        put_strike=args.put_strike,
                        expiry=args.expiry,
                        budget=args.budget,
                        dry_run=args.dry_run,
                    )
                else:
                    # Straddle mode — use ATM strike
                    import yfinance as yf_scan
                    tk = yf_scan.Ticker(ticker)
                    spot = tk.history(period='1d')['Close'].iloc[-1]
                    strike = round(spot)
                    # Find best expiry ~14 days out
                    from datetime import timedelta as td
                    target_exp = datetime.now() + td(days=14)
                    exps = tk.options
                    best_exp = min(exps, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_exp))

                    log.info(f"   📍 Live straddle: {ticker} ${strike} exp {best_exp} (spot ${spot:.2f})")

                    args.ticker = ticker
                    args.strike = str(strike)
                    args.expiry = best_exp
                    is_strangle = False
                    strategy_name = "STRADDLE"

                    trade = open_straddle(
                        ticker=args.ticker,
                        strike=args.strike,
                        expiry=args.expiry,
                        budget=args.budget,
                        dry_run=args.dry_run,
                    )

                if trade:
                    log.info(f"   ✅ Attempt {attempt} succeeded! {ticker} {strategy_name.lower()} opened.")
                    break
                else:
                    log.warning(f"   ❌ Attempt {attempt} failed for {ticker}. "
                                f"{'Trying next...' if attempt < MAX_COMBO_ATTEMPTS else 'No more attempts.'}")

            if not trade:
                log.error(f"❌ All {MAX_COMBO_ATTEMPTS} attempts failed. No trade opened. Exiting.")
                sys.exit(1)

        else:
            # Non-auto-scan: use CLI-specified ticker/strikes
            if is_strangle:
                trade = open_strangle(
                    ticker=args.ticker,
                    call_strike=args.call_strike,
                    put_strike=args.put_strike,
                    expiry=args.expiry,
                    budget=args.budget,
                    dry_run=args.dry_run,
                )
            else:
                trade = open_straddle(
                    ticker=args.ticker,
                    strike=args.strike,
                    expiry=args.expiry,
                    budget=args.budget,
                    dry_run=args.dry_run,
                )

            if not trade:
                log.error(f"❌ Failed to open {strategy_name.lower()}. Exiting.")
                sys.exit(1)

        log.info(f"✅ Phase 1 complete. {strategy_name} is open.")

        # Wait for the order to fill (give it 60 seconds)
        if not args.dry_run:
            log.info("⏳ Waiting 60 seconds for orders to fill...")
            for _ in range(12):
                time.sleep(5)

    # ─── Phase 2: Monitor and auto-close ─────────────────────────
    log.info("\n" + "─" * 40)
    log.info("PHASE 2: AUTO-CLOSE MONITOR")
    log.info("─" * 40)

    # Use the actual strike from the trade (auto-scan may have changed it)
    # Priority: active_trade.json > auto-scan args > CLI default
    active = _load_active_trade()
    if active:
        monitor_strike = str(active.get('call_strike', active.get('strike', args.strike)))
        monitor_expiry = active.get('expiry', args.expiry)
        log.info(f"   📋 Using strike from active trade: ${monitor_strike} exp {monitor_expiry}")
    else:
        monitor_strike = args.strike
        monitor_expiry = args.expiry

    monitor_and_close(
        ticker=args.ticker,
        strike=monitor_strike,
        expiry=monitor_expiry,
        dry_run=args.dry_run,
        target_pct=args.target_pct,
        no_stop_loss=args.no_stop_loss,
    )

    log.info("\n🏁 Auto-trader daemon finished.")


if __name__ == "__main__":
    main()

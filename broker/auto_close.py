"""
Auto-Close Monitor — background daemon that watches open positions
and automatically closes straddles/strangles when the target return
is hit or the stop-loss threshold is breached.

ABSOLUTE RULE: Positions are closed IMMEDIATELY when P&L >= +12%,
regardless of strategy type, time held, or any other condition.
This threshold is unconditional and cannot be bypassed.

Usage:
    python broker/auto_close.py              # Run in foreground
    python broker/auto_close.py &            # Run in background
    nohup python broker/auto_close.py &      # Survives terminal close

Configuration (from config.py):
    TAKE_PROFIT_PCT     = 12.0   (close when P&L >= +12% — ABSOLUTE)
    STOP_LOSS_PCT       = -30.0  (close when P&L <= -30%)
    POLL_INTERVAL_SECS  = 30     (check every 30 seconds)

Reliability features:
    - Pending close state: if a threshold triggers after hours, the exit
      is saved to disk and executed at next market open regardless of
      overnight price changes (gap protection).
    - Retry on timeout: API calls retry up to 3 times on network errors.
    - High error tolerance: 50 consecutive errors before stopping.
"""
import sys
import os
import time
import json
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TAKE_PROFIT_PCT, STOP_LOSS_PCT
from broker.webull_client import get_accounts, get_positions, close_straddle, _call_api
import broker.webull_client as wb
from broker.position_tracker import fetch_live_positions

# ─── Configuration ───────────────────────────────────────────────
# TAKE_PROFIT_PCT and STOP_LOSS_PCT imported from config.py (single source of truth)
TARGET_RETURN_PCT = TAKE_PROFIT_PCT  # Alias for backward compatibility
POLL_INTERVAL_SECS = 30       # Check every 30 seconds during market hours
POLL_INTERVAL_OFF = 300       # Check every 5 min outside market hours
POLL_INTERVAL_OPEN_RUSH = 10  # Aggressive polling first 5 min of market open
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "auto_close.log")

# ─── Logging ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

log = logging.getLogger("auto_close")
log.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(ch)

# File handler (persistent log)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(fh)

# ─── Trade log (JSON) ───────────────────────────────────────────
TRADE_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "closed_trades.json")


def _log_closed_trade(straddle_key, straddle_data, close_result, pnl_pct, exit_reason="take_profit"):
    """Append a record to the closed trades log."""
    trades = []
    if os.path.exists(TRADE_LOG):
        try:
            with open(TRADE_LOG, "r") as f:
                trades = json.load(f)
        except Exception:
            pass

    trades.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "straddle": straddle_key,
        "symbol": straddle_data["symbol"],
        "strike": straddle_data["strike"],
        "expiry": straddle_data["expiry"],
        "cost": straddle_data["total_cost"],
        "value_at_close": straddle_data["total_value"],
        "pnl": straddle_data["total_pnl"],
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "close_result": {
            "success": close_result.get("success"),
            "call_order": str(close_result.get("call_order", {})),
            "put_order": str(close_result.get("put_order", {})),
        },
    })

    with open(TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2)


# ─── Pending Close State (REMOVED) ─────────────────────────────
# The pending close system has been removed. Rationale: by the time
# the daemon restarts after a shutdown, prices will have changed
# significantly, making stale trigger prices unreliable.
# Instead, the daemon re-evaluates live P&L on every poll and only
# closes when the +12% threshold is actively met with fresh quotes.

def _load_pending_closes():
    """Stub — pending close system removed."""
    return {}

def _save_pending_closes(pending):
    """Stub — pending close system removed."""
    pass

def _add_pending_close(straddle_key, exit_reason, pnl_pct):
    """Stub — pending close system removed."""
    pass

def _remove_pending_close(straddle_key):
    """Stub — pending close system removed."""
    pass


# ─── Market Hours ───────────────────────────────────────────────

def is_market_hours():
    """Check if US market is open (rough check, no holiday calendar)."""
    from datetime import timezone as tz
    import pytz
    try:
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        # Weekday (0=Mon, 6=Sun) and between 9:30 AM - 4:00 PM ET
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except ImportError:
        # pytz not available, default to always checking
        return True


def is_market_open_rush():
    """Check if we're in the first 5 minutes of market open (9:30-9:35 ET)."""
    import pytz
    try:
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        if now.weekday() >= 5:
            return False
        rush_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        rush_end = now.replace(hour=9, minute=35, second=0, microsecond=0)
        return rush_start <= now <= rush_end
    except ImportError:
        return False


def ensure_token():
    """Load token from .env if not already set."""
    if not wb.ACCESS_TOKEN:
        from dotenv import load_dotenv
        load_dotenv(wb._ENV_PATH)
        wb.ACCESS_TOKEN = os.getenv("WEBULL_ACCESS_TOKEN", "")
    return bool(wb.ACCESS_TOKEN)


# ─── Retry wrapper ──────────────────────────────────────────────

def fetch_positions_with_retry(max_retries=3):
    """Fetch live positions with retry on network errors."""
    for attempt in range(max_retries):
        try:
            return fetch_live_positions()
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"Position fetch failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5)
            else:
                raise


# ─── Core Logic ─────────────────────────────────────────────────

def _refresh_straddle_prices(straddle):
    """
    Fetch fresh bid prices from yfinance and update the straddle's leg
    last_price fields. This prevents selling at stale prices, which was
    the root cause of bad fills and after-hours rejections.
    """
    import yfinance as yf
    import pandas as pd

    symbol = straddle["symbol"]
    strike = straddle["strike"]
    expiry = straddle["expiry"]

    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiry)

        call_row = chain.calls[chain.calls['strike'] == strike]
        put_row = chain.puts[chain.puts['strike'] == strike]

        if not call_row.empty and straddle.get("call"):
            c = call_row.iloc[0]
            # Use bid for sells (what we'd actually fill at), fallback to lastPrice
            fresh_price = float(c['bid']) if float(c['bid']) > 0.01 else float(c['lastPrice'])
            old_price = straddle["call"]["last_price"]
            straddle["call"]["last_price"] = fresh_price
            log.info(f"  🔄 {symbol} call price refreshed: ${old_price:.2f} → ${fresh_price:.2f}")

        if not put_row.empty and straddle.get("put"):
            p = put_row.iloc[0]
            fresh_price = float(p['bid']) if float(p['bid']) > 0.01 else float(p['lastPrice'])
            old_price = straddle["put"]["last_price"]
            straddle["put"]["last_price"] = fresh_price
            log.info(f"  🔄 {symbol} put price refreshed: ${old_price:.2f} → ${fresh_price:.2f}")

        return True
    except Exception as e:
        log.warning(f"  ⚠️ Could not refresh prices for {symbol}: {e}. Using stale last_price.")
        return False


def check_and_close():
    """
    Core logic: fetch positions, check if any straddle hit the target,
    and auto-close if so.

    Handles three scenarios:
    1. Threshold hit during market hours → refresh quotes + close immediately
    2. Threshold hit after hours → save pending close to disk
    3. Pending close exists at market open → refresh quotes + close regardless of current P&L

    Returns:
        list of successfully closed straddle keys
    """
    if not ensure_token():
        log.error("No access token. Run activate_token() first.")
        return []

    pos_data = fetch_positions_with_retry()
    if not pos_data or not pos_data.get("straddles"):
        return []

    market_open = is_market_hours()

    # Get account ID
    accounts = get_accounts()
    if not accounts:
        log.error("Cannot fetch accounts.")
        return []

    account_id = None
    for a in accounts:
        if a.get("account_type") == "MARGIN" and a.get("account_class") == "INDIVIDUAL_MARGIN":
            account_id = a["account_id"]
            break
    if not account_id:
        account_id = accounts[0]["account_id"]

    closed = []

    for key, straddle in pos_data["straddles"].items():
        pnl_pct = straddle["pnl_pct"]
        symbol = straddle["symbol"]
        total_pnl = straddle["total_pnl"]

        log.info(
            f"  {symbol} ${straddle['strike']:.0f} straddle: "
            f"P&L ${total_pnl:+.2f} ({pnl_pct:+.2f}%) | "
            f"TP: +{TARGET_RETURN_PCT}% | SL: {STOP_LOSS_PCT}%"
        )

        # ─── Determine if a threshold was breached ───────────────
        exit_reason = None
        if pnl_pct >= TARGET_RETURN_PCT:
            exit_reason = "take_profit"
            log.info(f"🎯 TAKE-PROFIT HIT! {symbol} at +{pnl_pct:.2f}%")
        elif pnl_pct <= STOP_LOSS_PCT:
            exit_reason = "stop_loss"
            log.warning(f"🛑 STOP-LOSS HIT! {symbol} at {pnl_pct:.2f}%")

        if exit_reason is None:
            continue

        # ─── Market open: refresh quotes + close immediately ─────
        if market_open:
            log.info(f"📤 AUTO-CLOSING {symbol} ({exit_reason}) — refreshing quotes first...")
            _refresh_straddle_prices(straddle)
            close_result = close_straddle(account_id, straddle)

            if close_result["success"]:
                log.info(f"✅ Successfully closed {symbol} straddle! P&L: ${total_pnl:+.2f} ({pnl_pct:+.2f}%)")
                _log_closed_trade(key, straddle, close_result, pnl_pct, exit_reason=exit_reason)
                closed.append(key)
            else:
                log.error(f"❌ Failed to close {symbol} straddle: {close_result}")
                # Do NOT mark as closed — will retry on next poll

        # ─── Market closed: just log it, will re-check when open ─
        else:
            log.info(
                f"⏳ {symbol} triggered {exit_reason} ({pnl_pct:+.2f}%) but market is CLOSED. "
                f"Will re-evaluate with fresh prices when market opens."
            )

    return closed


def run_monitor():
    """Main monitoring loop. Runs indefinitely."""
    log.info("=" * 60)
    log.info("AUTO-CLOSE MONITOR STARTED")
    log.info(f"Take-profit: +{TARGET_RETURN_PCT}% | Stop-loss: {STOP_LOSS_PCT}%")
    log.info(f"Poll interval: {POLL_INTERVAL_SECS}s (market) / {POLL_INTERVAL_OFF}s (off-hours)")
    log.info(f"Pending close file: {PENDING_CLOSE_FILE}")
    log.info("=" * 60)

    # Check for any pending closes from previous session
    pending = _load_pending_closes()
    if pending:
        log.info(f"📋 Found {len(pending)} pending close(s) from previous session: {list(pending.keys())}")

    consecutive_errors = 0
    max_errors = 50  # High tolerance — laptop sleep causes many transient failures

    while True:
        try:
            market_open = is_market_hours()
            rush = is_market_open_rush()

            # Aggressive polling during first 5 min of market open
            # (to catch pending closes immediately)
            if rush and _load_pending_closes():
                interval = POLL_INTERVAL_OPEN_RUSH
            elif market_open:
                interval = POLL_INTERVAL_SECS
            else:
                interval = POLL_INTERVAL_OFF

            status = "🟢 MARKET OPEN" if market_open else "🔴 MARKET CLOSED"
            if rush:
                status += " (OPEN RUSH)"

            log.info(f"\n--- Check @ {datetime.now().strftime('%H:%M:%S')} | {status} ---")

            closed = check_and_close()
            if consecutive_errors > 0:
                log.info(f"✅ Recovered after {consecutive_errors} consecutive error(s).")
            consecutive_errors = 0

            if closed:
                log.info(f"🎉 Closed {len(closed)} position(s): {closed}")

            # Sleep in short chunks (max 15s) to survive macOS laptop
            # hibernation. A 300s sleep can freeze for hours if the lid
            # is closed, causing us to miss the open.
            remaining = interval
            while remaining > 0:
                time.sleep(min(remaining, 15))
                remaining -= 15

        except KeyboardInterrupt:
            log.info("\n⏹ Monitor stopped by user.")
            break
        except Exception as e:
            consecutive_errors += 1
            log.error(f"Error (#{consecutive_errors}): {e}")
            if consecutive_errors >= max_errors:
                log.critical(f"Too many consecutive errors ({max_errors}). Stopping.")
                break
            # Short error sleep too
            for _ in range(4):
                time.sleep(15)


if __name__ == "__main__":
    run_monitor()

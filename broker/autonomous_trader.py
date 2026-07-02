"""
Autonomous Trader (AT) — continuous intraday scan + execute + monitor.

Architecture:
    1. 9:30 AM: Execute pre-commanded trades from night-before AT scan
    2. 10:00 AM: Calculate cash position → run AT scan → validate → execute
    3. Every 30 min: Recalculate cash → rescan → execute all passing candidates
    4. Continuously: Monitor ALL open positions for TP (+5%) / SL (-50%)

Budget tracking:
    cash_position = starting_budget
                    + sum(closed_proceeds)
                    - sum(open_position_costs)

Strict AT filters (in order):
    1. Persistence ≥ 0.70 (undampened)
    2. IV Rank < 50
    3. Margin > $1.00 past breakeven
    4. Realized vs Predicted ≥ 0.50
    5. Volume ≥ 10 both legs
    6. Yahoo/Webull consistency check
    7. All passing candidates executed if budget allows

Usage:
    # Dry run (no real orders)
    python broker/autonomous_trader.py --budget 1527 --dry-run

    # Live with pre-commanded trades
    python broker/autonomous_trader.py --budget 1527 \\
        --pre-commanded "HOOD:straddle:106:2026-07-10"

    # Background daemon
    nohup python broker/autonomous_trader.py --budget 1527 \\
        >> cache/autonomous_trader.log 2>&1 &
"""
import sys
import os
import time
import json
import argparse
import logging
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.clock import now as et_now, market_status, timestamp as et_timestamp
from config import (
    TAKE_PROFIT_PCT, STOP_LOSS_PCT,
    MIN_MARGIN_THRESHOLD, IV_RANK_MAX, REJECT_DAMPENED,
    REALIZED_VS_PREDICTED_MIN, MIN_PERSISTENCE,
    SCAN_INTERVAL_MINUTES, MIN_SCAN_BUDGET, AUTONOMOUS_TP_PCT,
    MAX_STRANGLE_SPREAD_PCT,
)
import broker.webull_client as wb
from broker.webull_client import (
    get_accounts, place_straddle, close_straddle, get_positions,
)
from broker.position_tracker import fetch_live_positions
from broker.auto_close import is_market_hours, _log_closed_trade

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
LOG_FILE = os.path.join(LOG_DIR, "autonomous_trader.log")
STATE_FILE = os.path.join(LOG_DIR, "autonomous_state.json")
os.makedirs(LOG_DIR, exist_ok=True)

log = logging.getLogger("autonomous")
log.setLevel(logging.INFO)
# Prevent duplicate handlers on re-import
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(ch)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)


# ═══════════════════════════════════════════════════════════════════
# Autonomous Trader
# ═══════════════════════════════════════════════════════════════════

class AutonomousTrader:
    def __init__(self, starting_budget, pre_commanded=None,
                 tp_pct=AUTONOMOUS_TP_PCT, sl_pct=STOP_LOSS_PCT,
                 dry_run=False):
        self.starting_budget = starting_budget
        self.pre_commanded = pre_commanded or []
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.dry_run = dry_run

        # ─── State ────────────────────────────────────────────────
        self.open_positions = {}    # key → {ticker, strategy, cost, trade_data, ...}
        self.closed_today = []      # list of {ticker, cost, proceeds, pnl, ...}
        self.last_scan_time = None
        self.scan_count = 0

        self._load_state()

        log.info("=" * 60)
        log.info("  AUTONOMOUS TRADER (AT)")
        log.info("=" * 60)
        log.info(f"  🕐 {et_timestamp()}")
        log.info(f"  💰 Starting budget: ${starting_budget:,.2f}")
        log.info(f"  🎯 TP: +{tp_pct}% | SL: {sl_pct}%")
        log.info(f"  📊 Scan every {SCAN_INTERVAL_MINUTES} min")
        log.info(f"  🔒 Min persistence: {MIN_PERSISTENCE}")
        log.info(f"  🔴 Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        if self.pre_commanded:
            log.info(f"  📋 Pre-commanded: {len(self.pre_commanded)} trades")
            for pc in self.pre_commanded:
                log.info(f"      {pc}")
        log.info("=" * 60)

    # ─── Budget Tracking ──────────────────────────────────────────

    def calculate_cash_position(self):
        """
        cash = starting_budget
               + sum(closed proceeds)
               - sum(open position costs)
        """
        cash = self.starting_budget
        for closed in self.closed_today:
            cash += closed['proceeds']
        for key, pos in self.open_positions.items():
            cash -= pos['cost']
        return cash

    # ─── State Persistence ────────────────────────────────────────

    def _save_state(self):
        """Save state to disk for crash recovery."""
        state = {
            'starting_budget': self.starting_budget,
            'open_positions': self.open_positions,
            'closed_today': self.closed_today,
            'scan_count': self.scan_count,
            'last_save': et_now().isoformat(),
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load state from disk if exists."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.open_positions = state.get('open_positions', {})
                self.closed_today = state.get('closed_today', [])
                self.scan_count = state.get('scan_count', 0)
                log.info(f"📂 Restored state: {len(self.open_positions)} open, "
                         f"{len(self.closed_today)} closed today")
                return True
            except Exception as e:
                log.warning(f"Failed to load state: {e}")
        return False

    # ─── Market Timing ────────────────────────────────────────────

    def wait_for_market_open(self):
        """Block until 9:30 AM ET. Handles weekends and laptop sleep."""
        last_log_30 = -1  # Log every 30 min, not every minute
        while True:
            n = et_now()
            if n.weekday() >= 5:
                next_monday = n + timedelta(days=(7 - n.weekday()))
                next_open = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
                wait_secs = (next_open - n).total_seconds()
                log.info(f"Weekend. Next open in {wait_secs/3600:.1f} hours.")
                time.sleep(min(wait_secs, 900))
                continue

            market_open = n.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = n.replace(hour=16, minute=0, second=0, microsecond=0)

            if n >= market_open and n < market_close:
                log.info("🟢 Market is OPEN.")
                return

            if n < market_open:
                remaining = (market_open - n).total_seconds()
                current_30 = int(remaining / 1800)  # 30-min blocks
                if current_30 != last_log_30:
                    hours = int(remaining // 3600)
                    mins = int((remaining % 3600) // 60)
                    log.info(f"  ⏳ {hours}h {mins}m until market open...")
                    last_log_30 = current_30
                time.sleep(min(remaining, 15))
                continue

            log.info("🔴 Market closed for today. Waiting for tomorrow...")
            time.sleep(15)

    # ─── Position Opening ─────────────────────────────────────────

    def verify_margin_before_execution(self, candidate):
        """
        Last-mile margin check right before placing an order.
        Re-fetches fresh yfinance option prices and recalculates margin
        against the GARCH predicted range from the scan.

        Returns (True, fresh_cost) if margin is acceptable,
                (False, None) if margin has eroded.
        """
        import warnings as _w
        _w.filterwarnings('ignore')
        import yfinance as yf
        import pandas as pd

        ticker = candidate['ticker']
        strategy = candidate['strategy']
        strike = candidate['strike']
        expiry = candidate['expiry']
        scan_margin = candidate.get('margin', 0)
        scan_move = candidate.get('move_pct', 0)

        try:
            tk = yf.Ticker(ticker)
            chain = tk.option_chain(expiry)
            spot = tk.history(period='1d')['Close'].iloc[-1]

            if strategy == 'straddle':
                c = chain.calls[chain.calls['strike'] == float(strike)]
                p = chain.puts[chain.puts['strike'] == float(strike)]
                if c.empty or p.empty:
                    log.warning(f"   ⚠️  Margin verify: strike ${strike} not in fresh chain")
                    return True, None  # Can't verify, proceed
                c, p = c.iloc[0], p.iloc[0]
                c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (c['bid'] + c['ask']) / 2
                p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (p['bid'] + p['ask']) / 2
                fresh_premium = c_price + p_price
                be_up = float(strike) + fresh_premium
                be_down = float(strike) - fresh_premium
            else:
                call_s = strike.get('call') if isinstance(strike, dict) else strike
                put_s = strike.get('put') if isinstance(strike, dict) else strike
                c = chain.calls[chain.calls['strike'] == float(call_s)]
                p = chain.puts[chain.puts['strike'] == float(put_s)]
                if c.empty or p.empty:
                    log.warning(f"   ⚠️  Margin verify: strikes not in fresh chain")
                    return True, None
                c, p = c.iloc[0], p.iloc[0]
                c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (c['bid'] + c['ask']) / 2
                p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (p['bid'] + p['ask']) / 2
                fresh_premium = c_price + p_price
                be_up = float(call_s) + fresh_premium
                be_down = float(put_s) - fresh_premium

            # Recalculate margin using GARCH range
            # GARCH range = spot ± move_pct%
            if scan_move > 0:
                upper_1sig = spot * (1 + scan_move / 100)
                lower_1sig = spot * (1 - scan_move / 100)
            else:
                # Fallback: use scan margin directly
                log.info(f"   ⚠️  No move_pct in candidate, using scan margin")
                return True, None

            upside_margin = upper_1sig - be_up
            downside_margin = be_down - lower_1sig
            fresh_margin = max(upside_margin, downside_margin)

            # Calculate fresh cost
            budget = candidate.get('cost', 0)
            contracts = candidate.get('contracts', 1)
            fresh_cost = fresh_premium * 100 * contracts + 1.30

            log.info(f"   📊 Margin verify: {ticker}")
            log.info(f"      Spot: ${spot:.2f} | Premium: ${fresh_premium:.2f} "
                     f"(scan: ${candidate.get('premium', 0):.2f})")
            log.info(f"      BE: ${be_down:.2f}—${be_up:.2f} | "
                     f"GARCH: ${lower_1sig:.2f}—${upper_1sig:.2f}")
            log.info(f"      Margin: ${fresh_margin:.2f} "
                     f"(scan: ${scan_margin:.2f}) | "
                     f"Cost: ${fresh_cost:.2f}")

            if fresh_margin < MIN_MARGIN_THRESHOLD:
                log.warning(f"   ❌ MARGIN ERODED: ${fresh_margin:.2f} < "
                            f"${MIN_MARGIN_THRESHOLD:.2f} — ABORTING")
                return False, None

            if fresh_margin <= 0:
                log.warning(f"   ❌ NEGATIVE MARGIN: ${fresh_margin:.2f} — ABORTING")
                return False, None

            log.info(f"   ✅ Margin acceptable: ${fresh_margin:.2f}")
            return True, fresh_cost

        except Exception as e:
            log.warning(f"   ⚠️  Margin verify failed ({e}) — proceeding with scan data")
            return True, None

    def open_position(self, ticker, strategy, strike, expiry, budget_for_this):
        """
        Open a straddle or strangle position.
        Returns cost if successful, None if failed.
        """
        from broker.auto_trader import open_straddle, open_strangle

        log.info(f"\n📈 OPENING {strategy.upper()}: {ticker} ${strike} exp {expiry}")
        log.info(f"   Budget: ${budget_for_this:.2f} | Mode: {'DRY' if self.dry_run else '🔴 LIVE'}")

        if strategy == 'straddle':
            trade = open_straddle(
                ticker=ticker,
                strike=str(strike),
                expiry=expiry,
                budget=budget_for_this,
                dry_run=self.dry_run,
            )
        elif strategy == 'strangle':
            call_strike = strike.get('call') if isinstance(strike, dict) else strike
            put_strike = strike.get('put') if isinstance(strike, dict) else strike
            trade = open_strangle(
                ticker=ticker,
                call_strike=str(call_strike),
                put_strike=str(put_strike),
                expiry=expiry,
                budget=budget_for_this,
                dry_run=self.dry_run,
            )
        else:
            log.error(f"Unknown strategy: {strategy}")
            return None

        if trade:
            cost = trade.get('total_cost', budget_for_this)
            key = f"{ticker}_{strike}_{expiry}_{et_now().strftime('%H%M')}"
            self.open_positions[key] = {
                'ticker': ticker,
                'strategy': strategy,
                'strike': strike,
                'expiry': expiry,
                'cost': cost,
                'opened_at': et_now().isoformat(),
                'trade_data': trade,
            }
            self._save_state()
            log.info(f"   ✅ Opened {ticker} {strategy} | Cost: ${cost:.2f}")
            return cost
        else:
            log.warning(f"   ❌ Failed to open {ticker} {strategy}")
            return None

    # ─── Position Monitoring ──────────────────────────────────────

    def monitor_all_positions(self):
        """
        Check TP/SL on every open position using Webull live data.
        Close any that hit thresholds.
        Also detects manual closes and syncs state.
        """
        if not self.open_positions:
            return

        if market_status() != "OPEN":
            return

        try:
            live_data = fetch_live_positions()
        except Exception as e:
            log.error(f"Monitor: Failed to fetch positions: {e}")
            return

        if not live_data:
            log.warning(f"Monitor: No live data returned")
            return

        # fetch_live_positions() returns dict with 'straddles' key
        straddles = live_data.get('straddles', {})
        if not straddles:
            # No positions on Webull — everything was closed externally
            if self.open_positions:
                log.info(f"   🔄 SYNC: Webull shows 0 positions but AT has "
                         f"{len(self.open_positions)} — detecting manual closes")
                self._sync_manual_closes(straddles)
            return

        # Log what Webull sees
        log.info(f"   📡 Webull: {len(straddles)} position group(s)")

        to_close = []
        matched_keys = set()  # Track which AT positions matched Webull

        for key, pos in list(self.open_positions.items()):
            ticker = pos['ticker']
            strike = pos['strike']
            expiry = pos['expiry']
            cost = pos['cost']

            # Match against Webull straddles
            # Webull keys look like: "HOOD_102.0_2026-07-10"
            matched = None
            for live_key, live_pos in straddles.items():
                live_symbol = live_pos.get('symbol', '').upper()
                if ticker.upper() == live_symbol:
                    # For strangles, check both strikes
                    if isinstance(strike, dict):
                        # Strangle: check if either call or put strike matches
                        live_strike = live_pos.get('strike', 0)
                        if (live_strike == strike.get('call') or
                            live_strike == strike.get('put')):
                            matched = live_pos
                            matched_keys.add(key)
                            break
                    else:
                        # Straddle: direct strike match
                        live_strike = live_pos.get('strike', 0)
                        if abs(live_strike - float(strike)) < 0.01:
                            matched = live_pos
                            matched_keys.add(key)
                            break

            # If no exact match, try loose match (same ticker)
            if not matched:
                for live_key, live_pos in straddles.items():
                    if ticker.upper() == live_pos.get('symbol', '').upper():
                        matched = live_pos
                        matched_keys.add(key)
                        break

            if not matched:
                continue

            # Calculate P&L from Webull data
            total_value = matched.get('total_value', 0)
            total_pnl = matched.get('total_pnl', 0)
            pnl_pct = matched.get('pnl_pct', 0)

            # Fallback: calculate from cost if Webull P&L is zero
            if total_value > 0 and total_pnl == 0:
                total_pnl = total_value - cost
                pnl_pct = (total_pnl / cost) * 100 if cost > 0 else 0

            strike_str = (f"${strike['call']}C/${strike['put']}P"
                          if isinstance(strike, dict) else f"${strike}")
            log.info(f"   {ticker} {strike_str} {pos['strategy']}: "
                     f"val ${total_value:.2f} | P&L ${total_pnl:+.2f} ({pnl_pct:+.1f}%) | "
                     f"TP: +{self.tp_pct}% | SL: {self.sl_pct}%")

            # Check TP
            if pnl_pct >= self.tp_pct:
                log.info(f"   🎯 {ticker} HIT TP (+{self.tp_pct}%)! Closing...")
                to_close.append((key, 'TP', total_value, total_pnl, pnl_pct))

            # Check SL
            elif pnl_pct <= self.sl_pct:
                log.info(f"   🛑 {ticker} HIT SL ({self.sl_pct}%)! Closing...")
                to_close.append((key, 'SL', total_value, total_pnl, pnl_pct))

        # Execute closes
        for key, reason, proceeds, pnl, pnl_pct in to_close:
            self._close_position(key, reason, proceeds, pnl, pnl_pct)

        # Detect manual closes: AT positions not found on Webull
        self._sync_manual_closes(straddles)

    def _sync_manual_closes(self, webull_straddles):
        """
        Detect positions that AT thinks are open but Webull doesn't have.
        These were closed manually by the user.
        """
        webull_tickers = set()
        for live_key, live_pos in webull_straddles.items():
            webull_tickers.add(live_pos.get('symbol', '').upper())

        to_remove = []
        for key, pos in self.open_positions.items():
            ticker = pos['ticker'].upper()
            # If ticker not on Webull at all, it was manually closed
            if ticker not in webull_tickers:
                log.info(f"   🔄 MANUAL CLOSE DETECTED: {pos['ticker']} "
                         f"{pos['strategy']} (cost ${pos['cost']:.2f}) "
                         f"— no longer on Webull")
                to_remove.append(key)

        for key in to_remove:
            pos = self.open_positions[key]
            # Record as manual close with unknown proceeds
            self.closed_today.append({
                'ticker': pos['ticker'],
                'strategy': pos['strategy'],
                'cost': pos['cost'],
                'proceeds': pos['cost'],  # Assume breakeven for budget calc
                'pnl': 0,
                'pnl_pct': 0,
                'reason': 'MANUAL_CLOSE',
                'closed_at': et_now().isoformat(),
            })
            del self.open_positions[key]
            log.info(f"   💰 Budget freed: ${pos['cost']:.2f} "
                     f"(assuming breakeven for manual close)")

        if to_remove:
            self._save_state()
            new_cash = self.calculate_cash_position()
            log.info(f"   💰 Cash after sync: ${new_cash:,.2f}")

    def _close_position(self, key, reason, proceeds, pnl, pnl_pct):
        """Close a position and update budget. Handles both straddles and strangles."""
        pos = self.open_positions[key]
        ticker = pos['ticker']
        strike = pos['strike']
        strategy = pos['strategy']

        log.info(f"\n{'='*60}")
        log.info(f"🔴 CLOSING {ticker} {strategy.upper()} ({reason})")
        log.info(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        log.info(f"{'='*60}")

        if not self.dry_run:
            try:
                from broker.webull_client import (
                    place_combo_strangle, place_combo_straddle,
                    get_accounts,
                )
                import yfinance as yf

                # Get account
                accounts = get_accounts()
                if not accounts:
                    log.error(f"   ❌ No accounts found")
                    return
                account_id = accounts[0]['account_id']
                for a in accounts:
                    if a.get('account_type') == 'MARGIN':
                        account_id = a['account_id']
                        break

                # Fetch fresh bid prices from yfinance for sell limits
                tk = yf.Ticker(ticker)
                expiry = pos['expiry']
                chain = tk.option_chain(expiry)

                if strategy == 'strangle' and isinstance(strike, dict):
                    call_s = strike['call']
                    put_s = strike['put']

                    c = chain.calls[chain.calls['strike'] == float(call_s)]
                    p = chain.puts[chain.puts['strike'] == float(put_s)]
                    if c.empty or p.empty:
                        log.error(f"   ❌ Can't find strikes in chain")
                        return

                    c_bid = float(c.iloc[0]['bid']) if c.iloc[0]['bid'] > 0 else float(c.iloc[0]['lastPrice'])
                    p_bid = float(p.iloc[0]['bid']) if p.iloc[0]['bid'] > 0 else float(p.iloc[0]['lastPrice'])

                    log.info(f"   Sell call ${call_s} @ ${c_bid:.2f} (bid)")
                    log.info(f"   Sell put  ${put_s} @ ${p_bid:.2f} (bid)")

                    result = place_combo_strangle(
                        account_id=account_id,
                        symbol=ticker,
                        call_strike=str(call_s),
                        put_strike=str(put_s),
                        expiry=expiry,
                        side="SELL",
                        quantity=1,
                        call_limit=f"{c_bid:.2f}",
                        put_limit=f"{p_bid:.2f}",
                    )

                else:
                    # Straddle close
                    s = float(strike) if not isinstance(strike, dict) else float(strike.get('call', strike))
                    c = chain.calls[chain.calls['strike'] == s]
                    p = chain.puts[chain.puts['strike'] == s]
                    if c.empty or p.empty:
                        log.error(f"   ❌ Can't find strike ${s} in chain")
                        return

                    c_bid = float(c.iloc[0]['bid']) if c.iloc[0]['bid'] > 0 else float(c.iloc[0]['lastPrice'])
                    p_bid = float(p.iloc[0]['bid']) if p.iloc[0]['bid'] > 0 else float(p.iloc[0]['lastPrice'])

                    log.info(f"   Sell call ${s} @ ${c_bid:.2f} (bid)")
                    log.info(f"   Sell put  ${s} @ ${p_bid:.2f} (bid)")

                    result = place_combo_straddle(
                        account_id=account_id,
                        symbol=ticker,
                        strike=str(s),
                        expiry=expiry,
                        side="SELL",
                        quantity=1,
                        call_limit=f"{c_bid:.2f}",
                        put_limit=f"{p_bid:.2f}",
                    )

                if not result.get('success'):
                    log.error(f"   ❌ Close order failed: {result}")
                    return
                log.info(f"   ✅ Close order placed successfully")

            except Exception as e:
                log.error(f"   ❌ Close error for {ticker}: {e}")
                return

        # Record the close
        self.closed_today.append({
            'ticker': ticker,
            'strategy': strategy,
            'cost': pos['cost'],
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'closed_at': et_now().isoformat(),
        })

        del self.open_positions[key]
        self._save_state()

        new_cash = self.calculate_cash_position()
        log.info(f"   ✅ CLOSED {ticker} ({reason}) | "
                 f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | "
                 f"💰 Cash now: ${new_cash:,.2f}")

    # ─── AT Scanning ──────────────────────────────────────────────

    def run_at_scan(self, budget):
        """
        Run both straddle + strangle AT scanners with strict autonomous filters.
        Returns list of validated candidates sorted by score.
        """
        import warnings
        warnings.filterwarnings('ignore')
        from models.garch_model import GARCHVolatilityModel
        from data.fetcher import fetch_price_data
        from signals.scanner import SCAN_UNIVERSE
        from signals.strangle_scanner import (
            compute_iv_rank, check_upcoming_earnings,
            scan_strangle_opportunities,
        )
        import yfinance as yf
        import pandas as pd

        self.scan_count += 1
        log.info(f"\n{'='*60}")
        log.info(f"🔍 AT SCAN #{self.scan_count} | {et_timestamp()}")
        log.info(f"   Budget: ${budget:,.2f}")
        log.info(f"{'='*60}")

        all_candidates = []

        # ─── Straddle scan ────────────────────────────────────────
        target_date = datetime.now() + timedelta(days=14)
        straddle_pass = 0
        straddle_reject = 0

        for sym in SCAN_UNIVERSE:
            try:
                tk = yf.Ticker(sym)
                exps = tk.options
                if not exps:
                    continue
                best_exp = min(exps, key=lambda x: abs(
                    datetime.strptime(x, '%Y-%m-%d') - target_date))
                chain = tk.option_chain(best_exp)
                prices = fetch_price_data(sym)
                prices = prices.dropna(subset=['Close'])
                spot = prices['Close'].iloc[-1]
                strike = round(spot)

                c = chain.calls[chain.calls['strike'] == strike]
                p = chain.puts[chain.puts['strike'] == strike]
                if c.empty or p.empty:
                    continue
                c, p = c.iloc[0], p.iloc[0]

                c_vol = int(c['volume']) if not pd.isna(c.get('volume', 0)) else 0
                p_vol = int(p['volume']) if not pd.isna(p.get('volume', 0)) else 0
                if c_vol < 10 or p_vol < 10:
                    continue

                c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (c['bid'] + c['ask']) / 2
                p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (p['bid'] + p['ask']) / 2
                if c_price < 0.15 or p_price < 0.15:
                    continue

                premium = c_price + p_price
                contracts = int((budget - 1.30) / (premium * 100))
                if contracts < 1:
                    continue
                total_cost = premium * 100 * contracts + 1.30

                # GARCH
                garch = GARCHVolatilityModel()
                garch.fit(prices, verbose=False)
                exp_date = datetime.strptime(best_exp, '%Y-%m-%d')
                holding_days = max(1, int((exp_date - datetime.now()).days * 5 / 7))
                pr = garch.forecast_price_range(spot, horizon_days=holding_days)

                be_up = strike + premium
                be_down = strike - premium
                best_margin = max(pr['upper_1sig'] - be_up, be_down - pr['lower_1sig'])

                # ─── STRICT FILTERS ───────────────────────────────
                if best_margin <= 0 or best_margin < MIN_MARGIN_THRESHOLD:
                    straddle_reject += 1; continue
                if garch.persistence < MIN_PERSISTENCE:
                    straddle_reject += 1; continue
                if REJECT_DAMPENED and garch.dampened:
                    straddle_reject += 1; continue

                iv_rank = compute_iv_rank(tk, spot, chain)
                if iv_rank is not None and iv_rank > IV_RANK_MAX:
                    straddle_reject += 1; continue

                recent_rets = prices['Close'].pct_change().dropna().tail(5)
                realized_5d = recent_rets.abs().mean() * np.sqrt(holding_days) * 100
                predicted = pr['expected_move_pct']
                if predicted > 0 and realized_5d / predicted < REALIZED_VS_PREDICTED_MIN:
                    straddle_reject += 1; continue

                # Score
                margin_score = min(1.0, best_margin / max(premium, 0.01))
                liq_score = min(1.0, np.log10(max(c_vol + p_vol, 1)) / 4)
                composite = 0.60 * margin_score + 0.40 * liq_score

                straddle_pass += 1
                all_candidates.append({
                    'ticker': sym, 'strategy': 'straddle',
                    'strike': strike, 'expiry': best_exp,
                    'c_price': round(c_price, 2), 'p_price': round(p_price, 2),
                    'premium': round(premium, 2), 'cost': round(total_cost, 2),
                    'contracts': contracts, 'margin': round(best_margin, 2),
                    'iv_rank': iv_rank, 'persistence': round(garch.persistence, 3),
                    'move_pct': pr['expected_move_pct'],
                    'score': round(composite, 3),
                })
            except Exception:
                continue

        # ─── Strangle scan ────────────────────────────────────────
        try:
            strangle_results, _ = scan_strangle_opportunities(budget=budget)
            for r in (strangle_results or []):
                # Apply strict persistence filter
                if r.get('persistence', 0) < MIN_PERSISTENCE:
                    continue
                if r.get('dampened', False):
                    continue

                # Spread width filter
                spread_pct = (r['call_strike'] - r['put_strike']) / r.get('spot', 1)
                if spread_pct > MAX_STRANGLE_SPREAD_PCT:
                    log.info(f"   ❌ {r['ticker']} strangle spread too wide: "
                             f"${r['call_strike']}-${r['put_strike']} = "
                             f"{spread_pct*100:.1f}% > {MAX_STRANGLE_SPREAD_PCT*100:.0f}% max")
                    continue

                all_candidates.append({
                    'ticker': r['ticker'], 'strategy': 'strangle',
                    'strike': {'call': r['call_strike'], 'put': r['put_strike']},
                    'expiry': r['expiry'],
                    'c_price': r.get('call_price', 0), 'p_price': r.get('put_price', 0),
                    'premium': r.get('strangle_cost', 0),
                    'cost': r.get('total_cost', 0),
                    'contracts': r.get('contracts', 1),
                    'margin': r.get('best_margin', 0),
                    'iv_rank': r.get('iv_rank'),
                    'persistence': r.get('persistence', 0),
                    'move_pct': r.get('expected_move_pct', 0),
                    'score': r.get('composite_score', 0),
                })
        except Exception as e:
            log.error(f"   Strangle scan error: {e}")

        # Sort by score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        log.info(f"   Straddle: {straddle_pass} pass, {straddle_reject} reject")
        log.info(f"   Total candidates: {len(all_candidates)}")
        for i, c in enumerate(all_candidates):
            iv_str = f"IV{c['iv_rank']:.0f}" if c['iv_rank'] else "IV?"
            log.info(f"      #{i+1}: [{c['strategy'].upper()}] {c['ticker']} "
                     f"${c['strike']} | ${c['cost']:.0f} | "
                     f"margin ${c['margin']:.2f} | {iv_str} | "
                     f"persist {c['persistence']:.3f} | score {c['score']:.3f}")

        return all_candidates

    # ─── Webull Validation ────────────────────────────────────────

    def validate_candidate(self, candidate):
        """
        Run Yahoo/Webull consistency check.
        Returns True if opportunity is still valid.

        If Webull API is unavailable (404), skip validation and proceed.
        Only reject on actual data-quality issues (wide spreads, margin erosion).
        """
        if market_status() != "OPEN":
            log.info(f"   ⚠️  Market not open — skipping validation for {candidate['ticker']}")
            return True

        try:
            from signals.quote_validator import validate_opportunity

            # Normalize strike format for validator
            scan_data = dict(candidate)
            if isinstance(scan_data.get('strike'), dict):
                scan_data['call_strike'] = scan_data['strike']['call']
                scan_data['put_strike'] = scan_data['strike']['put']

            result = validate_opportunity(scan_data, strategy=candidate['strategy'])

            if result['valid']:
                log.info(f"   ✅ {candidate['ticker']} validated")
                return True
            else:
                # Check if rejection was just "Webull unavailable" — that's not a real rejection
                rejections = result.get('rejections', [])
                webull_unavailable = any('unavailable' in r.lower() for r in rejections)
                if webull_unavailable and len(rejections) == 1:
                    log.info(f"   ⚠️  {candidate['ticker']} — Webull API unavailable, "
                             f"proceeding with yfinance data only")
                    return True
                else:
                    log.warning(f"   ❌ {candidate['ticker']} REJECTED: {rejections}")
                    return False
        except Exception as e:
            log.warning(f"   ⚠️  Validation error ({e}) — proceeding with yfinance data")
            return True

    # ─── Main Loop ────────────────────────────────────────────────

    def run(self):
        """Main execution loop."""
        # Wait for market
        self.wait_for_market_open()

        # ─── Phase 1: Pre-commanded trades at 9:30 ───────────────
        if self.pre_commanded:
            log.info("\n" + "─" * 60)
            log.info("PHASE 1: EXECUTING PRE-COMMANDED TRADES")
            log.info("─" * 60)

            for pc in self.pre_commanded:
                parts = pc.split(':')
                if len(parts) != 4:
                    log.error(f"   Invalid pre-commanded format: {pc}")
                    log.error(f"   Expected: TICKER:strategy:strike:expiry")
                    log.error(f"   Strangle: TICKER:strangle:CALL/PUT:expiry")
                    continue
                ticker, strategy, strike_str, expiry = parts

                # Parse strike — strangles use "108/98" format
                if '/' in strike_str:
                    call_s, put_s = strike_str.split('/')
                    strike = {'call': int(call_s), 'put': int(put_s)}
                else:
                    strike = float(strike_str) if '.' in strike_str else int(strike_str)

                # ─── IV Rank Safety Check ─────────────────────────
                try:
                    import yfinance as yf
                    from signals.strangle_scanner import compute_iv_rank
                    tk = yf.Ticker(ticker)
                    spot = tk.history(period='1d')['Close'].iloc[-1]
                    chain = tk.option_chain(expiry)
                    iv_rank = compute_iv_rank(tk, spot, chain)
                    if iv_rank is not None:
                        log.info(f"   📊 {ticker} IV Rank: {iv_rank:.0f} (max: {IV_RANK_MAX})")
                        if iv_rank > IV_RANK_MAX:
                            log.warning(f"   ❌ SKIPPING {ticker}: IV Rank {iv_rank:.0f} > {IV_RANK_MAX} — options overpriced")
                            continue
                    else:
                        log.info(f"   📊 {ticker} IV Rank: could not compute — proceeding anyway")
                except Exception as e:
                    log.warning(f"   ⚠️ IV rank check failed: {e} — proceeding anyway")

                cash = self.calculate_cash_position()
                log.info(f"   📋 {ticker} {strategy} ${strike_str} exp {expiry}")
                cost = self.open_position(ticker, strategy, strike, expiry,
                                          budget_for_this=cash)
                if cost:
                    log.info(f"   💰 Cash after: ${self.calculate_cash_position():,.2f}")

            # Wait 60s for fills
            if not self.dry_run:
                log.info("   ⏳ Waiting 60s for order fills...")
                time.sleep(60)

        # ─── Phase 2: Continuous scan + monitor loop ─────────────
        log.info("\n" + "─" * 60)
        log.info("PHASE 2: CONTINUOUS SCAN + MONITOR LOOP")
        log.info(f"   Scanning every {SCAN_INTERVAL_MINUTES} min | TP +{self.tp_pct}% | SL {self.sl_pct}%")
        log.info("─" * 60)

        while True:
            try:
                n = et_now()

                # Stop after market close (4 PM)
                if n.hour >= 16:
                    log.info("\n🔴 Market closed. Final summary:")
                    self._print_summary()
                    break

                # Monitor all positions (every poll cycle)
                if market_status() == "OPEN":
                    self.monitor_all_positions()

                # Run AT scan every 30 minutes
                should_scan = (
                    self.last_scan_time is None or
                    (n - self.last_scan_time).total_seconds() >= SCAN_INTERVAL_MINUTES * 60
                )

                if should_scan and market_status() == "OPEN":
                    cash = self.calculate_cash_position()
                    log.info(f"\n💰 Cash position: ${cash:,.2f}")

                    if cash >= MIN_SCAN_BUDGET:
                        candidates = self.run_at_scan(budget=cash)

                        # Execute ALL passing candidates if budget allows
                        remaining_cash = cash
                        for candidate in candidates:
                            if remaining_cash < MIN_SCAN_BUDGET:
                                log.info(f"   💸 Cash too low (${remaining_cash:.2f}) — stopping")
                                break

                            if candidate['cost'] > remaining_cash:
                                log.info(f"   💸 {candidate['ticker']} costs ${candidate['cost']:.0f} "
                                         f"but only ${remaining_cash:.0f} available — skipping")
                                continue

                            # Webull consistency check + margin verification
                            if self.validate_candidate(candidate):
                                # Last-mile margin check with fresh prices
                                margin_ok, fresh_cost = self.verify_margin_before_execution(candidate)
                                if not margin_ok:
                                    log.warning(f"   ❌ {candidate['ticker']} margin eroded — skipping")
                                    continue

                                cost = self.open_position(
                                    ticker=candidate['ticker'],
                                    strategy=candidate['strategy'],
                                    strike=candidate['strike'],
                                    expiry=candidate['expiry'],
                                    budget_for_this=remaining_cash,
                                )
                                if cost:
                                    remaining_cash -= cost
                                    log.info(f"   💰 Remaining cash: ${remaining_cash:,.2f}")

                                    # Wait for fill
                                    if not self.dry_run:
                                        time.sleep(30)
                    else:
                        log.info(f"   Cash ${cash:.2f} < ${MIN_SCAN_BUDGET} minimum — skipping scan")

                    self.last_scan_time = n

                # Poll interval: 30 seconds during market, 5 min off
                interval = 30 if market_status() == "OPEN" else 300
                time.sleep(interval)

            except KeyboardInterrupt:
                log.info("\n⏹  Stopped by user.")
                self._print_summary()
                break
            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(60)

    def _print_summary(self):
        """Print end-of-day summary."""
        log.info("\n" + "=" * 60)
        log.info("END OF DAY SUMMARY")
        log.info("=" * 60)
        log.info(f"   Starting budget:  ${self.starting_budget:,.2f}")
        log.info(f"   Scans executed:   {self.scan_count}")
        log.info(f"   Positions opened: {len(self.closed_today) + len(self.open_positions)}")
        log.info(f"   Positions closed: {len(self.closed_today)}")
        log.info(f"   Still open:       {len(self.open_positions)}")

        total_pnl = sum(c['pnl'] for c in self.closed_today)
        log.info(f"   Realized P&L:     ${total_pnl:+,.2f}")

        if self.closed_today:
            log.info("\n   Closed positions:")
            for c in self.closed_today:
                log.info(f"      {c['ticker']} {c['strategy']} | "
                         f"${c['pnl']:+.2f} ({c['pnl_pct']:+.1f}%) | {c['reason']}")

        if self.open_positions:
            log.info("\n   Still open:")
            for key, pos in self.open_positions.items():
                log.info(f"      {pos['ticker']} {pos['strategy']} | cost ${pos['cost']:.2f}")

        cash = self.calculate_cash_position()
        log.info(f"\n   Final cash:       ${cash:,.2f}")
        log.info("=" * 60)
        self._save_state()


# ─── CLI Entry Point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Trader (AT) — continuous intraday scan + execute + monitor",
    )
    parser.add_argument("--budget", "-b", type=float, required=True,
                        help="Starting budget (e.g., 1527)")
    parser.add_argument("--pre-commanded", type=str, default=None,
                        help="Comma-separated pre-commanded trades. "
                             "Format: TICKER:strategy:strike:expiry "
                             "(e.g., 'HOOD:straddle:106:2026-07-10')")
    parser.add_argument("--tp-pct", type=float, default=AUTONOMOUS_TP_PCT,
                        help=f"Take profit %% (default: {AUTONOMOUS_TP_PCT}%%)")
    parser.add_argument("--sl-pct", type=float, default=STOP_LOSS_PCT,
                        help=f"Stop loss %% (default: {STOP_LOSS_PCT}%%)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without placing real orders")

    args = parser.parse_args()

    pre_commanded = []
    if args.pre_commanded:
        pre_commanded = [t.strip() for t in args.pre_commanded.split(',')]

    trader = AutonomousTrader(
        starting_budget=args.budget,
        pre_commanded=pre_commanded,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        dry_run=args.dry_run,
    )
    trader.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
HAT v2 — Hybrid Autonomous Trader

An autonomous directional options trader that uses the Hybrid Scanner's
LSTM + Beta + GARCH signals to buy single-leg PUTs or CALLs.

v2 changes (2026-07-19):
  - Hard floor on option price ($0.30) — eliminates penny options
  - Minimum volume (50) and max bid-ask spread (20%) quality gates
  - Position cost clamped to $50-$60 range
  - Max 2 simultaneous positions (fewer, higher-conviction)
  - TP raised to +30%, SL widened to -35% (let winners run)
  - Trailing stop: once +15%, stop moves to breakeven
  - No hardcoded seed positions — all entries via live scanner
  - Dynamic cash recalculation after every close

Architecture:
  1. Waits for 9:30 AM market open
  2. Runs Hybrid Scanner for initial opportunities
  3. Monitors all open positions with TP/SL/trailing stop
  4. Every 30 minutes, re-scans for new opportunities if slots available
  5. Recalculates available cash after every close (cost + P&L returned)
  6. Loops until market close

Entirely separate from AT (straddle/strangle trader).
Hard capital allocation: $120 (does not coordinate with AT).

Usage:
    python3 broker/hat.py                    # Live trading
    python3 broker/hat.py --dry-run          # Simulate without real orders
    python3 broker/hat.py --budget 120       # Custom initial budget
"""
import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import broker.webull_client as wb
from broker.webull_client import (
    get_accounts, place_option_order, get_positions,
)
from broker.position_tracker import fetch_live_positions
from broker.auto_close import is_market_hours

# ═══════════════════════════════════════════════════════════════════
#  Configuration — HAT v2
# ═══════════════════════════════════════════════════════════════════

TP_PCT = 30.0                   # Take-profit: +30% — let winners run
SL_PCT = -35.0                  # Stop-loss: -35% — give trades room,
                                # don't stop-loss on bid-ask noise
TRAILING_STOP_PCT = 15.0        # Once a position reaches +15%, move stop
                                # to breakeven (entry price). Free roll.
SCAN_INTERVAL = 1800            # 30 minutes between hybrid scans
MONITOR_INTERVAL = 30           # 30 seconds between position checks
MIN_POSITION_COST = 50.0        # Min $50 per position — avoid tiny bets
MAX_POSITION_COST = 60.0        # Max $60 per position
MAX_POSITIONS = 2               # Max 2 simultaneous positions
INITIAL_BUDGET = 120.0          # $120 hard capital allocation

# ─── Quality Gates (v2) ──────────────────────────────────────────
# These filters eliminate penny options, illiquid contracts, and
# wide-spread traps that made v1 structurally unprofitable.
MIN_OPTION_PRICE = 0.30         # Hard floor — no penny options.
                                # At $0.30 ($30 cost), the typical
                                # bid-ask spread is ~3-5% vs 50%+ at $0.05.
MIN_OPTION_VOLUME = 50          # Minimum daily volume for acceptable
                                # liquidity — ensures we can exit.
MAX_BID_ASK_SPREAD_PCT = 0.20   # Reject if bid-ask spread > 20% of mid.
                                # Wide spreads eat profits on entry AND exit.

# Paths
STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cache", "hat_state.json"
)
LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cache", "hat.log"
)

# ─── Logging ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

log = logging.getLogger("hat")
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s [HAT] %(message)s"))
log.addHandler(ch)

fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(asctime)s [HAT] %(message)s"))
log.addHandler(fh)


# ═══════════════════════════════════════════════════════════════════
#  Seed Positions — REMOVED in v2
# ═══════════════════════════════════════════════════════════════════
# v1 used hardcoded seed positions (MARA, RIOT, RUN at $30-$40 each).
# These were all penny options that violated every quality gate.
# v2 uses the live Hybrid Scanner at market open instead — all entries
# must pass the quality gates (min price, volume, spread width).


# ═══════════════════════════════════════════════════════════════════
#  State Management
# ═══════════════════════════════════════════════════════════════════

class HATState:
    """Track positions, cash, P&L, and trailing stops for HAT v2."""

    def __init__(self, initial_budget: float):
        self.initial_budget = initial_budget
        self.cash = initial_budget
        self.open_positions = {}    # key -> position dict
        self.closed_positions = []  # history
        self.total_realized_pnl = 0.0
        self.high_water_marks = {}  # key -> highest P&L % seen
        self.trailing_active = {}   # key -> True if trailing stop engaged

    def record_open(self, key: str, ticker: str, strike: float,
                    expiry: str, option_type: str, cost: float,
                    price: float, contracts: int):
        """Record a newly opened position."""
        self.open_positions[key] = {
            "ticker": ticker,
            "strike": strike,
            "expiry": expiry,
            "option_type": option_type,
            "cost": cost,
            "entry_price": price,
            "contracts": contracts,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cash -= cost
        self.high_water_marks[key] = 0.0
        self.trailing_active[key] = False
        log.info(f"  📊 Cash: ${self.cash:.2f} (spent ${cost:.2f})")

    def record_close(self, key: str, exit_price: float, pnl: float,
                     reason: str):
        """
        Record a closed position.
        Returns capital + P&L to the cash pool for reinvestment.

        Example: $60 position closed at -20% loss
          → pnl = -$12
          → cash += $60 + (-$12) = $48 returned
          → If other position is open at $60, available cash = $48
        """
        pos = self.open_positions.pop(key, None)
        if pos:
            pos["exit_price"] = exit_price
            pos["pnl"] = pnl
            pos["pnl_pct"] = (pnl / pos["cost"]) * 100 if pos["cost"] > 0 else 0
            pos["reason"] = reason
            pos["closed_at"] = datetime.now(timezone.utc).isoformat()
            pos["high_water_mark"] = self.high_water_marks.pop(key, 0)
            pos["trailing_was_active"] = self.trailing_active.pop(key, False)
            self.closed_positions.append(pos)
            # Return capital + profit (or - loss) to cash pool
            returned = pos["cost"] + pnl
            self.cash += returned
            self.total_realized_pnl += pnl
            log.info(f"  💰 Cash: ${self.cash:.2f} (${returned:+.2f} returned)")
            log.info(f"     Breakdown: ${pos['cost']:.2f} cost + ${pnl:+.2f} P&L")

    @property
    def available_cash(self) -> float:
        return self.cash

    @property
    def open_cost(self) -> float:
        return sum(p["cost"] for p in self.open_positions.values())

    @property
    def n_open(self) -> int:
        return len(self.open_positions)

    def save(self):
        """Persist state to disk."""
        data = {
            "version": 2,
            "initial_budget": self.initial_budget,
            "cash": self.cash,
            "open_positions": self.open_positions,
            "closed_positions": self.closed_positions,
            "total_realized_pnl": self.total_realized_pnl,
            "high_water_marks": self.high_water_marks,
            "trailing_active": self.trailing_active,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls):
        """Load state from disk if available."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                state = cls(data["initial_budget"])
                state.cash = data["cash"]
                state.open_positions = data["open_positions"]
                state.closed_positions = data["closed_positions"]
                state.total_realized_pnl = data.get("total_realized_pnl", 0)
                state.high_water_marks = data.get("high_water_marks", {})
                state.trailing_active = data.get("trailing_active", {})
                return state
            except Exception:
                pass
        return None

    def summary(self):
        """Print state summary."""
        log.info(f"  ┌─ HAT v2 State ──────────────────────────────")
        log.info(f"  │ Initial budget:   ${self.initial_budget:.2f}")
        log.info(f"  │ Cash available:   ${self.cash:.2f}")
        log.info(f"  │ Open positions:   {self.n_open} / {MAX_POSITIONS}")
        log.info(f"  │ Open cost:        ${self.open_cost:.2f}")
        log.info(f"  │ Realized P&L:     ${self.total_realized_pnl:+.2f}")
        log.info(f"  │ Closed trades:    {len(self.closed_positions)}")
        log.info(f"  │ Account value:    ${self.cash + self.open_cost:.2f}")
        log.info(f"  └────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════
#  Market Timing
# ═══════════════════════════════════════════════════════════════════

def _get_et_now():
    import pytz
    return datetime.now(pytz.timezone("US/Eastern"))


def wait_for_market_open():
    """Block until 9:30 AM ET. Short sleep intervals for laptop resilience."""
    from datetime import timedelta as td
    last_log_min = -1

    while True:
        now = _get_et_now()
        if now.weekday() >= 5:
            log.info("Weekend. Sleeping 15 min...")
            time.sleep(900)
            continue

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if market_open <= now <= market_close:
            log.info("🟢 Market is open!")
            return

        if now < market_open:
            remaining = (market_open - now).total_seconds()
            mins = int(remaining / 60)
            if mins != last_log_min and mins % 5 == 0:
                log.info(f"⏳ {mins} minutes until market open...")
                last_log_min = mins
            time.sleep(min(15, remaining))
        else:
            log.info("🔴 Market closed for today.")
            return  # Don't block — let main loop handle


# ═══════════════════════════════════════════════════════════════════
#  Token / Account Management
# ═══════════════════════════════════════════════════════════════════

def ensure_token():
    """Ensure Webull access token is loaded."""
    if not wb.ACCESS_TOKEN:
        from dotenv import load_dotenv
        load_dotenv(wb._ENV_PATH)
        wb.ACCESS_TOKEN = os.getenv("WEBULL_ACCESS_TOKEN", "")
    return bool(wb.ACCESS_TOKEN)


def get_account_id():
    """Get the primary account ID."""
    accounts = get_accounts()
    if not accounts:
        return None
    for a in accounts:
        if a.get("account_type") == "MARGIN":
            return a["account_id"]
    return accounts[0]["account_id"]


# ═══════════════════════════════════════════════════════════════════
#  Order Execution
# ═══════════════════════════════════════════════════════════════════

def fetch_option_quote(ticker: str, strike: float, expiry: str,
                       option_type: str) -> dict:
    """Fetch a fresh quote for a single option leg."""
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        opts = chain.calls if option_type == "CALL" else chain.puts
        row = opts[opts['strike'] == strike]
        if row.empty:
            return None
        row = row.iloc[0]

        last = float(row['lastPrice'])
        bid = float(row['bid'])
        ask = float(row['ask'])
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        vol = int(row['volume']) if row['volume'] > 0 else 0
        spread_pct = (ask - bid) / mid if mid > 0 and bid > 0 and ask > 0 else 1.0

        return {
            "last": last, "bid": bid, "ask": ask, "mid": mid,
            "volume": vol, "iv": float(row.get('impliedVolatility', 0)),
            "spread_pct": round(spread_pct, 4),
        }
    except Exception as e:
        log.error(f"  Quote fetch failed for {ticker} ${strike} {option_type}: {e}")
        return None


def _passes_quality_gate(quote: dict, ticker: str, strike: float,
                         option_type: str) -> bool:
    """
    v2 Quality Gate — reject options that are structurally unprofitable.

    Checks:
      1. MIN_OPTION_PRICE: no penny options (eliminates 50%+ spread drag)
      2. MIN_OPTION_VOLUME: ensures we can actually exit the position
      3. MAX_BID_ASK_SPREAD_PCT: wide spreads eat profit on entry AND exit
    """
    if quote["mid"] < MIN_OPTION_PRICE:
        log.warning(f"  ❌ QUALITY GATE: {ticker} ${strike} {option_type}: "
                    f"mid ${quote['mid']:.2f} < ${MIN_OPTION_PRICE:.2f} floor")
        return False

    if quote["volume"] < MIN_OPTION_VOLUME:
        log.warning(f"  ❌ QUALITY GATE: {ticker} ${strike} {option_type}: "
                    f"volume {quote['volume']} < {MIN_OPTION_VOLUME} minimum")
        return False

    if quote["spread_pct"] > MAX_BID_ASK_SPREAD_PCT:
        log.warning(f"  ❌ QUALITY GATE: {ticker} ${strike} {option_type}: "
                    f"spread {quote['spread_pct']:.0%} > {MAX_BID_ASK_SPREAD_PCT:.0%} max")
        return False

    return True


def open_position(state: HATState, ticker: str, strike: float,
                  expiry: str, option_type: str, max_cost: float,
                  dry_run: bool = False) -> bool:
    """Open a single-leg option position with v2 quality gates."""
    key = f"{ticker}_{strike}_{expiry}_{option_type}"

    # Already open?
    if key in state.open_positions:
        log.info(f"  ⚠️  {key} already open — skipping")
        return False

    # Max positions check
    if state.n_open >= MAX_POSITIONS:
        log.info(f"  ⚠️  Max positions ({MAX_POSITIONS}) reached — skipping")
        return False

    # Fetch fresh quote
    quote = fetch_option_quote(ticker, strike, expiry, option_type)
    if not quote:
        log.error(f"  ❌ No quote for {ticker} ${strike} {option_type} {expiry}")
        return False

    # ─── v2 Quality Gate ──────────────────────────────────────
    if not _passes_quality_gate(quote, ticker, strike, option_type):
        return False

    price = quote["mid"]
    cost = price * 100 + 0.65  # 1 contract + commission

    # ─── Position cost bounds [$50, $60] ─────────────────────
    if cost < MIN_POSITION_COST:
        log.warning(f"  ⚠️  {ticker} ${strike} {option_type}: "
                    f"${cost:.2f} < ${MIN_POSITION_COST:.2f} minimum — skipping")
        return False

    if cost > max_cost:
        log.warning(f"  ⚠️  {ticker} ${strike} {option_type}: "
                    f"${cost:.2f} > ${max_cost:.2f} budget — skipping")
        return False

    if cost > state.available_cash:
        log.warning(f"  ⚠️  {ticker}: ${cost:.2f} > ${state.available_cash:.2f} cash — skipping")
        return False

    log.info(f"  📈 OPENING {ticker} ${strike} {option_type} exp {expiry}")
    log.info(f"     Price: ${price:.2f} (bid ${quote['bid']:.2f} / "
             f"ask ${quote['ask']:.2f}) | Spread: {quote['spread_pct']:.1%} | "
             f"Vol: {quote['volume']} | Cost: ${cost:.2f}")

    if dry_run:
        log.info(f"  🧪 DRY RUN — recording position")
        state.record_open(key, ticker, strike, expiry, option_type,
                         cost, price, 1)
        state.save()
        return True

    if not ensure_token():
        log.error("  ❌ No token")
        return False

    account_id = get_account_id()
    if not account_id:
        log.error("  ❌ No account ID")
        return False

    result = place_option_order(
        account_id=account_id,
        symbol=ticker,
        strike=f"{strike:.2f}",
        expiry=expiry,
        option_type=option_type,
        side="BUY",
        quantity=1,
        limit_price=f"{price:.2f}",
    )

    if result["success"]:
        log.info(f"  ✅ ORDER PLACED: {ticker} ${strike} {option_type}")
        state.record_open(key, ticker, strike, expiry, option_type,
                         cost, price, 1)
        state.save()
        return True
    else:
        log.error(f"  ❌ Order failed: {result.get('error', 'unknown')}")
        return False


def close_position(state: HATState, key: str, pos: dict,
                   reason: str, dry_run: bool = False) -> bool:
    """Close a single-leg position."""
    ticker = pos["ticker"]
    strike = pos["strike"]
    expiry = pos["expiry"]
    option_type = pos["option_type"]

    quote = fetch_option_quote(ticker, strike, expiry, option_type)
    if not quote:
        log.error(f"  ❌ No quote for closing {key}")
        return False

    exit_price = quote["bid"] if quote["bid"] > 0 else quote["mid"]
    pnl = (exit_price - pos["entry_price"]) * 100  # Per contract

    log.info(f"  📤 CLOSING {key} ({reason})")
    log.info(f"     Entry: ${pos['entry_price']:.2f} → "
             f"Exit: ${exit_price:.2f} | P&L: ${pnl:+.2f}")

    if dry_run:
        state.record_close(key, exit_price, pnl, reason)
        state.save()
        return True

    if not ensure_token():
        return False

    account_id = get_account_id()
    if not account_id:
        return False

    result = place_option_order(
        account_id=account_id,
        symbol=ticker,
        strike=f"{strike:.2f}",
        expiry=expiry,
        option_type=option_type,
        side="SELL",
        quantity=1,
        limit_price=f"{exit_price:.2f}",
    )

    if result["success"]:
        log.info(f"  ✅ CLOSED: {key} | P&L: ${pnl:+.2f} ({reason})")
        state.record_close(key, exit_price, pnl, reason)
        state.save()
        return True
    else:
        log.error(f"  ❌ Close failed: {result.get('error')}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  Hybrid Scanner Integration (v2 — quality-gated)
# ═══════════════════════════════════════════════════════════════════

def run_hybrid_scan(budget_per_position: float, top_n: int = 2):
    """
    Run the hybrid scanner and return top opportunities under budget.

    v2: All results are post-filtered through quality gates
    (min price, min volume, max spread) before being returned.

    Falls back to a quality-filtered manual scan if the hybrid scanner
    returns 0 (which happens when all index confidences are below 70%).
    Returns list of dicts with: ticker, strike, expiry, option_type, est_cost
    """
    log.info(f"\n  🔮 Running Hybrid Scanner v2 (${budget_per_position:.0f}/position, "
             f"quality-gated)...")

    # Try the full hybrid scanner first
    try:
        from signals.hybrid_scanner import scan_hybrid_opportunities
        results = scan_hybrid_opportunities(
            budget=budget_per_position, top_n=top_n * 2,  # Over-fetch to filter
            verbose=False
        )
        if results:
            opportunities = []
            for r in results:
                # Pre-filter: estimate cost and check bounds
                est_cost = r["total_cost"]
                if est_cost < MIN_POSITION_COST or est_cost > MAX_POSITION_COST:
                    log.info(f"  ⚠️  {r['ticker']}: cost ${est_cost:.2f} "
                             f"outside [{MIN_POSITION_COST}, {MAX_POSITION_COST}] — skip")
                    continue
                # Price floor check (option_price is per share)
                if r.get("option_price", 0) < MIN_OPTION_PRICE:
                    log.info(f"  ⚠️  {r['ticker']}: option price "
                             f"${r.get('option_price', 0):.2f} < ${MIN_OPTION_PRICE} — skip")
                    continue
                opportunities.append({
                    "ticker": r["ticker"],
                    "strike": r["strike"],
                    "expiry": r["expiry"],
                    "option_type": r["option_type"],
                    "est_cost": est_cost,
                    "confidence": r["confidence"],
                    "score": r["score"],
                    "direction": r["direction"],
                    "source": "hybrid_scanner",
                })
                if len(opportunities) >= top_n:
                    break
            if opportunities:
                log.info(f"  ✅ Hybrid scanner: {len(opportunities)} quality-gated "
                         f"opportunities")
                return opportunities
    except Exception as e:
        log.warning(f"  ⚠️  Hybrid scanner error: {e}")

    # Fallback: quality-filtered manual scan
    log.info(f"  ⚠️  No hybrid signals — running quality-filtered scan...")
    return _scan_quality_directional(budget_per_position, top_n)


def _scan_quality_directional(max_cost: float, top_n: int = 2):
    """
    v2 fallback scanner: find quality directional options
    with positive GARCH spread AND passing all quality gates.

    Key differences from v1 _scan_cheap_directional:
      - MIN_OPTION_PRICE floor ($0.30) — no penny options
      - MIN_OPTION_VOLUME (50) — must be liquid
      - OTM range tightened to 3-10% (was 3-25%)
      - Bid-ask spread check
      - Cost must be within [$50, $60] bounds
    """
    import numpy as np
    from data.fetcher import fetch_price_data
    from models.garch_model import GARCHVolatilityModel
    from config import TRADING_DAYS
    from signals.scanner import SCAN_UNIVERSE

    candidates = []

    for sym in SCAN_UNIVERSE:
        try:
            df = fetch_price_data(sym)
            if df is None or len(df) < 252:
                continue

            garch = GARCHVolatilityModel()
            diag = garch.fit(df, verbose=False)
            if diag is None:
                continue

            garch_rv = garch.get_conditional_volatility().iloc[-1]
            hv30 = df['Close'].pct_change().dropna().iloc[-30:].std() * np.sqrt(TRADING_DAYS)
            spread = (garch_rv - hv30) / max(hv30, 0.01)

            if spread <= 0 or garch.dampened:
                continue

            spot = float(df['Close'].iloc[-1])

            # Find quality options near the money
            try:
                tk = yf.Ticker(sym)
                exps = tk.options
                if not exps:
                    continue

                # Find expiry 14-30 days out
                from datetime import timedelta
                target = datetime.now() + timedelta(days=14)
                valid_exps = [e for e in exps
                             if datetime.strptime(e, '%Y-%m-%d') >= target]
                if not valid_exps:
                    continue
                exp = valid_exps[0]

                chain = tk.option_chain(exp)
                for opt_type, opts in [("PUT", chain.puts), ("CALL", chain.calls)]:
                    for _, row in opts.iterrows():
                        strike = float(row['strike'])
                        bid = float(row['bid'])
                        ask = float(row['ask'])
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(row['lastPrice'])
                        vol = int(row['volume']) if row['volume'] > 0 else 0

                        # ─── v2 Quality Gates ────────────────────
                        if mid < MIN_OPTION_PRICE:
                            continue
                        if vol < MIN_OPTION_VOLUME:
                            continue

                        spread_pct = (ask - bid) / mid if mid > 0 and bid > 0 else 1.0
                        if spread_pct > MAX_BID_ASK_SPREAD_PCT:
                            continue

                        cost = mid * 100 + 0.65

                        # Cost must be within position bounds [$50, $60]
                        if cost < MIN_POSITION_COST or cost > max_cost:
                            continue

                        # OTM distance: tightened to 3-10% (was 3-25%)
                        if opt_type == "PUT":
                            otm = (spot - strike) / spot
                            if otm < 0.03 or otm > 0.10:
                                continue
                        else:
                            otm = (strike - spot) / spot
                            if otm < 0.03 or otm > 0.10:
                                continue

                        # Quality-weighted score: GARCH signal × volume × (1 - spread)
                        liquidity_factor = min(1.0, vol / 500)
                        spread_penalty = 1.0 - spread_pct
                        score = spread * liquidity_factor * spread_penalty

                        candidates.append({
                            "ticker": sym,
                            "strike": strike,
                            "expiry": exp,
                            "option_type": opt_type,
                            "est_cost": round(cost, 2),
                            "price": round(mid, 2),
                            "volume": vol,
                            "spread_pct": round(spread_pct, 4),
                            "garch_spread": round(spread, 4),
                            "otm_pct": round(otm * 100, 1),
                            "score": round(score, 4),
                            "source": "quality_scan",
                        })
            except Exception:
                continue
        except Exception:
            continue

    # Sort by quality-weighted score
    candidates.sort(key=lambda x: -x["score"])

    if candidates:
        log.info(f"  ✅ Quality scan found {len(candidates)} candidates, "
                 f"returning top {top_n}")
        for i, c in enumerate(candidates[:top_n]):
            log.info(f"     {i+1}. {c['ticker']} ${c['strike']} "
                     f"{c['option_type']} exp {c['expiry']} | "
                     f"${c['est_cost']:.2f} | vol {c['volume']} | "
                     f"spread {c['spread_pct']:.0%} | "
                     f"GARCH {c['garch_spread']:+.1%}")
    else:
        log.info(f"  ⚠️  No quality candidates found in "
                 f"${MIN_POSITION_COST:.0f}-${max_cost:.0f} range")

    return candidates[:top_n]


# ═══════════════════════════════════════════════════════════════════
#  Core Loop
# ═══════════════════════════════════════════════════════════════════

def monitor_positions(state: HATState, dry_run: bool = False):
    """
    Check all open positions against TP/SL/Trailing Stop thresholds.

    v2 trailing stop logic:
      1. Track high-water mark (best P&L % seen)
      2. Once position hits +TRAILING_STOP_PCT, activate trailing stop
      3. Trailing stop = breakeven (entry price)
      4. If position retreats to breakeven after trailing activated → close
      5. TP and SL still apply as absolute outer bounds
    """
    if not state.open_positions:
        return

    for key in list(state.open_positions.keys()):
        pos = state.open_positions[key]

        quote = fetch_option_quote(
            pos["ticker"], pos["strike"], pos["expiry"], pos["option_type"]
        )
        if not quote:
            continue

        current_price = quote["mid"]
        entry_price = pos["entry_price"]
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Update high-water mark
        hwm = state.high_water_marks.get(key, 0)
        if pnl_pct > hwm:
            state.high_water_marks[key] = pnl_pct
            hwm = pnl_pct

        # Check if trailing stop should activate
        trailing = state.trailing_active.get(key, False)
        if not trailing and hwm >= TRAILING_STOP_PCT:
            state.trailing_active[key] = True
            trailing = True
            log.info(f"  🔒 TRAILING STOP ACTIVATED: {key} "
                     f"(peak +{hwm:.1f}%, stop at breakeven)")

        trailing_tag = " [TRAILING]" if trailing else ""
        log.info(f"  {pos['ticker']} ${pos['strike']} {pos['option_type']}: "
                 f"${current_price:.2f} (entry ${entry_price:.2f}) | "
                 f"P&L: {pnl_pct:+.1f}% | HWM: +{hwm:.1f}%{trailing_tag}")

        # ─── Close conditions (checked in priority order) ─────
        if pnl_pct >= TP_PCT:
            log.info(f"  🎯 TP HIT: {key} at +{pnl_pct:.1f}%")
            close_position(state, key, pos, "take_profit", dry_run)
        elif pnl_pct <= SL_PCT:
            log.info(f"  🛑 SL HIT: {key} at {pnl_pct:.1f}%")
            close_position(state, key, pos, "stop_loss", dry_run)
        elif trailing and pnl_pct <= 0:
            # Trailing stop: position was up +15%+, now retreated to breakeven
            log.info(f"  🔒 TRAILING STOP HIT: {key} at {pnl_pct:+.1f}% "
                     f"(was +{hwm:.1f}%)")
            close_position(state, key, pos, "trailing_stop", dry_run)


def run_hat(initial_budget: float = INITIAL_BUDGET,
            dry_run: bool = False):
    """Main HAT v2 loop."""

    log.info(f"\n{'═'*60}")
    log.info(f"  🎩 HAT v2 — Hybrid Autonomous Trader")
    log.info(f"  Budget: ${initial_budget:.2f} | TP: +{TP_PCT}% | "
             f"SL: {SL_PCT}% | Trail: +{TRAILING_STOP_PCT}%")
    log.info(f"  Position bounds: ${MIN_POSITION_COST:.0f}-${MAX_POSITION_COST:.0f} "
             f"| Max positions: {MAX_POSITIONS}")
    log.info(f"  Quality gates: min price ${MIN_OPTION_PRICE} | "
             f"min vol {MIN_OPTION_VOLUME} | max spread {MAX_BID_ASK_SPREAD_PCT:.0%}")
    log.info(f"  Mode: {'🧪 DRY RUN' if dry_run else '🔴 LIVE TRADING'}")
    log.info(f"  Scan interval: {SCAN_INTERVAL}s ({SCAN_INTERVAL//60} min)")
    log.info(f"{'═'*60}")

    # Initialize fresh state (v2 always starts clean)
    state = HATState(initial_budget)
    state.save()
    state.summary()

    # Wait for market
    wait_for_market_open()

    # Phase 1: Scanner-driven initial positions (replaces hardcoded seeds)
    log.info(f"\n{'─'*50}")
    log.info(f"  🔮 PHASE 1: Scanner-driven initial positions")
    log.info(f"  Budget: ${state.available_cash:.2f} | "
             f"Target: {MAX_POSITIONS} positions")
    log.info(f"{'─'*50}")

    initial_opportunities = run_hybrid_scan(
        budget_per_position=MAX_POSITION_COST,
        top_n=MAX_POSITIONS,
    )

    for opp in initial_opportunities:
        if state.n_open >= MAX_POSITIONS:
            break
        if state.available_cash < MIN_POSITION_COST:
            log.info(f"  ⚠️  Cash ${state.available_cash:.2f} < "
                     f"${MIN_POSITION_COST:.2f} min — done opening")
            break
        open_position(
            state,
            opp["ticker"], opp["strike"], opp["expiry"],
            opp["option_type"], MAX_POSITION_COST,
            dry_run=dry_run,
        )
        time.sleep(2)

    log.info(f"\n  🔮 Initial positions: {state.n_open}/{MAX_POSITIONS}")
    state.summary()

    # Phase 2: Main loop — monitor + periodic scanning
    last_scan_time = time.time()  # Don't re-scan immediately

    log.info(f"\n{'─'*50}")
    log.info(f"  🔄 PHASE 2: Monitor + Scan Loop")
    log.info(f"{'─'*50}")

    while True:
        try:
            now = _get_et_now()

            # Only active during market hours
            if not is_market_hours():
                if now.hour >= 16:
                    log.info(f"\n🔴 Market closed. HAT v2 session complete.")
                    state.summary()

                    # Log final results
                    if state.closed_positions:
                        log.info(f"\n  📋 Closed Trades:")
                        for cp in state.closed_positions:
                            pnl = cp.get('pnl', 0)
                            pnl_pct = cp.get('pnl_pct', 0)
                            log.info(f"     {cp['ticker']} ${cp['strike']} "
                                     f"{cp['option_type']}: "
                                     f"${pnl:+.2f} ({pnl_pct:+.1f}%) "
                                     f"[{cp['reason']}]")

                    # Log open positions (will carry overnight)
                    if state.open_positions:
                        log.info(f"\n  📊 Open Positions (overnight):")
                        for k, p in state.open_positions.items():
                            log.info(f"     {p['ticker']} ${p['strike']} "
                                     f"{p['option_type']} | "
                                     f"cost ${p['cost']:.2f}")
                    return

                time.sleep(60)
                continue

            # ─── Monitor existing positions ───────────────────────
            now_ts = time.time()
            log.info(f"\n--- HAT v2 Monitor @ {now.strftime('%H:%M:%S')} ---")
            monitor_positions(state, dry_run)

            # ─── Periodic scan every 30 minutes ──────────────────
            if now_ts - last_scan_time >= SCAN_INTERVAL:
                last_scan_time = now_ts

                slots_available = MAX_POSITIONS - state.n_open
                if slots_available <= 0:
                    log.info(f"  📊 Max positions ({MAX_POSITIONS}) reached — "
                             f"skipping scan")
                elif state.available_cash < MIN_POSITION_COST:
                    log.info(f"  📊 Cash ${state.available_cash:.2f} < "
                             f"${MIN_POSITION_COST:.2f} min — skipping scan")
                else:
                    opportunities = run_hybrid_scan(
                        budget_per_position=MAX_POSITION_COST,
                        top_n=slots_available,
                    )

                    for opp in opportunities:
                        if state.available_cash < MIN_POSITION_COST:
                            break
                        if state.n_open >= MAX_POSITIONS:
                            break
                        open_position(
                            state,
                            opp["ticker"], opp["strike"], opp["expiry"],
                            opp["option_type"],
                            MAX_POSITION_COST,
                            dry_run=dry_run,
                        )
                        time.sleep(2)

                state.summary()

            time.sleep(MONITOR_INTERVAL)

        except KeyboardInterrupt:
            log.info("\n⏹  HAT v2 stopped by user.")
            state.summary()
            state.save()
            return
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(30)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAT v2 — Hybrid Autonomous Trader"
    )
    parser.add_argument("--budget", type=float, default=INITIAL_BUDGET,
                        help=f"Initial budget (default ${INITIAL_BUDGET:.0f})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without placing real orders")
    parser.add_argument("--tp", type=float, default=TP_PCT,
                        help=f"Take-profit %% (default {TP_PCT}%%)")
    parser.add_argument("--sl", type=float, default=SL_PCT,
                        help=f"Stop-loss %% (default {SL_PCT}%%)")
    parser.add_argument("--scan-interval", type=int, default=SCAN_INTERVAL,
                        help=f"Scan interval in seconds (default {SCAN_INTERVAL})")
    args = parser.parse_args()

    # Apply CLI overrides
    TP_PCT = args.tp
    SL_PCT = args.sl
    SCAN_INTERVAL = args.scan_interval

    run_hat(
        initial_budget=args.budget,
        dry_run=args.dry_run,
    )

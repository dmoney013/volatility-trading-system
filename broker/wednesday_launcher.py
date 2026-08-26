#!/usr/bin/env python3
"""
Wednesday 07/08 — Two-Phase Launcher

Phase 1: Open PYPL $45 straddle at 9:30 AM, monitor with 10% TP / -50% SL
Phase 2: When PYPL closes, launch AT --auto-scan with remaining budget

This is NOT auto_trader.py — it's a standalone orchestrator that
uses AT's functions for execution then chains to AT for Phase 2.
"""
import sys
import os
import time
import subprocess
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.auto_trader import (
    open_straddle, monitor_and_close, fetch_fresh_option_quotes,
    ensure_token, get_account_id, _save_active_trade, _load_active_trade,
)
from broker.auto_close import is_market_hours

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

# Phase 1: PYPL straddle
TICKER = "PYPL"
STRIKE = 45
EXPIRY = "2026-07-24"
BUDGET = 2841.0       # ~10 contracts
TP_PCT = 10.0         # 10% take-profit (of initial cost)
SL_PCT = -50.0        # -50% stop-loss

# Phase 2: AT auto-scan
AT_BUDGET_TOTAL = 3000.0  # Total available
DRY_RUN = False

# Logging
LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", "wednesday_launcher.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("wednesday_launcher")


# ═══════════════════════════════════════════════════════════════════
#  Phase 1: Open and Monitor PYPL
# ═══════════════════════════════════════════════════════════════════

def wait_for_market_open():
    """Wait until 9:30 AM ET."""
    log.info("⏳ Phase 1: Waiting for market open (9:30 AM ET)...")
    while True:
        now = datetime.now()
        # Market open at 9:30 AM local (ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        remaining = (market_open - now).total_seconds()

        if remaining <= 0:
            # Already past 9:30
            if now.hour < 16:  # Still before market close
                break
            else:
                log.error("Market is already closed. Exiting.")
                sys.exit(1)

        if remaining > 60:
            log.info(f"   ⏳ {int(remaining / 60)} minutes until market open...")
            time.sleep(min(60, remaining))
        else:
            log.info(f"   ⏳ {int(remaining)}s until market open...")
            time.sleep(min(5, remaining))

    log.info("🟢 Market open!")


def phase1_open_pypl():
    """Open the PYPL straddle and monitor until closed."""
    log.info(f"\n{'='*60}")
    log.info(f"🔵 PHASE 1: PYPL ${STRIKE} Straddle")
    log.info(f"   Budget: ${BUDGET:.0f} | TP: {TP_PCT}% | SL: {SL_PCT}%")
    log.info(f"{'='*60}")

    # Open the straddle
    trade = open_straddle(
        ticker=TICKER,
        strike=str(STRIKE),
        expiry=EXPIRY,
        budget=BUDGET,
        dry_run=DRY_RUN,
    )

    if not trade:
        log.error("❌ Failed to open PYPL straddle. Skipping to Phase 2.")
        return None

    actual_cost = trade['total_cost']
    log.info(f"✅ PYPL opened: {trade['contracts']}x for ${actual_cost:.2f}")
    log.info(f"   TP target: ${actual_cost * (1 + TP_PCT/100):.2f} "
             f"(+${actual_cost * TP_PCT/100:.2f})")

    # Monitor with 10% TP
    log.info(f"\n👁️  Monitoring PYPL with {TP_PCT}% TP / {abs(SL_PCT)}% SL...")

    monitor_and_close(
        ticker=TICKER,
        strike=str(STRIKE),
        expiry=EXPIRY,
        dry_run=DRY_RUN,
        target_pct=TP_PCT,
    )

    # If we get here, the position was closed
    log.info("✅ Phase 1 complete — PYPL position closed.")
    return trade


# ═══════════════════════════════════════════════════════════════════
#  Phase 2: Launch AT Auto-Scan
# ═══════════════════════════════════════════════════════════════════

def phase2_launch_at(pypl_trade):
    """Calculate returned capital and launch AT in auto-scan mode."""
    # PYPL capital is returned when position closes (±P&L)
    # So Phase 2 gets the full budget back
    remaining = AT_BUDGET_TOTAL

    log.info(f"\n{'='*60}")
    log.info(f"🟢 PHASE 2: AT Auto-Scan")
    if pypl_trade:
        log.info(f"   PYPL closed — capital returned to pool")
    else:
        log.info(f"   PYPL failed — full budget available")
    log.info(f"   Budget for AT: ${remaining:.2f}")
    log.info(f"{'='*60}")

    if remaining < 50:
        log.warning("⚠️  Budget too low for AT auto-scan. Done.")
        return

    # Check if market is still open
    if not is_market_hours():
        log.warning("⚠️  Market is closed. AT auto-scan won't find live prices.")
        log.info("   AT will be launched anyway and wait for next market open.")

    # Launch AT in auto-scan mode as a subprocess
    at_cmd = [
        sys.executable, "-u",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "broker", "auto_trader.py"),
        "--auto-scan",
        "--budget", str(remaining),
        "--trade-file",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "cache", "active_trade_at_phase2.json"),
    ]

    if DRY_RUN:
        at_cmd.append("--dry-run")

    log.info(f"🚀 Launching AT auto-scan with ${remaining:.0f} budget...")
    log.info(f"   Command: {' '.join(at_cmd)}")

    at_log_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cache", "auto_trader_phase2.log"
    )

    with open(at_log_file, 'w') as lf:
        process = subprocess.Popen(
            at_cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    log.info(f"✅ AT launched (PID {process.pid})")
    log.info(f"   Monitor: tail -f {at_log_file}")
    log.info(f"\n🏁 Wednesday launcher complete.")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info(f"\n{'═'*60}")
    log.info(f"  🗓️  WEDNESDAY 07/08 — Two-Phase Launcher")
    log.info(f"  Phase 1: PYPL ${STRIKE} straddle (TP {TP_PCT}%)")
    log.info(f"  Phase 2: AT auto-scan with remaining budget")
    log.info(f"  Mode: {'DRY RUN' if DRY_RUN else '🔴 LIVE'}")
    log.info(f"{'═'*60}")

    # Wait for 9:30 AM
    wait_for_market_open()

    # Phase 1: PYPL
    pypl_trade = phase1_open_pypl()

    # Phase 2: AT auto-scan
    phase2_launch_at(pypl_trade)

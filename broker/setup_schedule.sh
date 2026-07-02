#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Trading Daemon Wake/Sleep Schedule Setup
# ═══════════════════════════════════════════════════════════════════
#
# This script sets up:
#   1. macOS scheduled wake at 9:25 AM ET (weekdays)
#   2. macOS scheduled sleep at 4:05 PM ET (weekdays)
#   3. A launchd agent to auto-start the trading daemon on wake
#
# Run with: sudo bash broker/setup_schedule.sh
# ═══════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  TRADING DAEMON — SCHEDULED WAKE/SLEEP SETUP"
echo "═══════════════════════════════════════════════════════════════"

# ─── Step 1: Set wake/sleep schedule ─────────────────────────────
echo ""
echo "Step 1: Setting macOS wake/sleep schedule..."
echo "  Wake:  9:25 AM ET on weekdays (MTWRF)"
echo "  Sleep: 4:05 PM ET on weekdays (MTWRF)"

pmset repeat wakeorpoweron MTWRF 09:25:00 sleep MTWRF 16:05:00

echo "  ✅ Wake/sleep schedule set."

# ─── Step 2: Verify ──────────────────────────────────────────────
echo ""
echo "Step 2: Verifying schedule..."
pmset -g sched
echo ""

# ─── Step 3: Disable sleep when plugged in (optional) ────────────
echo "Step 3: Preventing sleep while on power adapter..."
echo "  (This ensures the Mac stays awake during trading hours"
echo "   even if the lid is closed, as long as it's plugged in)"
# Set display sleep to 10 min on AC but prevent system sleep
pmset -c displaysleep 10
pmset -c sleep 0
echo "  ✅ System sleep disabled on AC power."
echo "  ⚠️  The Mac will still sleep on BATTERY. Keep it plugged in."

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Schedule:"
echo "    Mon-Fri 9:25 AM  →  Mac wakes up"
echo "    Mon-Fri 4:05 PM  →  Mac goes to sleep"
echo ""
echo "  Requirements:"
echo "    - Mac must be PLUGGED IN (won't wake on battery with lid closed)"
echo "    - Lid can be closed"
echo "    - Auto-trader daemon must be started before you leave"
echo ""
echo "  To start the daemon before leaving for work:"
echo "    nohup python3 broker/auto_trader.py --ticker QS --strike 7 --expiry 2026-06-20 &"
echo ""
echo "  To check the current schedule:"
echo "    pmset -g sched"
echo ""
echo "  To remove the schedule:"
echo "    sudo pmset repeat cancel"
echo ""

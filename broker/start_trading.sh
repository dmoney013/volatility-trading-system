#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Trading Daemon — Start Before Leaving for Work
# ═══════════════════════════════════════════════════════════════════
# 
# RUN THIS BEFORE YOU LEAVE FOR WORK:
#   bash broker/start_trading.sh
#
# What it does:
#   1. Prevents Mac from entering deep sleep/standby (caffeinate)
#   2. Starts the auto_trader daemon with auto-scan
#   3. Daemon waits for 9:30 AM market open
#   4. At open: scans live prices → picks optimal OTM strikes → buys
#   5. Monitors all day → closes at +12% take-profit
#   6. After 4 PM: caffeinate ends, Mac can sleep normally
#
# You can safely close the lid after running this.
# ═══════════════════════════════════════════════════════════════════

set -e

# ─── TRADE PARAMETERS ────────────────────────────────────────────
TICKER="XPEV"              # Fallback ticker (auto-scan overrides this)
# ─────────────────────────────────────────────────────────────────

# ─── Environment ─────────────────────────────────────────────────
export HOME="/Users/devongobay"
export PATH="/opt/miniconda3/bin:/opt/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="/Users/devongobay/qt"

QT_DIR="/Users/devongobay/qt"
LOG_FILE="$QT_DIR/cache/auto_trader_nohup.log"
PYTHON="/opt/miniconda3/bin/python3"

mkdir -p "$QT_DIR/cache"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  TRADING DAEMON — v3 AUTO-SCAN STRANGLE (COMBO ORDER)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Mode:      Auto-scan (scanner picks top 2 at market open)"
echo "  Order:     Atomic combo (both legs fill or neither)"
echo "  Attempts:  Up to 3 ranked opportunities"
echo "  TP/SL:     +8% / -50% (v3)"
echo "  Filters:   IV rank < 50, margin > \$1.00, calibrated GARCH"
echo "  Strategy:  Long strangle"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if already running
if pgrep -f "auto_trader.py" > /dev/null 2>&1; then
    echo "⚠️  auto_trader is already running (PID $(pgrep -f auto_trader.py))."
    echo "   Kill it first with: pkill -f auto_trader.py"
    exit 1
fi

# Step 1: Prevent deep sleep/standby until 4:15 PM
# caffeinate -s = prevent system sleep while on AC power
# We calculate seconds until 4:15 PM ET and use that as timeout
SECONDS_UNTIL_CLOSE=$(python3 -c "
import pytz
from datetime import datetime
et = pytz.timezone('US/Eastern')
now = datetime.now(et)
close = now.replace(hour=16, minute=15, second=0, microsecond=0)
if now >= close:
    close = close.replace(day=close.day+1)
print(max(int((close - now).total_seconds()), 3600))
")

echo "☕ Starting caffeinate (prevents deep sleep for ${SECONDS_UNTIL_CLOSE}s until ~4:15 PM ET)..."
caffeinate -s -t "$SECONDS_UNTIL_CLOSE" &
CAFF_PID=$!
echo "   caffeinate PID: $CAFF_PID"

# Step 2: Start the trading daemon
echo "🚀 Starting auto_trader daemon..."
cd "$QT_DIR"

# ─── CURRENT MODE: Pre-selected targets from night-before scan ──
# Skips full scanner, iterates through ranked tickers directly.
# ─────────────────────────────────────────────────────────────────

nohup "$PYTHON" broker/auto_trader.py \
    --ticker "$TICKER" \
    --auto-scan \
    >> "$LOG_FILE" 2>&1 &

DAEMON_PID=$!
echo "   Daemon PID: $DAEMON_PID"

# Step 3: Verify it started
sleep 3
if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo ""
    echo "✅ DAEMON IS RUNNING"
    echo ""
    echo "   You can now close the lid and leave for work."
    echo "   The daemon will:"
    echo "     • Wait for 9:30 AM market open"
    echo "     • Auto-scan universe for optimal strangle strikes"
    echo "     • Place the trade (top 2 picks, IV rank < 50)"
    echo "     • Monitor and close at +8% TP / -50% SL"
    echo ""
    echo "   When you get home, check results with:"
    echo "     tail -30 cache/auto_trader_nohup.log"
    echo ""
    echo "   To stop the daemon:"
    echo "     pkill -f auto_trader.py && kill $CAFF_PID"
    echo ""
else
    echo ""
    echo "❌ DAEMON CRASHED! Check logs:"
    echo "   tail -20 $LOG_FILE"
    kill "$CAFF_PID" 2>/dev/null
    exit 1
fi

#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Wednesday 6/17 Trade — NCLH Long Straddle × 2 (v3 Model)
# ═══════════════════════════════════════════════════════════════════
#
# RUN BEFORE MARKET OPEN:
#   cd ~/qt && bash broker/start_wednesday.sh
#
# What it does:
#   1. Prevents Mac from entering deep sleep (caffeinate)
#   2. Waits for 9:30 AM ET market open
#   3. Opens 2x NCLH $20 ATM straddle (combo order)
#   4. Monitors with TP +15% / SL -50% (per-position)
#   5. Auto-closes when threshold hit
#
# v3 model selected this trade because:
#   - IV rank: 48 (vol is fairly priced)
#   - Margin: $1.72 past breakeven (93% of premium)
#   - Predicted move: ±17.0% (breakeven needs ±9.25%)
#   - Score: 0.835 (highest in universe)
# ═══════════════════════════════════════════════════════════════════

set -e

# ─── TRADE PARAMETERS ────────────────────────────────────────────
TICKER="NCLH"
STRIKE="20"                # ATM straddle
EXPIRY="2026-07-02"        # 15 days out
BUDGET="750"               # $750 budget → 4 contracts at ~$185 each (~$741 total)
TP_PCT="15.0"              # Take profit at +15%
# SL uses config default: -50%
# ─────────────────────────────────────────────────────────────────

# ─── Environment ─────────────────────────────────────────────────
export HOME="/Users/devongobay"
export PATH="/opt/miniconda3/bin:/opt/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="/Users/devongobay/qt"

QT_DIR="/Users/devongobay/qt"
LOG_FILE="$QT_DIR/cache/auto_trader_wednesday.log"
PYTHON="/opt/miniconda3/bin/python3"

mkdir -p "$QT_DIR/cache"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  WEDNESDAY 6/17 — NCLH LONG STRADDLE × 2 (v3 Model)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Ticker:    $TICKER"
echo "  Strike:    \$$STRIKE ATM straddle"
echo "  Expiry:    $EXPIRY"
echo "  Contracts: 2 (budget \$$BUDGET)"
echo "  TP:        +${TP_PCT}%"
echo "  SL:        -50% (config default)"
echo "  Strategy:  Long straddle (combo order)"
echo "  Model:     GARCH v3 (IV rank 48, margin \$1.72, score 0.835)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if already running
if pgrep -f "auto_trader.py" > /dev/null 2>&1; then
    echo "⚠️  auto_trader is already running (PID $(pgrep -f auto_trader.py))."
    echo "   Kill it first with: pkill -f auto_trader.py"
    exit 1
fi

# Step 1: Prevent deep sleep until 4:15 PM
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

nohup "$PYTHON" broker/auto_trader.py \
    --ticker "$TICKER" \
    --strike "$STRIKE" \
    --expiry "$EXPIRY" \
    --budget "$BUDGET" \
    --target-pct "$TP_PCT" \
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
    echo "     • Open 2x NCLH \$20 straddle (combo order)"
    echo "     • Monitor and close at +${TP_PCT}% TP / -50% SL"
    echo ""
    echo "   Logs:"
    echo "     tail -30 $LOG_FILE"
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

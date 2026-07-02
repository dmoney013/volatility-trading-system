#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Monday 6/16 Trade — PYPL Long Straddle × 2 (v3 Model)
# ═══════════════════════════════════════════════════════════════════
# 
# RUN THIS BEFORE MARKET OPEN:
#   bash broker/start_monday.sh
#
# What it does:
#   1. Prevents Mac from entering deep sleep (caffeinate)
#   2. Waits for 9:30 AM ET market open
#   3. Opens 2x PYPL $42 ATM straddle (combo order)
#   4. Monitors with TP +15% / SL -50% (per-position)
#   5. Auto-closes when threshold hit
#
# v3 model selected this trade because:
#   - IV rank: 37 (vol is cheap)
#   - Margin: $1.34 past breakeven
#   - Predicted move: ±7.6% (breakeven needs ±4.9%)
#   - Calibrated GARCH (1.3x scale, 64% 1σ on unseen data)
# ═══════════════════════════════════════════════════════════════════

set -e

# ─── TRADE PARAMETERS ────────────────────────────────────────────
TICKER="PYPL"
STRIKE="42"               # ATM straddle
EXPIRY="2026-06-26"        # 10 days out (~$208/contract, 2 fit in budget)
BUDGET="500"               # $500 budget → 2 contracts at ~$208 each
TP_PCT="15.0"              # Take profit at +15%
# Stop loss handled by config.py STOP_LOSS_PCT = -50%
# We override TP here; SL uses the config default
# ─────────────────────────────────────────────────────────────────

# ─── Environment ─────────────────────────────────────────────────
export HOME="/Users/devongobay"
export PATH="/opt/miniconda3/bin:/opt/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="/Users/devongobay/qt"

QT_DIR="/Users/devongobay/qt"
LOG_FILE="$QT_DIR/cache/auto_trader_monday.log"
PYTHON="/opt/miniconda3/bin/python3"

mkdir -p "$QT_DIR/cache"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  MONDAY 6/16 — PYPL LONG STRADDLE × 2 (v3 Model)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Ticker:    $TICKER"
echo "  Strike:    \$$STRIKE ATM straddle"
echo "  Expiry:    $EXPIRY"
echo "  Contracts: 2 (budget \$$BUDGET)"
echo "  TP:        +${TP_PCT}%"
echo "  SL:        -50% (config default)"
echo "  Strategy:  Long straddle (combo order)"
echo "  Model:     GARCH v3 (1.3x cal, IV rank 37, margin \$1.34)"
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

# PYPL $42 straddle, 2 contracts, combo order
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
    echo "     • Open 2x PYPL \$42 straddle (combo order)"
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

#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Tuesday 6/23 — 3 Simultaneous Positions (GARCH v4)
# ═══════════════════════════════════════════════════════════════════
#
# RUN BEFORE MARKET OPEN:
#   cd ~/qt && bash broker/start_tuesday.sh
#
# Positions:
#   1. NCLH $20 straddle (MONITOR-ONLY) — close when loss ≤ $70
#   2. HIMS $35 straddle (NEW) — open at 9:35 AM, TP +15%
#   3. HOOD $108C/$103P strangle (NEW) — open at 9:35 AM, TP +15%
#
# Each daemon gets its own trade file + log file.
# ═══════════════════════════════════════════════════════════════════

set -e

# ─── Environment ─────────────────────────────────────────────────
export HOME="/Users/devongobay"
export PATH="/opt/miniconda3/bin:/opt/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="/Users/devongobay/qt"

QT_DIR="/Users/devongobay/qt"
PYTHON="/opt/miniconda3/bin/python3"
CACHE="$QT_DIR/cache"

mkdir -p "$CACHE"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  TUESDAY 6/23 — 3 POSITIONS (GARCH v4)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  [MONITOR] NCLH  $20 straddle      | Close at ≤ -$70 loss"
echo "  [NEW]     HIMS  $35 straddle      | TP +15% / SL -50%"
echo "  [NEW]     HOOD  $108C/$103P strngl | TP +15% / SL -50%"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if already running
if pgrep -f "auto_trader.py" > /dev/null 2>&1; then
    echo "⚠️  auto_trader processes already running:"
    pgrep -af "auto_trader.py"
    echo ""
    echo "   Kill them first with: pkill -f auto_trader.py"
    exit 1
fi

# ─── Prevent deep sleep until 4:15 PM ────────────────────────────
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

echo "☕ Starting caffeinate (prevents deep sleep for ${SECONDS_UNTIL_CLOSE}s)..."
caffeinate -s -t "$SECONDS_UNTIL_CLOSE" &
CAFF_PID=$!
echo "   caffeinate PID: $CAFF_PID"
echo ""

cd "$QT_DIR"

# ═══════════════════════════════════════════════════════════════════
# DAEMON 1: NCLH — Monitor-only, close at ≤ -$70 loss (-9.44%)
# ═══════════════════════════════════════════════════════════════════
# -$70 / $741.30 = -9.44%
echo "🟡 Starting NCLH monitor daemon..."
nohup "$PYTHON" broker/auto_trader.py \
    --ticker NCLH \
    --strike 20 \
    --expiry 2026-07-02 \
    --budget 750 \
    --target-pct -9.44 \
    --monitor-only \
    --trade-file "$CACHE/active_trade_NCLH.json" \
    >> "$CACHE/auto_trader_NCLH.log" 2>&1 &
NCLH_PID=$!
echo "   PID: $NCLH_PID | Close at ≤ -\$70 loss (TP -9.44%)"

# ═══════════════════════════════════════════════════════════════════
# DAEMON 2: HIMS — Open $35 straddle at 9:35 AM, TP +15%
# ═══════════════════════════════════════════════════════════════════
echo "🟢 Starting HIMS straddle daemon..."
nohup "$PYTHON" broker/auto_trader.py \
    --ticker HIMS \
    --strike 35 \
    --expiry 2026-07-10 \
    --budget 600 \
    --target-pct 15.0 \
    --trade-file "$CACHE/active_trade_HIMS.json" \
    >> "$CACHE/auto_trader_HIMS.log" 2>&1 &
HIMS_PID=$!
echo "   PID: $HIMS_PID | $35 straddle, TP +15% / SL -50%"

# ═══════════════════════════════════════════════════════════════════
# DAEMON 3: HOOD — Open $108C/$103P strangle at 9:35 AM, TP +15%
# ═══════════════════════════════════════════════════════════════════
echo "🟢 Starting HOOD strangle daemon..."
nohup "$PYTHON" broker/auto_trader.py \
    --ticker HOOD \
    --call-strike 108 \
    --put-strike 103 \
    --expiry 2026-07-10 \
    --budget 900 \
    --target-pct 15.0 \
    --trade-file "$CACHE/active_trade_HOOD.json" \
    >> "$CACHE/auto_trader_HOOD.log" 2>&1 &
HOOD_PID=$!
echo "   PID: $HOOD_PID | $108C/$103P strangle, TP +15% / SL -50%"

# ─── Verify all started ──────────────────────────────────────────
echo ""
sleep 3
FAILURES=0

for TICKER in NCLH HIMS HOOD; do
    LOG="$CACHE/auto_trader_${TICKER}.log"
    if tail -1 "$LOG" 2>/dev/null | grep -q "ERROR\|Traceback"; then
        echo "❌ $TICKER daemon may have crashed. Check: tail -20 $LOG"
        FAILURES=$((FAILURES + 1))
    else
        echo "✅ $TICKER daemon running"
    fi
done

echo ""
if [ "$FAILURES" -eq 0 ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ALL 3 DAEMONS RUNNING"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  You can close the lid and leave for work."
    echo ""
    echo "  Logs:"
    echo "    tail -30 $CACHE/auto_trader_NCLH.log"
    echo "    tail -30 $CACHE/auto_trader_HIMS.log"
    echo "    tail -30 $CACHE/auto_trader_HOOD.log"
    echo ""
    echo "  To stop all:"
    echo "    pkill -f auto_trader.py && kill $CAFF_PID"
    echo ""
else
    echo "⚠️  $FAILURES daemon(s) may have issues. Check logs above."
fi

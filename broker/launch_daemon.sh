#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Trading Daemon Launcher
# ═══════════════════════════════════════════════════════════════════
# This script is called by launchd at 9:26 AM on weekdays.
# It starts the auto_trader daemon which:
#   1. Waits for market open (9:30 AM)
#   2. Auto-scans for optimal OTM strikes based on live spot price
#   3. Opens the strangle position
#   4. Monitors and auto-closes at +12% take-profit
#
# ONLY EDIT THE TICKER — strikes are picked dynamically.
# ═══════════════════════════════════════════════════════════════════

# ─── TRADE PARAMETERS ────────────────────────────────────────────
TICKER="QUBT"              # Edit this to change the target ticker
# ─────────────────────────────────────────────────────────────────

# ─── Environment Setup (CRITICAL for launchd) ────────────────────
# launchd runs with a minimal environment — no PATH, no conda, etc.
export HOME="/Users/devongobay"
export PATH="/opt/miniconda3/bin:/opt/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="/Users/devongobay/qt"

QT_DIR="/Users/devongobay/qt"
LOG_FILE="$QT_DIR/cache/daemon_launcher.log"
NOHUP_LOG="$QT_DIR/cache/auto_trader_nohup.log"
PYTHON="/opt/miniconda3/bin/python3"

# Ensure cache dir exists
mkdir -p "$QT_DIR/cache"

echo "" >> "$LOG_FILE"
echo "$(date): ═══════════════════════════════════════════" >> "$LOG_FILE"
echo "$(date): Daemon launcher started" >> "$LOG_FILE"
echo "  Ticker: $TICKER | Mode: auto-scan strangle" >> "$LOG_FILE"
echo "  Python: $PYTHON" >> "$LOG_FILE"
echo "  PATH: $PATH" >> "$LOG_FILE"

# Verify Python is accessible
if [ ! -x "$PYTHON" ]; then
    echo "$(date): ERROR — Python not found at $PYTHON" >> "$LOG_FILE"
    exit 1
fi

# Check if auto_trader is already running
if pgrep -f "auto_trader.py" > /dev/null 2>&1; then
    echo "$(date): auto_trader already running, skipping." >> "$LOG_FILE"
    exit 0
fi

# Start the daemon with auto-scan (picks optimal strikes at market open)
cd "$QT_DIR"
echo "$(date): Starting auto_trader.py..." >> "$LOG_FILE"
nohup "$PYTHON" broker/auto_trader.py \
    --ticker "$TICKER" \
    --auto-scan \
    >> "$NOHUP_LOG" 2>&1 &

DAEMON_PID=$!
echo "$(date): Daemon started with PID $DAEMON_PID (auto-scan mode)" >> "$LOG_FILE"

# Wait 3 seconds and check if it's still alive
sleep 3
if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "$(date): ✅ Daemon PID $DAEMON_PID is running." >> "$LOG_FILE"
else
    echo "$(date): ❌ Daemon PID $DAEMON_PID DIED within 3 seconds!" >> "$LOG_FILE"
    echo "  Last 10 lines of nohup log:" >> "$LOG_FILE"
    tail -10 "$NOHUP_LOG" >> "$LOG_FILE" 2>&1
fi

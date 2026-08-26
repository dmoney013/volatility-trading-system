#!/bin/bash
# AT Pipeline — Monitor PYPL (15% TP) then auto-scan with $3000
QT_DIR="/Users/devongobay/qt"
cd "$QT_DIR"

echo "═══ AT Pipeline — $(date) ═══"
echo "Phase 1: Monitor PYPL \$45 straddle (15% TP)"
echo "Phase 2: Auto-scan with \$3000 budget"

# Phase 1: Monitor PYPL until it closes
echo "[$(date)] Phase 1: Starting PYPL monitor..."
python3 -u "$QT_DIR/broker/auto_trader.py" \
  --ticker PYPL \
  --strike 45 \
  --expiry 2026-07-24 \
  --monitor-only \
  --target-pct 15 \
  --budget 3000

echo ""
echo "[$(date)] Phase 1 complete — PYPL closed."
echo "[$(date)] Phase 2: Launching AT auto-scan with \$3000..."

# Phase 2: Full auto-scan (straddle + strangle scanners with rigorous filters)
python3 -u "$QT_DIR/broker/auto_trader.py" \
  --auto-scan \
  --budget 3000 \
  --target-pct 15 \
  --trade-file "$QT_DIR/cache/active_trade_phase2.json"

echo ""
echo "[$(date)] AT Pipeline complete."

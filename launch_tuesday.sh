#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Tuesday 07/07 — AT Launch Script (LIVE)
#  Portfolio: ABNB $148 straddle + PYPL $45 straddle
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"
mkdir -p cache

echo "🔴 LIVE MODE — Tuesday 07/07"
echo "═══════════════════════════════════════════════════"

# ─── Kill any leftover daemons from Monday ──────────────────
pkill -f 'auto_trader.py' 2>/dev/null || true
rm -f cache/active_trade_abnb.json cache/active_trade_pypl.json
sleep 1

# ─── 1. ABNB $148 straddle — $2,170 (2 contracts) ──────────
echo "🚀 Starting ABNB $148 straddle daemon..."
nohup python3 -u broker/auto_trader.py \
    --ticker ABNB \
    --strike 148 \
    --expiry 2026-07-31 \
    --budget 2170 \
    --trade-file cache/active_trade_abnb.json \
    > cache/auto_trader_abnb.log 2>&1 &
echo "   PID: $!"

# ─── 2. PYPL $45 straddle — $482 (1 contract) ──────────────
echo "🚀 Starting PYPL $45 straddle daemon..."
nohup python3 -u broker/auto_trader.py \
    --ticker PYPL \
    --strike 45 \
    --expiry 2026-07-31 \
    --budget 482 \
    --trade-file cache/active_trade_pypl.json \
    > cache/auto_trader_pypl.log 2>&1 &
echo "   PID: $!"

echo ""
echo "═══════════════════════════════════════════════════"
echo "✅ 2 daemons launched. Both will wait for 9:30 AM."
echo ""
echo "Monitor:"
echo "  tail -f cache/auto_trader_abnb.log"
echo "  tail -f cache/auto_trader_pypl.log"
echo "═══════════════════════════════════════════════════"

#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  HAT v2 — Monday 07/21 Launch Script (LIVE)
#
#  Budget:         $120 (hard cap, no coordination with AT)
#  Max positions:  2 (at $50-$60 each)
#  Take-profit:    +30%
#  Stop-loss:      -35%
#  Trailing stop:  +15% → breakeven
#
#  Quality gates:
#    - Min option price: $0.30 (no penny options)
#    - Min volume: 50 contracts
#    - Max bid-ask spread: 20%
#
#  No hardcoded seed positions — all entries via live scanner
#  at market open. Scans every 30 minutes for new opportunities
#  when slots are available.
#
#  ⚠️  LIVE MODE — real orders will be placed on Webull
# ══════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

echo "══════════════════════════════════════════════════════"
echo "  🎩 HAT v2 — Monday 07/21 (LIVE)"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Budget:        \$120"
echo "  Positions:     max 2 × \$50-\$60 each"
echo "  TP/SL/Trail:   +30% / -35% / +15%→BE"
echo "  Quality gates: min price \$0.30, vol≥50, spread≤20%"
echo ""

# Ensure cache directory exists
mkdir -p cache

# Kill any leftover HAT processes
pkill -f 'hat.py' 2>/dev/null || true
sleep 1

# Clear old state file (v2 starts fresh)
rm -f cache/hat_state.json

# Launch HAT v2
echo "🚀 Starting HAT v2 daemon..."
nohup python3 -u broker/hat.py \
    --budget 120 \
    > cache/hat_v2_monday.log 2>&1 &
echo "   PID: $!"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ HAT v2 launched. Will wait for 9:30 AM ET."
echo ""
echo "  Monitor:"
echo "    tail -f cache/hat_v2_monday.log"
echo ""
echo "  Kill:"
echo "    pkill -f 'hat.py'"
echo "══════════════════════════════════════════════════════"

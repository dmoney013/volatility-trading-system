#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Monday 07/07 Market Open — 4 Positions
#
#  AT Straddles (scanner-backed):
#    1. PFE  $24 straddle  ~$931  exp 2026-07-31
#    2. XPEV $13 straddle  ~$951  exp 2026-07-31
#    3. FUBO $10 straddle  ~$882  exp 2026-07-31
#
#  Hybrid Directional (Index-Beta):
#    4. XPEV $13 CALL      ~$93   exp 2026-07-31
#
#  Total estimated capital: ~$2,857
#
#  All daemons wait for 9:30 AM ET, open positions, then monitor
#  for auto-close at +5% TP / -50% SL.
#
#  ⚠️  CURRENTLY IN DRY-RUN MODE — remove --dry-run for live trading
# ══════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════"
echo "  🚀 Launching 4 Auto-Trader Daemons (DRY RUN)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Ensure cache directory exists for trade state files
mkdir -p cache

# ─── Position 1: PFE $24 Straddle ─────────────────────────────
echo "  1. PFE  \$24 straddle  exp 2026-07-31  (~\$931)"
nohup python3 -u broker/auto_trader.py \
    --ticker PFE \
    --strike 24 \
    --expiry 2026-07-31 \
    --budget 1000 \
    --trade-file cache/active_trade_pfe.json \
    --dry-run \
    > cache/auto_trader_pfe.log 2>&1 &
echo "     PID: $!  Log: cache/auto_trader_pfe.log"

# ─── Position 2: XPEV $13 Straddle ────────────────────────────
echo "  2. XPEV \$13 straddle  exp 2026-07-31  (~\$951)"
nohup python3 -u broker/auto_trader.py \
    --ticker XPEV \
    --strike 13 \
    --expiry 2026-07-31 \
    --budget 1000 \
    --trade-file cache/active_trade_xpev_straddle.json \
    --dry-run \
    > cache/auto_trader_xpev_straddle.log 2>&1 &
echo "     PID: $!  Log: cache/auto_trader_xpev_straddle.log"

# ─── Position 3: FUBO $10 Straddle ────────────────────────────
echo "  3. FUBO \$10 straddle  exp 2026-07-31  (~\$882)"
nohup python3 -u broker/auto_trader.py \
    --ticker FUBO \
    --strike 10 \
    --expiry 2026-07-31 \
    --budget 1000 \
    --trade-file cache/active_trade_fubo.json \
    --dry-run \
    > cache/auto_trader_fubo.log 2>&1 &
echo "     PID: $!  Log: cache/auto_trader_fubo.log"

# ─── Position 4: XPEV $13 CALL (Hybrid Directional) ───────────
echo "  4. XPEV \$13 CALL     exp 2026-07-31  (~\$93) [HYBRID]"
nohup python3 -u broker/auto_trader.py \
    --ticker XPEV \
    --strike 13 \
    --expiry 2026-07-31 \
    --budget 150 \
    --directional \
    --option-type CALL \
    --trade-file cache/active_trade_xpev_call.json \
    --dry-run \
    > cache/auto_trader_xpev_call.log 2>&1 &
echo "     PID: $!  Log: cache/auto_trader_xpev_call.log"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ 4 daemons launched. They will wait for 9:30 AM ET."
echo ""
echo "  Monitor logs:"
echo "    tail -f cache/auto_trader_pfe.log"
echo "    tail -f cache/auto_trader_xpev_straddle.log"
echo "    tail -f cache/auto_trader_fubo.log"
echo "    tail -f cache/auto_trader_xpev_call.log"
echo ""
echo "  Kill all:"
echo "    pkill -f 'auto_trader.py'"
echo "═══════════════════════════════════════════════════════════"

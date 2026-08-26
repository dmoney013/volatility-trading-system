"""
Futures Directional Scanner — generates directional signals on futures contracts
using existing Index LSTM ensembles.

Flow per contract:
  1. Fetch fresh ETF data (SPY, QQQ, etc.) — LSTM was trained on this
  2. Fetch fresh futures data (ES=F, NQ=F, etc.) — for GARCH and price quotes
  3. Fit GARCH on BOTH ETF and futures for dual confirmation
  4. Compute features from ETF data → run Index LSTM ensemble
  5. If confidence >= threshold AND GARCH spread > 0 → emit signal
  6. No β-translation — the futures contract IS the index

Entirely separate from the equity hybrid scanner.
Signal-only — no auto-execution.

Usage:
    python3 -m futures.scanner                  # Scan all contracts
    python3 -m futures.scanner --contracts SPY  # Scan specific
    python3 -m futures.scanner --micro          # Show micro contract prices
"""
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures.config import (
    ALL_FUTURES, FUTURES_UNIVERSE, CRYPTO_FUTURES,
    CONFIDENCE_THRESHOLD, MIN_GARCH_SPREAD, PREDICTION_HORIZON
)
from futures.data import (
    fetch_etf_prices, fetch_futures_prices, fetch_vix, get_latest_quotes
)

# Read-only imports from equity system (no mutation)
from models.garch_model import GARCHVolatilityModel
from models.lstm_model import (
    EnsembleLSTM, compute_features, normalize_features, NUM_HYBRID_FEATURES
)
from config import DEVICE, TRADING_DAYS, DMT_LSTM_SEQ_LEN


# ═══════════════════════════════════════════════════════════════════
#  LSTM + Normalization Loading
# ═══════════════════════════════════════════════════════════════════

WEIGHTS_DIR = Path("cache/index_lstm_weights")


def _load_ensemble(etf_symbol: str, horizon: int) -> Optional[EnsembleLSTM]:
    """Load a pre-trained Index LSTM ensemble for the given ETF."""
    model_dir = WEIGHTS_DIR / f"{etf_symbol}_{horizon}d_ensemble"
    if not model_dir.exists():
        return None
    try:
        return EnsembleLSTM.load(str(model_dir), input_dim=NUM_HYBRID_FEATURES)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ❌ Could not load {etf_symbol} ensemble: {e}")
        return None


def _load_norm_stats(etf_symbol: str, horizon: int):
    """Load normalization stats (mean/std) from training."""
    stats_file = WEIGHTS_DIR / f"{etf_symbol}_{horizon}d_ensemble" / "stats.npz"
    if stats_file.exists():
        stats = np.load(stats_file)
        return stats['mean'], stats['std']
    return None, None


# ═══════════════════════════════════════════════════════════════════
#  Core Scanner
# ═══════════════════════════════════════════════════════════════════

def scan_futures_signals(
    contracts: Optional[List[str]] = None,
    horizon: int = PREDICTION_HORIZON,
    show_micro: bool = False,
    verbose: bool = True,
) -> List[Dict]:
    """
    Scan all futures contracts for directional signals.

    Args:
        contracts: List of ETF symbols to scan (e.g., ['SPY', 'QQQ']).
                   Defaults to all futures in the universe.
        horizon: Prediction horizon in trading days (default 5).
        show_micro: If True, include micro contract pricing.
        verbose: Print detailed output.

    Returns:
        List of signal dicts, sorted by score (descending).
    """
    universe = contracts or list(ALL_FUTURES.keys())

    if verbose:
        print(f"\n{'='*70}")
        print(f"  🔮 FUTURES DIRECTIONAL SCANNER")
        print(f"  Horizon: {horizon}d | Confidence threshold: "
              f"{CONFIDENCE_THRESHOLD:.0%}")
        print(f"  Contracts: {', '.join(universe)}")
        print(f"{'='*70}\n")

    # ─── Fetch VIX once ───────────────────────────────────────────
    vix = fetch_vix()

    signals = []

    for etf_sym in universe:
        spec = ALL_FUTURES[etf_sym]

        if verbose:
            print(f"\n{'─'*50}")
            print(f"  📋 {spec['name']} ({etf_sym} → {spec['future']})")
            print(f"{'─'*50}")

        # ─── 1. Load LSTM ensemble ────────────────────────────────
        ensemble = _load_ensemble(etf_sym, horizon)
        if ensemble is None:
            if verbose:
                print(f"  ⚠️  No trained LSTM for {etf_sym} — skipping")
            continue

        mean, std = _load_norm_stats(etf_sym, horizon)
        if mean is None:
            if verbose:
                print(f"  ⚠️  No normalization stats for {etf_sym} — skipping")
            continue

        # ─── 2. Fetch FRESH ETF data (for LSTM features) ─────────
        etf_prices = fetch_etf_prices(etf_sym, years=4)
        if etf_prices is None or len(etf_prices) < 300:
            if verbose:
                print(f"  ❌ Insufficient ETF data")
            continue

        # ─── 3. Fetch FRESH futures data (for GARCH + quotes) ────
        fut_prices = fetch_futures_prices(etf_sym, years=4)

        # ─── 4. Fit GARCH on ETF data ────────────────────────────
        garch = GARCHVolatilityModel()
        diag = garch.fit(etf_prices, verbose=False)
        if diag is None:
            if verbose:
                print(f"  ❌ GARCH fit failed on {etf_sym}")
            continue

        garch_rv = garch.get_conditional_volatility().iloc[-1]
        hv30 = (etf_prices['Close'].pct_change().dropna()
                .iloc[-30:].std() * np.sqrt(TRADING_DAYS))
        garch_spread = (garch_rv - hv30) / max(hv30, 0.01)

        if verbose:
            print(f"  GARCH: RV={garch_rv:.1%} | 30dHV={hv30:.1%} | "
                  f"Spread={garch_spread:+.1%} | "
                  f"Persist={garch.persistence:.3f} | "
                  f"Dampened={'Y' if garch.dampened else 'N'}")

        # Also fit GARCH on futures for dual confirmation
        fut_garch_spread = None
        if fut_prices is not None and len(fut_prices) >= 252:
            try:
                fut_garch = GARCHVolatilityModel()
                fut_garch.fit(fut_prices, verbose=False)
                fut_rv = fut_garch.get_conditional_volatility().iloc[-1]
                fut_hv30 = (fut_prices['Close'].pct_change().dropna()
                           .iloc[-30:].std() * np.sqrt(TRADING_DAYS))
                fut_garch_spread = (fut_rv - fut_hv30) / max(fut_hv30, 0.01)
                if verbose:
                    print(f"  GARCH (futures): RV={fut_rv:.1%} | "
                          f"Spread={fut_garch_spread:+.1%}")
            except Exception:
                pass

        # ─── 5. Compute ETF features for LSTM inference ──────────
        try:
            # Get GARCH components for feature embedding
            persistence = garch.persistence
            dampened = garch.dampened
            log_returns = np.log(etf_prices['Close'] /
                                 etf_prices['Close'].shift(1))
            cond_vol = log_returns.rolling(21).std() * np.sqrt(252)

            features_df = compute_features(
                etf_prices, vix=vix,
                garch_persistence=persistence,
                garch_spread=garch_spread,
                garch_dampened=dampened,
                garch_cond_vol=cond_vol
            )

            if features_df is None or len(features_df) < DMT_LSTM_SEQ_LEN:
                if verbose:
                    print(f"  ❌ Insufficient features computed")
                continue

        except Exception as e:
            if verbose:
                print(f"  ❌ Feature computation failed: {e}")
            continue

        # ─── 6. Run LSTM inference ────────────────────────────────
        feat_np = features_df.values.astype(np.float32)
        feat_norm, _, _ = normalize_features(feat_np, mean=mean, std=std)
        seq = feat_norm[-DMT_LSTM_SEQ_LEN:]
        x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        direction, confidence = ensemble.predict_direction(
            x, confidence_threshold=CONFIDENCE_THRESHOLD
        )

        if verbose:
            print(f"  LSTM: Direction={direction} | Confidence={confidence:.1%}")

        # ─── 7. Apply signal filters ─────────────────────────────
        passed = True
        reject_reason = None

        if confidence < CONFIDENCE_THRESHOLD:
            passed = False
            reject_reason = (f"confidence {confidence:.1%} < "
                            f"{CONFIDENCE_THRESHOLD:.0%}")

        elif garch_spread <= MIN_GARCH_SPREAD:
            passed = False
            reject_reason = f"GARCH spread {garch_spread:+.1%} <= 0"

        if not passed:
            if verbose:
                print(f"  ❌ REJECTED: {reject_reason}")
            continue

        # ─── 8. Build signal ─────────────────────────────────────
        score = confidence * max(garch_spread, 0)

        # Get latest price quotes
        etf_spot = float(etf_prices['Close'].iloc[-1])
        fut_spot = None
        micro_spot = None
        basis_pct = None

        if fut_prices is not None and len(fut_prices) > 0:
            fut_spot = float(fut_prices['Close'].iloc[-1])
            basis_pct = round((fut_spot - etf_spot) / etf_spot * 100, 3)

        # Calculate expected P&L per contract
        predicted_move_pct = garch_rv * np.sqrt(horizon / TRADING_DAYS)
        predicted_move_pts = (fut_spot or etf_spot) * predicted_move_pct

        signal = {
            'etf_symbol': etf_sym,
            'futures_symbol': spec['future'],
            'micro_symbol': spec['micro'],
            'name': spec['name'],
            'direction': direction,
            'confidence': round(confidence, 4),
            'garch_spread': round(garch_spread, 4),
            'garch_rv': round(garch_rv, 4),
            'hv30': round(hv30, 4),
            'persistence': round(garch.persistence, 4),
            'dampened': garch.dampened,
            'fut_garch_spread': round(fut_garch_spread, 4) if fut_garch_spread else None,
            'score': round(score, 4),
            'etf_price': round(etf_spot, 2),
            'futures_price': round(fut_spot, 2) if fut_spot else None,
            'basis_pct': basis_pct,
            'predicted_move_pct': round(predicted_move_pct * 100, 2),
            'predicted_move_pts': round(predicted_move_pts, 2),
            'point_value': spec['point_value'],
            'micro_pv': spec['micro_pv'],
            'margin': spec['margin'],
            'micro_margin': spec['micro_margin'],
            'pnl_per_contract': round(predicted_move_pts * spec['point_value'], 2),
            'pnl_per_micro': round(predicted_move_pts * spec['micro_pv'], 2),
            'timestamp': datetime.now().isoformat(),
        }
        signals.append(signal)

        if verbose:
            print(f"  ✅ SIGNAL: {direction} {spec['future']}")
            print(f"     Score: {score:.3f} | Confidence: {confidence:.1%}")
            print(f"     Futures: ${fut_spot:.2f} | Basis: {basis_pct:+.3f}%"
                  if fut_spot else "     Futures: N/A")
            print(f"     Predicted move: {predicted_move_pct*100:.2f}% "
                  f"(±{predicted_move_pts:.1f} pts)")
            print(f"     Est P&L: ${signal['pnl_per_contract']:.0f}/contract | "
                  f"${signal['pnl_per_micro']:.0f}/micro")

    # Sort by score
    signals.sort(key=lambda x: x['score'], reverse=True)

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"  📊 FUTURES SIGNAL SUMMARY — {len(signals)} signals "
              f"(of {len(universe)} scanned)")
        print(f"{'='*70}")
        if not signals:
            print(f"  No actionable signals.")
        for i, s in enumerate(signals):
            arrow = '🟢 LONG' if s['direction'] == 'UP' else '🔴 SHORT'
            print(f"\n  {i+1}. {arrow} {s['futures_symbol']} "
                  f"({s['name']})")
            print(f"     Confidence: {s['confidence']:.1%} | "
                  f"GARCH spread: {s['garch_spread']:+.1%} | "
                  f"Score: {s['score']:.3f}")
            print(f"     Futures: ${s['futures_price']:.2f} | "
                  f"ETF: ${s['etf_price']:.2f} | "
                  f"Basis: {s['basis_pct']:+.3f}%"
                  if s['futures_price'] else
                  f"     ETF: ${s['etf_price']:.2f}")
            print(f"     Predicted {s['predicted_move_pct']:.1f}% move "
                  f"(±{s['predicted_move_pts']:.1f} pts) over {PREDICTION_HORIZON}d")
            print(f"     Est P&L:  ${s['pnl_per_contract']:>8,.0f} /ES contract "
                  f"(${s['margin']:,} margin)")
            print(f"               ${s['pnl_per_micro']:>8,.0f} /micro "
                  f"(${s['micro_margin']:,} margin)")

    return signals


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Futures Directional Scanner — "
                    "LSTM signals on futures contracts"
    )
    parser.add_argument("--contracts", type=str, default=None,
                        help="Comma-separated ETF symbols (e.g., SPY,QQQ)")
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON,
                        help=f"Prediction horizon in days (default {PREDICTION_HORIZON})")
    parser.add_argument("--micro", action="store_true",
                        help="Show micro contract pricing")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override confidence threshold")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    args = parser.parse_args()

    contracts = None
    if args.contracts:
        contracts = [c.strip().upper() for c in args.contracts.split(',')]

    if args.threshold:
        # Module-level override
        import futures.config as fc
        fc.CONFIDENCE_THRESHOLD = args.threshold

    results = scan_futures_signals(
        contracts=contracts,
        horizon=args.horizon,
        show_micro=args.micro,
        verbose=not args.quiet,
    )

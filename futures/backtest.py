"""
Futures Backtest — Walk-forward evaluation of LSTM directional signals
on historical futures data.

For each trading day in the backtest window:
  1. Run LSTM inference using features available up to that day
  2. If signal fires (confidence >= threshold), open a simulated position
  3. Hold for `horizon` trading days, then close
  4. Track P&L using actual futures price changes

Reports: accuracy, Sharpe, max drawdown, win rate, vs random baseline.

Usage:
    python3 -m futures.backtest                         # Backtest all contracts, 1yr
    python3 -m futures.backtest --contracts SPY --years 2
    python3 -m futures.backtest --threshold 0.60        # Looser threshold
"""
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures.config import (
    ALL_FUTURES, CONFIDENCE_THRESHOLD, PREDICTION_HORIZON
)
from futures.data import fetch_etf_prices, fetch_futures_prices, fetch_vix

from models.garch_model import GARCHVolatilityModel
from models.lstm_model import (
    EnsembleLSTM, compute_features, normalize_features, NUM_HYBRID_FEATURES
)
from config import DEVICE, TRADING_DAYS, DMT_LSTM_SEQ_LEN


WEIGHTS_DIR = Path("cache/index_lstm_weights")


def _load_ensemble(etf_symbol: str, horizon: int) -> Optional[EnsembleLSTM]:
    model_dir = WEIGHTS_DIR / f"{etf_symbol}_{horizon}d_ensemble"
    if not model_dir.exists():
        return None
    try:
        return EnsembleLSTM.load(str(model_dir), input_dim=NUM_HYBRID_FEATURES)
    except (FileNotFoundError, RuntimeError):
        return None


def _load_norm_stats(etf_symbol: str, horizon: int):
    stats_file = WEIGHTS_DIR / f"{etf_symbol}_{horizon}d_ensemble" / "stats.npz"
    if stats_file.exists():
        stats = np.load(stats_file)
        return stats['mean'], stats['std']
    return None, None


def backtest_contract(
    etf_symbol: str,
    horizon: int = PREDICTION_HORIZON,
    threshold: float = CONFIDENCE_THRESHOLD,
    backtest_days: int = 252,
    garch_window: int = 252,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Walk-forward backtest of LSTM directional signals on a single contract.

    Args:
        etf_symbol: ETF key (e.g., 'SPY')
        horizon: Holding period in trading days
        threshold: Minimum LSTM confidence to fire
        backtest_days: Number of trading days to backtest
        garch_window: Window for GARCH fitting
        verbose: Print progress

    Returns:
        Dict with backtest metrics, or None on failure.
    """
    spec = ALL_FUTURES[etf_symbol]

    if verbose:
        print(f"\n{'='*65}")
        print(f"  📊 BACKTEST: {spec['name']} ({etf_symbol} → {spec['future']})")
        print(f"  Horizon: {horizon}d | Threshold: {threshold:.0%} | "
              f"Window: {backtest_days}d")
        print(f"{'='*65}")

    # ─── Load LSTM ────────────────────────────────────────────────
    ensemble = _load_ensemble(etf_symbol, horizon)
    if ensemble is None:
        print(f"  ❌ No trained LSTM for {etf_symbol}")
        return None

    mean, std = _load_norm_stats(etf_symbol, horizon)
    if mean is None:
        print(f"  ❌ No normalization stats for {etf_symbol}")
        return None

    # ─── Fetch data ───────────────────────────────────────────────
    # Need extra history for GARCH + features warmup
    total_needed = backtest_days + garch_window + DMT_LSTM_SEQ_LEN + 60
    years_needed = max(4, int(total_needed / 252) + 1)

    etf_prices = fetch_etf_prices(etf_symbol, years=years_needed)
    fut_prices = fetch_futures_prices(etf_symbol, years=years_needed)

    if etf_prices is None or len(etf_prices) < total_needed:
        print(f"  ❌ Insufficient ETF data ({len(etf_prices) if etf_prices is not None else 0} rows, "
              f"need {total_needed})")
        return None

    # Use futures prices for P&L if available, else ETF prices
    pnl_prices = fut_prices if fut_prices is not None and len(fut_prices) > backtest_days else etf_prices

    # Fetch VIX
    vix = fetch_vix(years=years_needed)

    # ─── Walk-forward backtest ────────────────────────────────────
    trades = []
    start_idx = len(etf_prices) - backtest_days

    if verbose:
        print(f"\n  Walking forward {backtest_days} days "
              f"({etf_prices.index[start_idx].strftime('%Y-%m-%d')} → "
              f"{etf_prices.index[-1].strftime('%Y-%m-%d')})...\n")

    for i in range(start_idx, len(etf_prices) - horizon):
        # Slice data available up to day i
        etf_slice = etf_prices.iloc[:i + 1]

        if len(etf_slice) < garch_window + DMT_LSTM_SEQ_LEN + 30:
            continue

        date = etf_prices.index[i]

        # ─── Fit GARCH on available data ──────────────────────────
        try:
            garch = GARCHVolatilityModel()
            garch_data = etf_slice.iloc[-garch_window:]
            diag = garch.fit(garch_data, verbose=False)
            if diag is None:
                continue

            garch_rv = garch.get_conditional_volatility().iloc[-1]
            hv30 = (garch_data['Close'].pct_change().dropna()
                    .iloc[-30:].std() * np.sqrt(TRADING_DAYS))
            garch_spread = (garch_rv - hv30) / max(hv30, 0.01)

            # Skip if GARCH says vol is contracting
            if garch_spread <= 0:
                continue

        except Exception:
            continue

        # ─── Compute features ─────────────────────────────────────
        try:
            persistence = garch.persistence
            dampened = garch.dampened
            log_returns = np.log(etf_slice['Close'] /
                                 etf_slice['Close'].shift(1))
            cond_vol = log_returns.rolling(21).std() * np.sqrt(252)

            vix_slice = vix.loc[:date] if vix is not None else None

            features_df = compute_features(
                etf_slice, vix=vix_slice,
                garch_persistence=persistence,
                garch_spread=garch_spread,
                garch_dampened=dampened,
                garch_cond_vol=cond_vol
            )

            if features_df is None or len(features_df) < DMT_LSTM_SEQ_LEN:
                continue

        except Exception:
            continue

        # ─── LSTM inference ───────────────────────────────────────
        feat_np = features_df.values.astype(np.float32)
        feat_norm, _, _ = normalize_features(feat_np, mean=mean, std=std)
        seq = feat_norm[-DMT_LSTM_SEQ_LEN:]
        x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        direction, confidence = ensemble.predict_direction(
            x, confidence_threshold=threshold
        )

        if confidence < threshold:
            continue

        # ─── Simulate trade ───────────────────────────────────────
        entry_date = etf_prices.index[i]
        exit_idx = min(i + horizon, len(pnl_prices) - 1)
        exit_date = pnl_prices.index[exit_idx]

        entry_price = float(pnl_prices['Close'].iloc[i])
        exit_price = float(pnl_prices['Close'].iloc[exit_idx])

        if direction == 'UP':
            pnl_pct = (exit_price - entry_price) / entry_price
            correct = exit_price > entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
            correct = exit_price < entry_price

        pnl_pts = abs(exit_price - entry_price)
        pnl_dollar = pnl_pts * spec['point_value'] * (1 if correct else -1)
        pnl_micro = pnl_pts * spec['micro_pv'] * (1 if correct else -1)

        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'direction': direction,
            'confidence': round(confidence, 4),
            'garch_spread': round(garch_spread, 4),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'pnl_pct': round(pnl_pct * 100, 3),
            'pnl_dollar': round(pnl_dollar, 2),
            'pnl_micro': round(pnl_micro, 2),
            'correct': correct,
        })

    if not trades:
        if verbose:
            print(f"  ⚠️  No signals fired during backtest period.")
        return None

    # ─── Compute metrics ──────────────────────────────────────────
    df = pd.DataFrame(trades)
    n_trades = len(df)
    n_correct = df['correct'].sum()
    accuracy = n_correct / n_trades
    win_rate = accuracy

    pnl_series = df['pnl_pct'].values
    total_pnl_pct = pnl_series.sum()
    avg_pnl = pnl_series.mean()
    std_pnl = pnl_series.std() if n_trades > 1 else 0
    sharpe = (avg_pnl / std_pnl * np.sqrt(252 / horizon)) if std_pnl > 0 else 0

    # Max drawdown on cumulative P&L
    cum_pnl = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = drawdown.max()

    # Dollar P&L
    total_pnl_dollar = df['pnl_dollar'].sum()
    total_pnl_micro = df['pnl_micro'].sum()

    # Direction breakdown
    n_long = len(df[df['direction'] == 'UP'])
    n_short = len(df[df['direction'] == 'DOWN'])
    long_acc = df[df['direction'] == 'UP']['correct'].mean() if n_long > 0 else 0
    short_acc = df[df['direction'] == 'DOWN']['correct'].mean() if n_short > 0 else 0

    # Random baseline (50/50)
    random_sharpe = 0.0

    metrics = {
        'contract': etf_symbol,
        'futures': spec['future'],
        'name': spec['name'],
        'n_trades': n_trades,
        'n_correct': int(n_correct),
        'accuracy': round(accuracy, 4),
        'win_rate': round(win_rate, 4),
        'total_pnl_pct': round(total_pnl_pct, 3),
        'avg_pnl_pct': round(avg_pnl, 3),
        'sharpe': round(sharpe, 3),
        'max_drawdown_pct': round(max_dd, 3),
        'total_pnl_dollar': round(total_pnl_dollar, 2),
        'total_pnl_micro': round(total_pnl_micro, 2),
        'n_long': n_long,
        'n_short': n_short,
        'long_accuracy': round(long_acc, 4),
        'short_accuracy': round(short_acc, 4),
        'avg_confidence': round(df['confidence'].mean(), 4),
        'avg_garch_spread': round(df['garch_spread'].mean(), 4),
        'trades': trades,
    }

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  RESULTS: {spec['name']}")
        print(f"  {'─'*50}")
        print(f"  Trades:     {n_trades} ({n_long} long, {n_short} short)")
        print(f"  Accuracy:   {accuracy:.1%} ({n_correct}/{n_trades})")
        print(f"    Long:     {long_acc:.1%}" if n_long > 0 else "")
        print(f"    Short:    {short_acc:.1%}" if n_short > 0 else "")
        print(f"  Total P&L:  {total_pnl_pct:+.2f}%")
        print(f"  Avg P&L:    {avg_pnl:+.3f}% per trade")
        print(f"  Sharpe:     {sharpe:.3f}")
        print(f"  Max DD:     {max_dd:.2f}%")
        print(f"  $ P&L:      ${total_pnl_dollar:+,.0f} (full) | "
              f"${total_pnl_micro:+,.0f} (micro)")
        print(f"  Avg conf:   {df['confidence'].mean():.1%}")
        print(f"  vs Random:  Random Sharpe ≈ 0.00 (50/50)")

    return metrics


def backtest_all(
    contracts: Optional[List[str]] = None,
    horizon: int = PREDICTION_HORIZON,
    threshold: float = CONFIDENCE_THRESHOLD,
    backtest_days: int = 252,
    verbose: bool = True,
) -> List[Dict]:
    """Run backtest on all futures contracts."""
    universe = contracts or list(ALL_FUTURES.keys())
    all_results = []

    for etf_sym in universe:
        result = backtest_contract(
            etf_sym,
            horizon=horizon,
            threshold=threshold,
            backtest_days=backtest_days,
            verbose=verbose,
        )
        if result:
            all_results.append(result)

    # Summary table
    if verbose and all_results:
        print(f"\n\n{'='*75}")
        print(f"  📊 BACKTEST SUMMARY — ALL CONTRACTS")
        print(f"{'='*75}")
        print(f"  {'Contract':<8} {'Trades':>7} {'Acc':>6} {'Sharpe':>7} "
              f"{'Tot P&L':>8} {'Max DD':>7} {'$ Full':>10} {'$ Micro':>10}")
        print(f"  {'─'*8} {'─'*7} {'─'*6} {'─'*7} "
              f"{'─'*8} {'─'*7} {'─'*10} {'─'*10}")
        for r in all_results:
            print(f"  {r['contract']:<8} {r['n_trades']:>7} "
                  f"{r['accuracy']:>5.1%} {r['sharpe']:>7.3f} "
                  f"{r['total_pnl_pct']:>+7.2f}% {r['max_drawdown_pct']:>6.2f}% "
                  f"${r['total_pnl_dollar']:>+9,.0f} "
                  f"${r['total_pnl_micro']:>+9,.0f}")

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Futures Backtest — Walk-forward LSTM signal evaluation"
    )
    parser.add_argument("--contracts", type=str, default=None,
                        help="Comma-separated ETF symbols (e.g., SPY,QQQ)")
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--days", type=int, default=252,
                        help="Backtest window in trading days")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    contracts = None
    if args.contracts:
        contracts = [c.strip().upper() for c in args.contracts.split(',')]

    backtest_all(
        contracts=contracts,
        horizon=args.horizon,
        threshold=args.threshold,
        backtest_days=args.days,
        verbose=not args.quiet,
    )

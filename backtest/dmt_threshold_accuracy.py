"""
Hybrid Threshold Accuracy Tester — measures how often daily highs/lows match
or exceed the hybrid ensemble's directional predictions.

Now uses:
  - Binary classification (UP/DOWN only, no FLAT)
  - Ensemble LSTM (averaged softmax from N models)
  - GARCH features embedded in the input vector

Metrics:
  - For UP predictions: How often does max High over horizon rise by >= threshold?
  - For DOWN predictions: How often does min Low over horizon fall by >= threshold?
  - Breakdowns by all-confidence and high-confidence (>=60% and >=70%)
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DMT_HORIZONS
from models.train_lstm import prepare_ticker_data, train_walk_forward


def run_threshold_test(ticker: str, horizon: int = 5, threshold: float = 0.01):
    """
    Measure directional hit rate based on highs/lows using hybrid ensemble.
    """
    # Generate walk-forward predictions (uses ensemble internally)
    wf_results = train_walk_forward(ticker, horizon=horizon, verbose=False)
    if not wf_results:
        return None

    # Load price data
    data_prep = prepare_ticker_data(ticker, horizon=horizon)
    if not data_prep:
        return None
    _, _, _, prices = data_prep  # Updated: 4 return values

    predictions = wf_results['predictions']
    dates = pd.to_datetime(predictions['dates'])
    pred_classes = predictions['predicted']
    confs = predictions['confidence']

    pred_df = pd.DataFrame({
        'predicted': pred_classes,
        'confidence': confs
    }, index=dates[:len(pred_classes)])

    # Reindex prices to match predictions
    prices = prices.loc[pred_df.index[0]:]

    # ─── Count hits ────────────────────────────────────────
    up_signals = 0
    up_hits = 0
    down_signals = 0
    down_hits = 0

    # High confidence (>=60%)
    up_signals_60 = 0
    up_hits_60 = 0
    down_signals_60 = 0
    down_hits_60 = 0

    # Very high confidence (>=70%)
    up_signals_70 = 0
    up_hits_70 = 0
    down_signals_70 = 0
    down_hits_70 = 0

    for i in range(len(pred_df)):
        current_date = pred_df.index[i]
        pred = pred_df.iloc[i]['predicted']  # 0=DOWN, 1=UP (binary)
        conf = pred_df.iloc[i]['confidence']

        # Find future prices over horizon
        future_prices = prices.loc[current_date:].iloc[1:horizon + 1]
        if len(future_prices) < horizon:
            continue

        entry_close = prices.loc[current_date, 'Close']
        max_high = future_prices['High'].max()
        min_low = future_prices['Low'].min()

        if pred == 1:  # UP
            up_signals += 1
            hit = max_high >= entry_close * (1 + threshold)
            if hit:
                up_hits += 1
            if conf >= 0.60:
                up_signals_60 += 1
                if hit:
                    up_hits_60 += 1
            if conf >= 0.70:
                up_signals_70 += 1
                if hit:
                    up_hits_70 += 1

        elif pred == 0:  # DOWN
            down_signals += 1
            hit = min_low <= entry_close * (1 - threshold)
            if hit:
                down_hits += 1
            if conf >= 0.60:
                down_signals_60 += 1
                if hit:
                    down_hits_60 += 1
            if conf >= 0.70:
                down_signals_70 += 1
                if hit:
                    down_hits_70 += 1

    # ─── Summarize ─────────────────────────────────────────
    total = up_signals + down_signals
    total_hits = up_hits + down_hits
    dir_hit_rate = total_hits / total if total > 0 else 0.0

    total_60 = up_signals_60 + down_signals_60
    total_hits_60 = up_hits_60 + down_hits_60
    hit_rate_60 = total_hits_60 / total_60 if total_60 > 0 else 0.0

    total_70 = up_signals_70 + down_signals_70
    total_hits_70 = up_hits_70 + down_hits_70
    hit_rate_70 = total_hits_70 / total_70 if total_70 > 0 else 0.0

    return {
        'ticker': ticker,
        'horizon': horizon,
        'threshold': threshold,
        'up_signals': up_signals,
        'up_hit_rate': up_hits / up_signals if up_signals > 0 else 0.0,
        'down_signals': down_signals,
        'down_hit_rate': down_hits / down_signals if down_signals > 0 else 0.0,
        'directional_hit_rate': dir_hit_rate,
        'total_signals': total,
        # ≥60% confidence
        'signals_60': total_60,
        'hit_rate_60': hit_rate_60,
        # ≥70% confidence
        'signals_70': total_70,
        'hit_rate_70': hit_rate_70,
    }


def run_multi_ticker_test(tickers, horizons=None, thresholds=None):
    """Run threshold test across multiple tickers, horizons, and thresholds."""
    if horizons is None:
        horizons = [5, 10]
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05]

    all_results = []

    for h in horizons:
        for thresh in thresholds:
            print(f"\n{'='*60}")
            print(f"  HORIZON: {h}d | THRESHOLD: {thresh*100:.0f}%")
            print(f"{'='*60}")

            for ticker in tickers:
                print(f"  📊 {ticker}...", end=" ", flush=True)
                res = run_threshold_test(ticker, horizon=h, threshold=thresh)
                if res:
                    all_results.append(res)
                    print(f"DIR: {res['directional_hit_rate']:.1%} "
                          f"(UP: {res['up_hit_rate']:.1%} | "
                          f"DOWN: {res['down_hit_rate']:.1%}) | "
                          f"≥60%: {res['hit_rate_60']:.1%} "
                          f"({res['signals_60']}) | "
                          f"≥70%: {res['hit_rate_70']:.1%} "
                          f"({res['signals_70']})")
                else:
                    print("N/A")

    # ─── Summary table ─────────────────────────────────────
    if all_results:
        print(f"\n\n{'='*80}")
        print("  HYBRID ENSEMBLE THRESHOLD HIT RATE SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Ticker':<8} {'Horizon':<8} {'Thresh':<8} "
              f"{'DirHit%':<10} {'≥60%Hit':<10} {'≥60%#':<8} "
              f"{'≥70%Hit':<10} {'≥70%#':<8}")
        print(f"  {'─'*75}")

        for r in all_results:
            print(f"  {r['ticker']:<8} {r['horizon']}d{'':<5} "
                  f"{r['threshold']*100:.0f}%{'':<5} "
                  f"{r['directional_hit_rate']:<10.1%} "
                  f"{r['hit_rate_60']:<10.1%} "
                  f"{r['signals_60']:<8} "
                  f"{r['hit_rate_70']:<10.1%} "
                  f"{r['signals_70']:<8}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Hybrid ensemble hit rates on highs/lows")
    parser.add_argument("--tickers", "-t", type=str,
                        default="SOFI,HOOD,PLTR,PYPL")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Single horizon (default: test 5d and 10d)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Single threshold (default: 1%%, 3%%, 5%%)")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(',')]
    horizons = [args.horizon] if args.horizon else None
    thresholds = [args.threshold] if args.threshold else None

    run_multi_ticker_test(tickers, horizons=horizons, thresholds=thresholds)

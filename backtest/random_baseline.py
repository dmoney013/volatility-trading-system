"""
Random Baseline Control — proves whether the Hybrid Scanner's LSTM
actually beats random directional predictions.

For each ticker, runs the exact same threshold hit-rate evaluation but with
RANDOMLY SHUFFLED predictions instead of LSTM predictions. If the LSTM
doesn't meaningfully beat this baseline, it has no directional edge.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_price_data


def run_random_baseline(ticker: str, horizon: int = 10,
                        thresholds: list = None, n_trials: int = 100):
    """
    Measure hit rate with random UP/DOWN predictions as a control.

    Runs n_trials of random predictions and averages the hit rates.
    This tells us: "What hit rate do you get just from volatility alone,
    regardless of directional accuracy?"
    """
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05]

    prices = fetch_price_data(ticker)
    if prices is None or len(prices) < 300:
        return None

    # Use the same date range as the walk-forward test (~last 2 years)
    prices = prices.iloc[-504:]  # ~2 years

    results_by_thresh = {}

    for thresh in thresholds:
        trial_rates = []

        for trial in range(n_trials):
            np.random.seed(trial)

            hits = 0
            total = 0

            for i in range(len(prices) - horizon):
                # Random prediction: 0=DOWN, 1=UP (50/50)
                pred = np.random.randint(0, 2)

                entry_close = prices['Close'].iloc[i]
                future = prices.iloc[i + 1:i + 1 + horizon]
                if len(future) < horizon:
                    continue

                max_high = future['High'].max()
                min_low = future['Low'].min()

                if pred == 1:  # Random UP
                    if max_high >= entry_close * (1 + thresh):
                        hits += 1
                elif pred == 0:  # Random DOWN
                    if min_low <= entry_close * (1 - thresh):
                        hits += 1

                total += 1

            if total > 0:
                trial_rates.append(hits / total)

        avg_rate = np.mean(trial_rates) if trial_rates else 0.0
        std_rate = np.std(trial_rates) if trial_rates else 0.0
        results_by_thresh[f"{thresh*100:.0f}%"] = {
            'avg_hit_rate': round(avg_rate, 4),
            'std': round(std_rate, 4),
            'n_trials': n_trials,
        }

    return results_by_thresh


def main():
    tickers = ['SOFI', 'HOOD', 'PLTR', 'PYPL']
    horizons = [5, 10]
    thresholds = [0.01, 0.03, 0.05]

    # Load LSTM results for comparison
    lstm_results_file = 'cache/lstm_results/hybrid_test_20260706_001103.json'
    lstm_data = {}
    if os.path.exists(lstm_results_file):
        with open(lstm_results_file, 'r') as f:
            lstm_data = json.load(f)

    for horizon in horizons:
        print(f"\n{'='*70}")
        print(f"  RANDOM BASELINE vs LSTM — {horizon}d Horizon")
        print(f"{'='*70}")
        print(f"  {'Ticker':<8} {'Thresh':<8} {'Random Hit%':<14} {'LSTM Hit%':<14} {'Edge':<10}")
        print(f"  {'─'*60}")

        for ticker in tickers:
            baseline = run_random_baseline(ticker, horizon=horizon,
                                           thresholds=thresholds, n_trials=50)
            if not baseline:
                continue

            # Get LSTM hit rates from saved results
            h_label = f"{horizon}d"

            for thresh in thresholds:
                t_label = f"{thresh*100:.0f}%"
                rand_rate = baseline[t_label]['avg_hit_rate']

                # Compute LSTM hit rate from saved predictions
                lstm_rate = _compute_lstm_hit_rate(
                    lstm_data, h_label, ticker, horizon, thresh)

                edge = lstm_rate - rand_rate if lstm_rate > 0 else 0

                edge_str = f"{edge:+.1%}" if lstm_rate > 0 else "N/A"
                marker = " ✅" if edge > 0.03 else " ⚠️" if edge > 0 else " ❌"

                print(f"  {ticker:<8} {t_label:<8} {rand_rate:<14.1%} "
                      f"{lstm_rate:<14.1%} {edge_str:<10}{marker}")


def _compute_lstm_hit_rate(lstm_data, h_label, ticker, horizon, threshold):
    """Compute the LSTM directional hit rate from saved walk-forward results."""
    if h_label not in lstm_data or ticker not in lstm_data[h_label]:
        return 0.0

    res = lstm_data[h_label][ticker]
    preds = res['predictions']
    dates = pd.to_datetime(preds['dates'])
    pred_classes = preds['predicted']

    pred_df = pd.DataFrame({
        'predicted': pred_classes,
    }, index=dates[:len(pred_classes)])

    prices = fetch_price_data(ticker)
    if prices is None:
        return 0.0

    prices = prices.loc[pred_df.index[0]:]

    hits = 0
    total = 0

    for i in range(len(pred_df)):
        date = pred_df.index[i]
        pred = pred_df.iloc[i]['predicted']

        future = prices.loc[date:].iloc[1:horizon + 1]
        if len(future) < horizon:
            continue

        entry = prices.loc[date, 'Close']
        max_high = future['High'].max()
        min_low = future['Low'].min()

        if pred == 1:  # UP
            if max_high >= entry * (1 + threshold):
                hits += 1
        else:  # DOWN
            if min_low <= entry * (1 - threshold):
                hits += 1

        total += 1

    return hits / total if total > 0 else 0.0


if __name__ == "__main__":
    main()

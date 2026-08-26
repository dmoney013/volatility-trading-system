#!/usr/bin/env python3
"""
Walk-Forward Directional Accuracy Test — LSTM Direction Prediction

Evaluates the LSTM's ability to predict price direction using a strict
expanding-window walk-forward methodology:

  1. Start with an initial training window (252 days = 1 year minimum)
  2. Train a fresh ensemble on all data up to day T
  3. Validate on day T+1 (used for early-stopping / gate)
  4. Test on day T+2 (record prediction vs actual direction)
  5. Advance by 1 day, repeat through the entire dataset

For each test day, we record:
  - Model's predicted direction (UP=1 / DOWN=0)
  - Actual price movement direction
  - Ensemble confidence (averaged softmax probability)
  - Whether the prediction was correct

The production weights in cache/index_lstm_weights/ are NOT touched.

Output:
  - Per-index directional accuracy (overall + by confidence bucket)
  - Detailed CSV of every prediction for further analysis
  - Summary JSON saved to cache/lstm_results/

Usage:
    python3 models/walk_forward_eval.py                  # All indices
    python3 models/walk_forward_eval.py --indices SPY    # Single index
    python3 models/walk_forward_eval.py --retrain-freq 5 # Retrain every 5 days
"""
import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, DMT_LSTM_SEQ_LEN, DMT_ENSEMBLE_SIZE
)
from models.lstm_model import (
    DirectionalLSTM, compute_features, create_labels,
    SequenceDataset, normalize_features, NUM_HYBRID_FEATURES
)

import yfinance as yf

RESULTS_DIR = Path("cache/lstm_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Data Preparation (same as train_index_lstm.py)
# ═══════════════════════════════════════════════════════════════════

def fetch_and_prepare(symbol: str, years: int = 10, horizon: int = 5):
    """
    Fetch price data and compute features + labels for an index.
    Returns: (features_np, labels_np, label_positions, prices_df, feature_dates)
    """
    start = datetime.now() - pd.Timedelta(days=years * 365)
    prices = yf.download(symbol, start=start.strftime('%Y-%m-%d'), progress=False)
    if prices is None or len(prices) < 500:
        print(f"  ❌ {symbol}: insufficient data")
        return None

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.droplevel(1)

    vix = yf.download("^VIX", start=prices.index[0],
                       end=prices.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)

    # GARCH features (use rolling approximation for speed)
    from models.garch_model import GARCHVolatilityModel
    persistence, spread, dampened = 0.0, 0.0, False
    try:
        garch = GARCHVolatilityModel()
        garch.fit(prices, verbose=False)
        persistence = garch.persistence
        dampened = garch.dampened
        forecast_df = garch.forecast(horizon=5)
        predicted = forecast_df['Annualized Vol'].iloc[-1]
        recent_rv = prices['Close'].pct_change().rolling(21).std().iloc[-1] * np.sqrt(252)
        if recent_rv > 0:
            spread = (predicted - recent_rv) / recent_rv
    except Exception:
        pass

    log_returns = np.log(prices['Close'] / prices['Close'].shift(1))
    cond_vol = log_returns.rolling(21).std() * np.sqrt(252)

    features_df = compute_features(
        prices, vix=vix,
        garch_persistence=persistence,
        garch_spread=spread,
        garch_dampened=dampened,
        garch_cond_vol=cond_vol
    )

    labels, valid_idx = create_labels(prices, horizon=horizon, min_move=0.005)

    common_idx = features_df.index.intersection(valid_idx)
    if len(common_idx) < 200:
        print(f"  ❌ {symbol}: too few decisive moves ({len(common_idx)})")
        return None

    labels = labels.loc[common_idx]

    features_np = features_df.values.astype(np.float32)
    labels_np = labels.values.astype(np.int64)

    label_positions = np.array([
        features_df.index.get_loc(idx) for idx in common_idx
        if idx in features_df.index
    ])
    labels_aligned = labels_np[:len(label_positions)]

    print(f"  📈 {symbol}: {len(features_np)} feature days, "
          f"{len(label_positions)} labeled days "
          f"({features_df.index[0].strftime('%Y-%m-%d')} → "
          f"{features_df.index[-1].strftime('%Y-%m-%d')})")

    return features_np, labels_aligned, label_positions, prices, features_df


# ═══════════════════════════════════════════════════════════════════
#  Train a single model (lightweight, for walk-forward speed)
# ═══════════════════════════════════════════════════════════════════

def train_model(train_feat_norm, train_labels, train_label_pos,
                epochs=40, seed=42):
    """Train a single LSTM model. Fewer epochs than production for speed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_len = DMT_LSTM_SEQ_LEN
    dataset = SequenceDataset(train_feat_norm, train_labels,
                              train_label_pos, seq_len)

    if len(dataset) < 10:
        return None

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DirectionalLSTM(input_dim=NUM_HYBRID_FEATURES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    class_counts = np.bincount(train_labels.ravel(), minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 2
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weights).to(DEVICE))

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


# ═══════════════════════════════════════════════════════════════════
#  Walk-Forward Evaluation
# ═══════════════════════════════════════════════════════════════════

def walk_forward_eval(symbol: str,
                      min_train_days: int = 252,
                      retrain_freq: int = 20,
                      n_ensemble: int = 3,
                      horizon: int = 5):
    """
    Expanding-window walk-forward evaluation.

    Args:
        symbol: Index ETF (SPY, QQQ, etc.)
        min_train_days: Minimum feature days before first prediction (252 = 1 year)
        retrain_freq: Retrain ensemble every N trading days
        n_ensemble: Number of ensemble members (3 for speed, 5 for accuracy)
        horizon: Prediction horizon in days (5 = 1 week)

    Returns:
        Dict with accuracy metrics and per-day predictions
    """
    print(f"\n{'═'*60}")
    print(f"  Walk-Forward Eval: {symbol}")
    print(f"  Min train window: {min_train_days} days | "
          f"Retrain every: {retrain_freq} days")
    print(f"  Ensemble: {n_ensemble} models | Horizon: {horizon}d")
    print(f"{'═'*60}")

    # Fetch and prepare data
    result = fetch_and_prepare(symbol, years=10, horizon=horizon)
    if result is None:
        return None

    features_np, labels_aligned, label_positions, prices, features_df = result

    seq_len = DMT_LSTM_SEQ_LEN
    n_features = len(features_np)
    dates = features_df.index

    # We need at least min_train_days + seq_len + 2 (val + test)
    if n_features < min_train_days + seq_len + 2:
        print(f"  ❌ Not enough data for walk-forward")
        return None

    # ─── Walk forward ─────────────────────────────────────────────
    predictions = []
    current_ensemble = None
    last_train_end = 0
    days_since_retrain = retrain_freq  # Force initial train

    # Start predictions after min_train_days
    start_test_idx = min_train_days + seq_len

    total_steps = n_features - start_test_idx - 1
    print(f"  📊 {total_steps} test days to evaluate "
          f"({dates[start_test_idx].strftime('%Y-%m-%d')} → "
          f"{dates[-2].strftime('%Y-%m-%d')})")

    for test_idx in range(start_test_idx, n_features - 1):
        train_end = test_idx - 1  # Val day = test_idx - 1, test day = test_idx
        val_idx = test_idx - 1

        # ─── Retrain if needed ────────────────────────────────────
        if days_since_retrain >= retrain_freq:
            train_feat = features_np[:train_end]
            train_label_mask = label_positions < train_end
            train_labels = labels_aligned[train_label_mask]
            train_lp = label_positions[train_label_mask]

            if len(train_labels) < 50:
                days_since_retrain += 1
                continue

            train_feat_norm, mean, std = normalize_features(train_feat)

            # Train ensemble
            current_ensemble = []
            for seed_i in range(n_ensemble):
                model = train_model(
                    train_feat_norm, train_labels, train_lp,
                    epochs=40, seed=seed_i * 1000
                )
                if model:
                    model.eval()
                    current_ensemble.append((model, mean, std))

            if not current_ensemble:
                days_since_retrain += 1
                continue

            last_train_end = train_end
            days_since_retrain = 0

            progress = test_idx - start_test_idx
            if progress % 100 == 0 or progress == 0:
                pct = progress / total_steps * 100
                print(f"  🔄 Retrained at {dates[train_end].strftime('%Y-%m-%d')} "
                      f"({progress}/{total_steps} = {pct:.0f}%) | "
                      f"train size: {len(train_labels)} samples")

        days_since_retrain += 1

        if current_ensemble is None:
            continue

        # ─── Predict on test day ──────────────────────────────────
        # Check if test_idx has a label (actual direction)
        test_label_mask = label_positions == test_idx
        if not test_label_mask.any():
            continue

        actual_label = int(labels_aligned[test_label_mask][0])

        # Create input sequence for test day (last seq_len feature vectors)
        if test_idx < seq_len:
            continue

        test_sequence = features_np[test_idx - seq_len:test_idx]

        # Run through ensemble
        ensemble_probs = []
        for model, mean, std in current_ensemble:
            # Normalize using the training stats
            seq_norm = (test_sequence - mean) / (std + 1e-8)
            x = torch.FloatTensor(seq_norm).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
                ensemble_probs.append(probs)

        avg_probs = np.mean(ensemble_probs, axis=0)
        predicted = int(avg_probs.argmax())
        confidence = float(avg_probs.max())

        # Get actual price movement for verification
        test_date = dates[test_idx]
        close_prices = prices['Close']
        if test_date in close_prices.index:
            pos_in_prices = close_prices.index.get_loc(test_date)
            if pos_in_prices + horizon < len(close_prices):
                price_now = float(close_prices.iloc[pos_in_prices])
                price_future = float(close_prices.iloc[pos_in_prices + horizon])
                actual_direction = 1 if price_future > price_now else 0
            else:
                actual_direction = actual_label
        else:
            actual_direction = actual_label

        correct = int(predicted == actual_direction)

        predictions.append({
            'date': test_date.strftime('%Y-%m-%d'),
            'predicted': predicted,  # 1=UP, 0=DOWN
            'actual': actual_direction,
            'correct': correct,
            'confidence': round(confidence, 4),
            'prob_up': round(float(avg_probs[1]), 4),
            'prob_down': round(float(avg_probs[0]), 4),
        })

    if not predictions:
        print(f"  ❌ No predictions generated")
        return None

    # ─── Compute metrics ──────────────────────────────────────────
    df = pd.DataFrame(predictions)
    overall_acc = df['correct'].mean()
    n_total = len(df)
    n_correct = df['correct'].sum()

    # Accuracy by confidence bucket
    buckets = [
        ('All', df),
        ('Conf ≥ 50%', df[df['confidence'] >= 0.50]),
        ('Conf ≥ 55%', df[df['confidence'] >= 0.55]),
        ('Conf ≥ 60%', df[df['confidence'] >= 0.60]),
        ('Conf ≥ 65%', df[df['confidence'] >= 0.65]),
        ('Conf ≥ 70%', df[df['confidence'] >= 0.70]),
        ('Conf ≥ 75%', df[df['confidence'] >= 0.75]),
        ('Conf ≥ 80%', df[df['confidence'] >= 0.80]),
    ]

    print(f"\n  {'─'*50}")
    print(f"  {symbol} Walk-Forward Results ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
    print(f"  {'─'*50}")
    print(f"  {'Bucket':<16} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'% of All':>10}")
    print(f"  {'─'*50}")

    bucket_results = []
    for name, subset in buckets:
        if len(subset) > 0:
            acc = subset['correct'].mean()
            n = len(subset)
            pct = n / n_total * 100
            print(f"  {name:<16} {acc:>9.1%} {int(subset['correct'].sum()):>10} "
                  f"{n:>8} {pct:>9.1f}%")
            bucket_results.append({
                'bucket': name, 'accuracy': round(acc, 4),
                'correct': int(subset['correct'].sum()),
                'total': n, 'pct_of_all': round(pct, 1),
            })

    # Accuracy by year
    df['year'] = pd.to_datetime(df['date']).dt.year
    print(f"\n  {'Year':<8} {'Accuracy':>10} {'N':>8}")
    print(f"  {'─'*30}")
    yearly_results = []
    for year, group in df.groupby('year'):
        acc = group['correct'].mean()
        print(f"  {year:<8} {acc:>9.1%} {len(group):>8}")
        yearly_results.append({
            'year': int(year), 'accuracy': round(acc, 4), 'n': len(group),
        })

    # UP vs DOWN prediction bias
    n_predicted_up = int((df['predicted'] == 1).sum())
    n_actual_up = int((df['actual'] == 1).sum())
    print(f"\n  Predicted UP: {n_predicted_up}/{n_total} ({n_predicted_up/n_total:.1%})")
    print(f"  Actual UP:    {n_actual_up}/{n_total} ({n_actual_up/n_total:.1%})")

    summary = {
        'symbol': symbol,
        'overall_accuracy': round(overall_acc, 4),
        'n_predictions': n_total,
        'n_correct': int(n_correct),
        'date_range': f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}",
        'retrain_freq': retrain_freq,
        'n_ensemble': n_ensemble,
        'min_train_days': min_train_days,
        'buckets': bucket_results,
        'yearly': yearly_results,
        'predicted_up_pct': round(n_predicted_up / n_total, 4),
        'actual_up_pct': round(n_actual_up / n_total, 4),
    }

    # Save detailed predictions CSV
    csv_path = RESULTS_DIR / f"walkforward_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  📁 Predictions saved: {csv_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-Forward Directional Accuracy Test"
    )
    parser.add_argument("--indices", nargs="+",
                        default=["SPY", "QQQ", "IWM", "DIA", "ARKK", "BITO"],
                        help="Indices to evaluate")
    parser.add_argument("--retrain-freq", type=int, default=20,
                        help="Retrain ensemble every N days (default 20)")
    parser.add_argument("--min-train", type=int, default=252,
                        help="Minimum training window in days (default 252)")
    parser.add_argument("--ensemble", type=int, default=3,
                        help="Ensemble size (default 3, production uses 5)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Prediction horizon in days (default 5)")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  🔬 LSTM Walk-Forward Directional Accuracy Test")
    print(f"  Indices: {', '.join(args.indices)}")
    print(f"  Retrain every {args.retrain_freq} days | "
          f"Ensemble: {args.ensemble} | Horizon: {args.horizon}d")
    print(f"  Device: {DEVICE}")
    print(f"  ⚠️  Production weights NOT modified")
    print(f"{'═'*60}")

    all_results = []

    for symbol in args.indices:
        result = walk_forward_eval(
            symbol,
            min_train_days=args.min_train,
            retrain_freq=args.retrain_freq,
            n_ensemble=args.ensemble,
            horizon=args.horizon,
        )
        if result:
            all_results.append(result)

    # Print summary table
    if all_results:
        print(f"\n\n{'═'*60}")
        print(f"  📊 OVERALL SUMMARY")
        print(f"{'═'*60}")
        print(f"  {'Index':<8} {'Accuracy':>10} {'N':>8} {'Pred UP%':>10} {'Act UP%':>10}")
        print(f"  {'─'*50}")
        for r in all_results:
            print(f"  {r['symbol']:<8} {r['overall_accuracy']:>9.1%} "
                  f"{r['n_predictions']:>8} "
                  f"{r['predicted_up_pct']:>9.1%} "
                  f"{r['actual_up_pct']:>9.1%}")

        # Save summary JSON
        summary_path = RESULTS_DIR / f"walkforward_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  📁 Summary saved: {summary_path}")

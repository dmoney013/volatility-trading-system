"""
Index LSTM Training — trains directional prediction models on market indices
instead of individual stocks.

Key insight: Index direction is more predictable than individual stock direction
because diversification removes idiosyncratic noise, leaving cleaner systematic
signals. Beta then translates index predictions to stock-level expectations.

Trains on: SPY, QQQ, IWM, DIA
Features: Same 15-feature LSTM architecture but computed on index data
Target: Binary UP/DOWN direction of the index over the prediction horizon

Usage:
    python3 models/train_index_lstm.py                    # Walk-forward test all indices
    python3 models/train_index_lstm.py --save              # Train and save ensembles
    python3 models/train_index_lstm.py --baseline          # Compare vs random
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
    DEVICE, DMT_LSTM_SEQ_LEN, DMT_ENSEMBLE_SIZE, DMT_HORIZONS
)
from models.lstm_model import (
    DirectionalLSTM, EnsembleLSTM, compute_features, create_labels,
    SequenceDataset, normalize_features, NUM_HYBRID_FEATURES
)
from models.beta_model import INDEX_BENCHMARKS, TRAINABLE_INDICES

import yfinance as yf

WEIGHTS_DIR = Path("cache/index_lstm_weights")
RESULTS_DIR = Path("cache/lstm_results")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Data Preparation for Indices
# ═══════════════════════════════════════════════════════════════════

def fetch_index_data(symbol: str, years: int = 10) -> pd.DataFrame:
    """
    Fetch long-history price data for an index.
    Indices have much longer histories than individual stocks.
    """
    start = datetime.now() - pd.Timedelta(days=years * 365)
    data = yf.download(symbol, start=start.strftime('%Y-%m-%d'),
                       progress=False)
    if data is not None and len(data) > 0:
        # Flatten multi-level columns from newer yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        print(f"  📈 {symbol}: {len(data)} rows "
              f"({data.index[0].strftime('%Y-%m')} → "
              f"{data.index[-1].strftime('%Y-%m')})")
    return data


def prepare_index_data(symbol: str, horizon: int = 5, years: int = 10):
    """
    Fetch index prices, compute features, create binary labels.
    Similar to prepare_ticker_data but for indices with longer history.
    """
    try:
        prices = fetch_index_data(symbol, years=years)
        if prices is None or len(prices) < 500:
            print(f"  ⚠️  {symbol}: insufficient data")
            return None

        # Fetch VIX
        vix = yf.download("^VIX", start=prices.index[0],
                          end=prices.index[-1], progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)

        # GARCH features for the index itself
        from models.garch_model import GARCHVolatilityModel
        persistence = 0.0
        spread = 0.0
        dampened = False

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

        # Binary labels (UP/DOWN, filtering out flat periods)
        labels, valid_idx = create_labels(prices, horizon=horizon,
                                           min_move=0.005)  # 0.5% for indices (less volatile)

        common_idx = features_df.index.intersection(valid_idx)
        if len(common_idx) < 200:
            print(f"  ⚠️  {symbol}: too few decisive moves ({len(common_idx)})")
            return None

        labels = labels.loc[common_idx]
        return features_df, labels, common_idx, prices

    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  Training (reuses train_single_model from train_lstm.py)
# ═══════════════════════════════════════════════════════════════════

def train_single_model(train_feat, train_labels, train_label_indices,
                       epochs=50, lr=1e-3, seed=42):
    """Train a single LSTM model with a specific random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_len = DMT_LSTM_SEQ_LEN
    dataset = SequenceDataset(train_feat, train_labels,
                              train_label_indices, seq_len)

    if len(dataset) < 10:
        return None

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DirectionalLSTM(input_dim=NUM_HYBRID_FEATURES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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


def train_walk_forward_index(symbol: str, horizon: int = 5,
                              train_window: int = 504, test_window: int = 30,
                              n_ensemble: int = DMT_ENSEMBLE_SIZE,
                              verbose: bool = True):
    """
    Walk-forward ensemble training on an index.
    Uses larger train_window (504 = ~2 years) since indices have more data.
    """
    result = prepare_index_data(symbol, horizon=horizon)
    if result is None:
        return None

    features_df, labels, label_idx, prices = result

    features_np = features_df.values.astype(np.float32)
    labels_np = labels.values.astype(np.int64)
    dates = features_df.index

    label_positions = np.array([
        features_df.index.get_loc(idx) for idx in label_idx
        if idx in features_df.index
    ])
    labels_aligned = labels_np[:len(label_positions)]

    n = len(features_np)
    seq_len = DMT_LSTM_SEQ_LEN

    if n < train_window + test_window + seq_len:
        if verbose:
            print(f"  ⚠️  {symbol}: not enough data for walk-forward")
        return None

    all_predictions = []
    all_actuals = []
    all_confidences = []
    all_dates = []
    fold_metrics = []

    fold = 0
    start = 0

    while start + train_window + test_window + seq_len <= n:
        fold += 1
        train_end = start + train_window
        test_end = min(train_end + test_window, n - seq_len)

        train_feat = features_np[start:train_end]
        test_feat = features_np[train_end:test_end + seq_len]

        train_label_mask = (label_positions >= start) & (label_positions < train_end)
        test_label_mask = (label_positions >= train_end) & (label_positions < test_end + seq_len)

        train_labels_fold = labels_aligned[train_label_mask]
        test_labels_fold = labels_aligned[test_label_mask]
        train_label_pos = label_positions[train_label_mask] - start
        test_label_pos = label_positions[test_label_mask] - train_end

        if len(train_labels_fold) < 50 or len(test_labels_fold) < 5:
            start += test_window
            continue

        train_feat_norm, mean, std = normalize_features(train_feat)
        test_feat_norm, _, _ = normalize_features(test_feat, mean=mean, std=std)

        ensemble_probs = []
        for seed_i in range(n_ensemble):
            model = train_single_model(
                train_feat_norm, train_labels_fold, train_label_pos,
                epochs=50, seed=seed_i * 1000 + fold
            )
            if model is None:
                continue

            model.eval()
            test_ds = SequenceDataset(test_feat_norm, test_labels_fold,
                                      test_label_pos, seq_len)
            if len(test_ds) == 0:
                continue

            test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test = x_test.to(DEVICE)
                    probs = torch.softmax(model(x_test), dim=1).cpu().numpy()
                    ensemble_probs.append(probs)

        if not ensemble_probs:
            start += test_window
            continue

        avg_probs = np.mean(ensemble_probs, axis=0)
        preds = avg_probs.argmax(axis=1)
        confs = avg_probs.max(axis=1)
        actuals = y_test.numpy()

        accuracy = (preds == actuals).mean()

        all_predictions.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())
        all_confidences.extend(confs.tolist())

        test_dates_fold = []
        for pos in test_label_pos:
            abs_pos = train_end + pos
            if abs_pos < len(dates):
                test_dates_fold.append(dates[abs_pos].strftime('%Y-%m-%d'))
        all_dates.extend(test_dates_fold[:len(preds)])

        fold_metrics.append({
            'fold': fold, 'accuracy': round(accuracy, 4),
            'n_test': len(preds), 'avg_confidence': round(confs.mean(), 4),
        })

        if verbose:
            print(f"  Fold {fold}: acc={accuracy:.1%} conf={confs.mean():.1%} "
                  f"(train {dates[start].strftime('%Y-%m')} → "
                  f"test {dates[train_end].strftime('%Y-%m')}, n={len(preds)})")

        start += test_window

    if not all_predictions:
        return None

    preds = np.array(all_predictions)
    actuals = np.array(all_actuals)
    confs = np.array(all_confidences)

    overall_acc = (preds == actuals).mean()

    high_conf_mask = confs >= 0.60
    high_conf_acc = (preds[high_conf_mask] == actuals[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0
    high_conf_pct = high_conf_mask.mean()

    very_high_mask = confs >= 0.70
    very_high_acc = (preds[very_high_mask] == actuals[very_high_mask]).mean() if very_high_mask.sum() > 0 else 0
    very_high_pct = very_high_mask.mean()

    return {
        'symbol': symbol,
        'horizon': horizon,
        'overall_accuracy': round(overall_acc, 4),
        'high_conf_accuracy': round(high_conf_acc, 4),
        'high_conf_pct': round(high_conf_pct, 4),
        'very_high_accuracy': round(very_high_acc, 4),
        'very_high_pct': round(very_high_pct, 4),
        'avg_confidence': round(confs.mean(), 4),
        'n_predictions': len(preds),
        'n_folds': len(fold_metrics),
        'fold_metrics': fold_metrics,
        'predictions': {
            'dates': all_dates,
            'predicted': all_predictions,
            'actual': all_actuals,
            'confidence': [round(c, 4) for c in all_confidences],
        },
    }


# ═══════════════════════════════════════════════════════════════════
#  Train & Save Index Ensembles
# ═══════════════════════════════════════════════════════════════════

def train_and_save_index(symbol: str, horizon: int = 5,
                          n_ensemble: int = DMT_ENSEMBLE_SIZE):
    """Train and save ensemble models for an index."""
    result = prepare_index_data(symbol, horizon=horizon, years=10)
    if result is None:
        return None

    features_df, labels, label_idx, prices = result
    features_np = features_df.values.astype(np.float32)
    labels_np = labels.values.astype(np.int64)

    label_positions = np.array([
        features_df.index.get_loc(idx) for idx in label_idx
        if idx in features_df.index
    ])
    labels_aligned = labels_np[:len(label_positions)]

    features_norm, mean, std = normalize_features(features_np)

    ensemble_dir = WEIGHTS_DIR / f"{symbol}_{horizon}d_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_ensemble):
        print(f"  Training model {i+1}/{n_ensemble} (seed={i*1000})...")
        model = train_single_model(
            features_norm, labels_aligned, label_positions,
            epochs=80, seed=i * 1000
        )
        if model:
            torch.save(model.state_dict(), ensemble_dir / f"model_{i}.pt")

    np.savez(ensemble_dir / "stats.npz", mean=mean, std=std)
    print(f"  💾 Saved to {ensemble_dir}")


# ═══════════════════════════════════════════════════════════════════
#  Random Baseline for Indices
# ═══════════════════════════════════════════════════════════════════

def random_baseline_index(symbol: str, horizon: int = 5, n_trials: int = 100):
    """Random baseline for index direction prediction."""
    prices = fetch_index_data(symbol, years=4)
    if prices is None or len(prices) < 500:
        return None

    close = prices['Close']
    returns = close.pct_change(horizon).dropna()

    # What % of the time does the index go UP over the horizon?
    up_pct = float((returns > 0).mean())

    # Random baseline accuracy = max(up_pct, 1-up_pct)
    baseline_acc = max(up_pct, 1 - up_pct)

    return {
        'symbol': symbol,
        'horizon': horizon,
        'up_frequency': round(up_pct, 4),
        'random_baseline': round(baseline_acc, 4),
        'random_50_50': 0.5000,
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train Index LSTM (SPY, QQQ, IWM, DIA)")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices (default: all 4)")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--save", action="store_true",
                        help="Train and save ensembles")
    parser.add_argument("--baseline", action="store_true",
                        help="Run random baseline comparison")
    args = parser.parse_args()

    indices = [i.strip() for i in args.indices.split(',')] if args.indices else TRAINABLE_INDICES

    print(f"🏛️  Index LSTM Training Pipeline")
    print(f"   Indices: {', '.join(indices)}")
    print(f"   Horizon: {args.horizon}d")
    print(f"   Ensemble: {DMT_ENSEMBLE_SIZE} models")
    print(f"   Device: {DEVICE}\n")

    if args.save:
        for symbol in indices:
            print(f"\n📊 Training ensemble for {symbol}...")
            train_and_save_index(symbol, horizon=args.horizon)

    elif args.baseline:
        print(f"\n{'='*60}")
        print(f"  RANDOM BASELINE vs INDEX LSTM — {args.horizon}d Horizon")
        print(f"{'='*60}")
        print(f"  {'Index':<8} {'Random':<10} {'LSTM':<10} {'Edge':<10}")
        print(f"  {'─'*40}")

        for symbol in indices:
            base = random_baseline_index(symbol, horizon=args.horizon)
            result = train_walk_forward_index(symbol, horizon=args.horizon)

            if base and result:
                edge = result['overall_accuracy'] - 0.50  # vs coin flip
                marker = " ✅" if edge > 0.03 else " ⚠️" if edge > 0 else " ❌"
                print(f"  {symbol:<8} {0.50:<10.1%} "
                      f"{result['overall_accuracy']:<10.1%} "
                      f"{edge:+.1%}{marker}")
                print(f"           ≥60%: {result['high_conf_accuracy']:.1%} "
                      f"({result['high_conf_pct']:.0%}) | "
                      f"≥70%: {result['very_high_accuracy']:.1%} "
                      f"({result['very_high_pct']:.0%})")

    else:
        for symbol in indices:
            print(f"\n📊 Walk-forward {symbol} (horizon={args.horizon}d)...")
            result = train_walk_forward_index(symbol, horizon=args.horizon)
            if result:
                print(f"   ✅ Overall: {result['overall_accuracy']:.1%} | "
                      f"≥60%: {result['high_conf_accuracy']:.1%} | "
                      f"≥70%: {result['very_high_accuracy']:.1%}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"index_test_{timestamp}.json"
    print(f"\n  📁 Results: {results_file}")


if __name__ == "__main__":
    main()

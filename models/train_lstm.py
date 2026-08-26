"""
Hybrid LSTM Training Pipeline — ensemble training with GARCH-informed features.

Key differences from original DMT training:
    1. Binary labels (UP/DOWN only, no FLAT) — filters out sideways periods
    2. GARCH features embedded in input vector (persistence, spread, dampened)
    3. Ensemble training: trains N models with different random seeds
    4. Walk-forward validation with ensemble averaging

Usage:
    python3 models/train_lstm.py --tickers SOFI,HOOD,PLTR --ensemble --save
    python3 models/train_lstm.py --test-horizons --ensemble
"""
import argparse
import json
import os
import sys
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, DMT_LSTM_SEQ_LEN, DMT_ENSEMBLE_SIZE,
    DMT_HORIZONS, REJECT_DAMPENED, GARCH_FIT_WINDOW
)
from models.lstm_model import (
    DirectionalLSTM, EnsembleLSTM, compute_features, create_labels,
    SequenceDataset, normalize_features, NUM_HYBRID_FEATURES
)
from data.fetcher import fetch_price_data
from signals.scanner import SCAN_UNIVERSE

# ─── Paths ────────────────────────────────────────────────────────
WEIGHTS_DIR = Path("cache/lstm_weights")
RESULTS_DIR = Path("cache/lstm_results")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Data Preparation (with GARCH features)
# ═══════════════════════════════════════════════════════════════════

def compute_garch_features(prices: pd.DataFrame):
    """
    Compute GARCH features for embedding in LSTM input.
    Returns (persistence, spread, dampened, cond_vol_series).
    Uses rolling realized vol as cond_vol to avoid windowing issues.
    """
    from models.garch_model import GARCHVolatilityModel

    persistence = 0.0
    spread = 0.0
    dampened = False

    try:
        garch = GARCHVolatilityModel()
        garch.fit(prices, verbose=False)
        persistence = garch.persistence
        dampened = garch.dampened

        # Compute spread: (predicted RV - IV) / IV
        # Use predicted vol vs recent realized vol as proxy
        forecast_df = garch.forecast(horizon=5)
        predicted = forecast_df['Annualized Vol'].iloc[-1]
        recent_rv = prices['Close'].pct_change().rolling(21).std().iloc[-1] * np.sqrt(252)
        if recent_rv > 0:
            spread = (predicted - recent_rv) / recent_rv
    except Exception:
        pass

    # Use rolling RV for conditional vol (full history, no windowing issues)
    log_returns = np.log(prices['Close'] / prices['Close'].shift(1))
    cond_vol = log_returns.rolling(21).std() * np.sqrt(252)

    return persistence, spread, dampened, cond_vol


def prepare_ticker_data(ticker: str, horizon: int = 5):
    """
    Fetch prices, compute features with GARCH, create binary labels.
    Returns (features_df, labels, label_indices, prices_df) or None.
    """
    try:
        prices = fetch_price_data(ticker)
        if prices is None or len(prices) < 300:
            print(f"  ⚠️  {ticker}: insufficient data "
                  f"({len(prices) if prices is not None else 0} rows)")
            return None

        # Fetch VIX
        import yfinance as yf
        vix = yf.download("^VIX", start=prices.index[0],
                          end=prices.index[-1], progress=False)

        # Compute GARCH features
        persistence, spread, dampened, cond_vol = compute_garch_features(prices)

        # Compute full feature set (including GARCH)
        features_df = compute_features(
            prices, vix=vix,
            garch_persistence=persistence,
            garch_spread=spread,
            garch_dampened=dampened,
            garch_cond_vol=cond_vol
        )

        # Create binary labels (UP/DOWN only, no FLAT)
        labels, valid_idx = create_labels(prices, horizon=horizon, min_move=0.01)

        # Align features and labels
        common_idx = features_df.index.intersection(valid_idx)
        if len(common_idx) < 100:
            print(f"  ⚠️  {ticker}: too few decisive moves ({len(common_idx)})")
            return None

        labels = labels.loc[common_idx]

        return features_df, labels, common_idx, prices

    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  Walk-Forward Ensemble Training
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

    # Class weights for imbalance
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
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


def train_walk_forward(ticker: str, horizon: int = 5,
                       train_window: int = 252, test_window: int = 30,
                       n_ensemble: int = DMT_ENSEMBLE_SIZE,
                       verbose: bool = True):
    """
    Walk-forward train + test with ensemble averaging.
    """
    result = prepare_ticker_data(ticker, horizon=horizon)
    if result is None:
        return None

    features_df, labels, label_idx, prices = result

    features_np = features_df.values.astype(np.float32)
    labels_np = labels.values.astype(np.int64)
    dates = features_df.index

    # Map label_idx to integer positions within features_df
    label_positions = np.array([
        features_df.index.get_loc(idx) for idx in label_idx
        if idx in features_df.index
    ])
    labels_aligned = labels_np[:len(label_positions)]

    n = len(features_np)
    seq_len = DMT_LSTM_SEQ_LEN

    if n < train_window + test_window + seq_len:
        if verbose:
            print(f"  ⚠️  {ticker}: not enough data for walk-forward "
                  f"({n} < {train_window + test_window + seq_len})")
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

        # Split features
        train_feat = features_np[start:train_end]
        test_feat = features_np[train_end:test_end + seq_len]

        # Get label positions within train/test ranges
        train_label_mask = (label_positions >= start) & (label_positions < train_end)
        test_label_mask = (label_positions >= train_end) & (label_positions < test_end + seq_len)

        train_labels_fold = labels_aligned[train_label_mask]
        test_labels_fold = labels_aligned[test_label_mask]
        train_label_pos = label_positions[train_label_mask] - start
        test_label_pos = label_positions[test_label_mask] - train_end

        if len(train_labels_fold) < 20 or len(test_labels_fold) < 5:
            start += test_window
            continue

        # Normalize using training stats
        train_feat_norm, mean, std = normalize_features(train_feat)
        test_feat_norm, _, _ = normalize_features(test_feat, mean=mean, std=std)

        # Train ensemble
        ensemble_probs = []
        for seed_i in range(n_ensemble):
            model = train_single_model(
                train_feat_norm, train_labels_fold, train_label_pos,
                epochs=50, seed=seed_i * 1000 + fold
            )
            if model is None:
                continue

            # Evaluate on test set
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

        # Average ensemble predictions
        avg_probs = np.mean(ensemble_probs, axis=0)
        preds = avg_probs.argmax(axis=1)
        confs = avg_probs.max(axis=1)

        # Get actuals from last test loader
        actuals = y_test.numpy()

        # Record results
        accuracy = (preds == actuals).mean()
        test_dates_fold = []
        for pos in test_label_pos:
            abs_pos = train_end + pos
            if abs_pos < len(dates):
                test_dates_fold.append(dates[abs_pos].strftime('%Y-%m-%d'))

        all_predictions.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())
        all_confidences.extend(confs.tolist())
        all_dates.extend(test_dates_fold[:len(preds)])

        fold_metrics.append({
            'fold': fold,
            'train_start': dates[start].strftime('%Y-%m-%d'),
            'test_start': dates[train_end].strftime('%Y-%m-%d'),
            'accuracy': round(accuracy, 4),
            'n_test': len(preds),
            'avg_confidence': round(confs.mean(), 4),
        })

        if verbose:
            print(f"  Fold {fold}: acc={accuracy:.1%} conf={confs.mean():.1%} "
                  f"(train {dates[start].strftime('%Y-%m')} → "
                  f"test {dates[train_end].strftime('%Y-%m')}, n={len(preds)})")

        start += test_window

    if not all_predictions:
        return None

    # ─── Aggregate results ─────────────────────────────────
    preds = np.array(all_predictions)
    actuals = np.array(all_actuals)
    confs = np.array(all_confidences)

    overall_acc = (preds == actuals).mean()

    # High-confidence accuracy
    high_conf_mask = confs >= 0.60
    if high_conf_mask.sum() > 0:
        high_conf_acc = (preds[high_conf_mask] == actuals[high_conf_mask]).mean()
        high_conf_pct = high_conf_mask.mean()
    else:
        high_conf_acc = 0.0
        high_conf_pct = 0.0

    # Very high confidence (≥70%)
    very_high_mask = confs >= 0.70
    if very_high_mask.sum() > 0:
        very_high_acc = (preds[very_high_mask] == actuals[very_high_mask]).mean()
        very_high_pct = very_high_mask.mean()
    else:
        very_high_acc = 0.0
        very_high_pct = 0.0

    return {
        'ticker': ticker,
        'horizon': horizon,
        'overall_accuracy': round(overall_acc, 4),
        'high_conf_accuracy': round(high_conf_acc, 4),
        'high_conf_pct': round(high_conf_pct, 4),
        'very_high_conf_accuracy': round(very_high_acc, 4),
        'very_high_conf_pct': round(very_high_pct, 4),
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
#  Multi-Horizon Accuracy Test
# ═══════════════════════════════════════════════════════════════════

def test_horizons(tickers: list, horizons: list = None, verbose: bool = True):
    """Test ensemble accuracy across multiple prediction horizons."""
    if horizons is None:
        horizons = DMT_HORIZONS

    results = {}

    for h in horizons:
        h_label = f"{h}d"
        print(f"\n{'='*60}")
        print(f"  HORIZON: {h} trading day(s) | Ensemble: {DMT_ENSEMBLE_SIZE} models")
        print(f"{'='*60}")
        results[h_label] = {}

        for ticker in tickers:
            print(f"\n  📊 {ticker} (horizon={h}d)...")
            result = train_walk_forward(ticker, horizon=h, verbose=verbose)
            if result:
                results[h_label][ticker] = result
                print(f"     Overall: {result['overall_accuracy']:.1%} | "
                      f"≥60% conf: {result['high_conf_accuracy']:.1%} "
                      f"({result['high_conf_pct']:.0%}) | "
                      f"≥70% conf: {result['very_high_conf_accuracy']:.1%} "
                      f"({result['very_high_conf_pct']:.0%})")

    # ─── Summary table ─────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  HYBRID ENSEMBLE ACCURACY COMPARISON (Binary UP/DOWN)")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8}", end="")
    for h in horizons:
        print(f"  {h}d-Overall  {h}d-≥60%  {h}d-≥70%", end="")
    print()

    for ticker in tickers:
        print(f"  {ticker:<8}", end="")
        for h in horizons:
            h_label = f"{h}d"
            if h_label in results and ticker in results[h_label]:
                r = results[h_label][ticker]
                print(f"  {r['overall_accuracy']:>10.1%}"
                      f"  {r['high_conf_accuracy']:>7.1%}"
                      f"  {r['very_high_conf_accuracy']:>7.1%}", end="")
            else:
                print(f"  {'N/A':>10}  {'N/A':>7}  {'N/A':>7}", end="")
        print()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"hybrid_test_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  📁 Results saved to {results_file}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Train & Save Ensemble
# ═══════════════════════════════════════════════════════════════════

def train_and_save_ensemble(ticker: str, horizon: int = 5,
                            n_ensemble: int = DMT_ENSEMBLE_SIZE,
                            verbose: bool = True):
    """Train and save an ensemble of models for a ticker."""
    result = prepare_ticker_data(ticker, horizon=horizon)
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

    # Normalize
    features_norm, mean, std = normalize_features(features_np)

    # Save dir
    ensemble_dir = WEIGHTS_DIR / f"{ticker}_{horizon}d_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_ensemble):
        if verbose:
            print(f"  Training model {i+1}/{n_ensemble} (seed={i*1000})...")

        model = train_single_model(
            features_norm, labels_aligned, label_positions,
            epochs=80, seed=i * 1000
        )
        if model:
            torch.save(model.state_dict(), ensemble_dir / f"model_{i}.pt")

    # Save normalization stats
    np.savez(ensemble_dir / "stats.npz", mean=mean, std=std)
    if verbose:
        print(f"  💾 Saved ensemble to {ensemble_dir}")


# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid LSTM (GARCH + directional) models")
    parser.add_argument("--tickers", "-t", type=str, default=None,
                        help="Comma-separated tickers")
    parser.add_argument("--all", action="store_true",
                        help="Train on full SCAN_UNIVERSE")
    parser.add_argument("--test-horizons", action="store_true",
                        help="Compare accuracy across horizons")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Prediction horizon (default: 5)")
    parser.add_argument("--save", action="store_true",
                        help="Train final ensemble and save")
    parser.add_argument("--ensemble", action="store_true", default=True,
                        help="Use ensemble (default: True)")
    args = parser.parse_args()

    if args.all:
        tickers = SCAN_UNIVERSE
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        tickers = ['PYPL', 'HOOD', 'MSTR', 'SOFI', 'PLTR']

    print(f"🧠 HYBRID LSTM Training Pipeline (GARCH + Directional)")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Device: {DEVICE}")
    print(f"   Ensemble size: {DMT_ENSEMBLE_SIZE}")
    print(f"   Binary classification: UP vs DOWN (no FLAT)")
    print()

    if args.test_horizons:
        test_horizons(tickers, horizons=DMT_HORIZONS)
    elif args.save:
        for ticker in tickers:
            print(f"\n📊 Training ensemble for {ticker} "
                  f"(horizon={args.horizon}d)...")
            train_and_save_ensemble(ticker, horizon=args.horizon)
    else:
        for ticker in tickers:
            print(f"\n📊 Walk-forward {ticker} (horizon={args.horizon}d)...")
            result = train_walk_forward(ticker, horizon=args.horizon)
            if result:
                print(f"   ✅ Overall: {result['overall_accuracy']:.1%} | "
                      f"≥60%: {result['high_conf_accuracy']:.1%} | "
                      f"≥70%: {result['very_high_conf_accuracy']:.1%}")


if __name__ == "__main__":
    main()

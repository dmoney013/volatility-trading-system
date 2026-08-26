"""
Hybrid LSTM Model — GARCH-informed directional prediction (UP vs DOWN).

Architecture:
    Input(seq_len=30, features=15) → LSTM(64, 2 layers) → BatchNorm → Dense(32) → Dense(2, Softmax)

Key differences from original DMT LSTM:
    1. Binary classification (UP/DOWN only, no FLAT) — forces decisive predictions
    2. GARCH features embedded in input vector (persistence, spread, predicted vol)
    3. Ensemble inference (5 models, averaged softmax) for higher confidence

Used by the Hybrid Scanner to determine directional call/put trades.
GARCH identifies WHEN a big move is coming; LSTM predicts WHICH direction.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, DMT_LSTM_SEQ_LEN, DMT_LSTM_HIDDEN, DMT_LSTM_LAYERS,
    DMT_LSTM_DROPOUT, DMT_ENSEMBLE_SIZE
)


# ═══════════════════════════════════════════════════════════════════
#  Feature Engineering (with GARCH features)
# ═══════════════════════════════════════════════════════════════════

HYBRID_FEATURE_NAMES = [
    # ─── Price / Momentum (5) ──────────────────────────────
    "close_return",         # Close-to-close log return
    "open_close_return",    # Open-to-close return (intraday direction)
    "intraday_range",       # (High - Low) / Close (intraday volatility)
    "momentum_5d",          # 5-day momentum (cumulative return)
    "momentum_10d",         # 10-day momentum
    # ─── Volume (1) ───────────────────────────────────────
    "volume_surge",         # Volume / 20-day avg volume
    # ─── Technical Indicators (3) ──────────────────────────
    "rsi_14",               # RSI(14)
    "macd_histogram",       # MACD histogram (momentum)
    "bollinger_pct_b",      # Bollinger %B (price position within bands)
    # ─── Market Context (2) ────────────────────────────────
    "vix_level",            # VIX level (market fear)
    "vix_change",           # VIX daily % change
    # ─── GARCH Embedded Features (4) ───────────────────────
    "garch_cond_vol",       # GARCH conditional volatility
    "garch_persistence",    # α + β (vol regime stickiness)
    "garch_spread",         # (Predicted RV - IV) / IV (vol mispricing)
    "garch_dampened",       # Binary: was persistence clamped? (0/1)
]

NUM_HYBRID_FEATURES = len(HYBRID_FEATURE_NAMES)


def compute_features(prices: pd.DataFrame,
                     vix: Optional[pd.DataFrame] = None,
                     garch_persistence: float = 0.0,
                     garch_spread: float = 0.0,
                     garch_dampened: bool = False,
                     garch_cond_vol: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Compute LSTM input features from OHLCV + GARCH outputs.

    Args:
        prices: DataFrame with columns [Open, High, Low, Close, Volume]
        vix: Optional DataFrame with VIX Close prices
        garch_persistence: GARCH α + β value (scalar, applied to all rows)
        garch_spread: (Predicted RV - IV) / IV (scalar)
        garch_dampened: Whether GARCH persistence was clamped
        garch_cond_vol: Optional Series of GARCH conditional volatility

    Returns:
        DataFrame with 15 computed features, NaN rows dropped
    """
    df = pd.DataFrame(index=prices.index)

    close = prices['Close']
    open_ = prices['Open']
    high = prices['High']
    low = prices['Low']
    volume = prices['Volume']

    # ─── Price / Momentum ──────────────────────────────────
    df['close_return'] = np.log(close / close.shift(1))
    df['open_close_return'] = (close - open_) / open_
    df['intraday_range'] = (high - low) / close
    df['momentum_5d'] = close.pct_change(5)
    df['momentum_10d'] = close.pct_change(10)

    # ─── Volume ────────────────────────────────────────────
    vol_avg_20 = volume.rolling(20).mean()
    df['volume_surge'] = volume / vol_avg_20

    # ─── RSI(14) ───────────────────────────────────────────
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ─── MACD ──────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = macd_line - signal_line

    # ─── Bollinger %B ──────────────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['bollinger_pct_b'] = (close - lower) / (upper - lower)

    # ─── VIX ───────────────────────────────────────────────
    if vix is not None and not vix.empty:
        vix_close = vix['Close'].reindex(df.index, method='ffill')
        df['vix_level'] = vix_close
        df['vix_change'] = vix_close.pct_change()
    else:
        df['vix_level'] = 0.0
        df['vix_change'] = 0.0

    # ─── GARCH Features (embedded) ─────────────────────────
    if garch_cond_vol is not None:
        df['garch_cond_vol'] = garch_cond_vol.reindex(df.index)
    else:
        # Fallback: use 21-day rolling RV as proxy
        df['garch_cond_vol'] = df['close_return'].rolling(21).std() * np.sqrt(252)

    df['garch_persistence'] = garch_persistence
    df['garch_spread'] = garch_spread
    df['garch_dampened'] = 1.0 if garch_dampened else 0.0

    return df.dropna()


def create_labels(prices: pd.DataFrame, horizon: int = 5,
                  min_move: float = 0.01) -> Tuple[pd.Series, pd.Index]:
    """
    Create binary labels: 0=DOWN, 1=UP.
    Excludes samples where |forward_return| < min_move (true sideways).

    Returns:
        (labels, valid_index) — labels for decisive moves only
    """
    close = prices['Close']
    forward_return = close.shift(-horizon) / close - 1

    # Only keep decisive moves
    decisive = forward_return.abs() >= min_move
    labels = (forward_return > 0).astype(int)  # 1=UP, 0=DOWN

    valid_idx = decisive[decisive].index
    labels = labels.loc[valid_idx]

    return labels, valid_idx


# ═══════════════════════════════════════════════════════════════════
#  LSTM Model (Binary: UP vs DOWN)
# ═══════════════════════════════════════════════════════════════════

class DirectionalLSTM(nn.Module):
    """
    2-layer LSTM for binary directional prediction (DOWN/UP).

    Architecture:
        LSTM(hidden=64, layers=2, dropout=0.2)
        → BatchNorm(64)
        → Linear(64 → 32) → ReLU → Dropout(0.3)
        → Linear(32 → 2) → Softmax
    """

    def __init__(self, input_dim: int = NUM_HYBRID_FEATURES,
                 hidden_dim: int = DMT_LSTM_HIDDEN,
                 num_layers: int = DMT_LSTM_LAYERS,
                 dropout: float = DMT_LSTM_DROPOUT,
                 num_classes: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            logits: (batch, 2) — raw scores for [DOWN, UP]
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        # BatchNorm requires >1 samples during training; skip for single-sample
        if last_hidden.size(0) > 1 or not self.training:
            out = self.bn(last_hidden)
        else:
            out = last_hidden
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Get softmax probabilities for each class.

        Returns:
            probs: (batch, 2) — [P(DOWN), P(UP)]
        """
        self.eval()
        with torch.no_grad():
            # Ensure input is on the same device as the model
            device = next(self.parameters()).device
            x = x.to(device)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════
#  Ensemble LSTM (N models, averaged softmax)
# ═══════════════════════════════════════════════════════════════════

class EnsembleLSTM:
    """
    Wraps N DirectionalLSTM models. At inference, averages their
    softmax outputs for more robust, better-calibrated predictions.
    """

    def __init__(self, models: List[DirectionalLSTM]):
        self.models = models
        for m in self.models:
            m.eval()

    @classmethod
    def load(cls, model_dir: str, input_dim: int = NUM_HYBRID_FEATURES,
             n_models: int = DMT_ENSEMBLE_SIZE) -> 'EnsembleLSTM':
        """Load ensemble from a directory of saved weights."""
        model_dir = Path(model_dir)
        models = []
        for i in range(n_models):
            path = model_dir / f"model_{i}.pt"
            if path.exists():
                model = DirectionalLSTM(input_dim=input_dim)
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                models.append(model)
        if not models:
            raise FileNotFoundError(f"No models found in {model_dir}")
        return cls(models)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Average softmax across all models.

        Returns:
            probs: (batch, 2) — [P(DOWN), P(UP)]
        """
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(x)
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def predict_direction(self, x: torch.Tensor,
                          confidence_threshold: float = 0.60
                          ) -> Tuple[str, float]:
        """
        Get directional prediction with ensemble confidence.

        Returns:
            (direction, confidence) where direction is 'UP' or 'DOWN'
        """
        probs = self.predict_proba(x)
        if probs.ndim == 2:
            probs = probs[0]

        p_down, p_up = probs[0], probs[1]
        direction = 'UP' if p_up >= p_down else 'DOWN'
        confidence = max(p_up, p_down)

        return direction, confidence


# ═══════════════════════════════════════════════════════════════════
#  Dataset Utilities
# ═══════════════════════════════════════════════════════════════════

class SequenceDataset(torch.utils.data.Dataset):
    """
    Creates sliding-window sequences from feature matrix + labels.
    Handles the case where labels may be sparse (binary, no FLAT).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 label_indices: np.ndarray,
                 seq_len: int = DMT_LSTM_SEQ_LEN):
        """
        Args:
            features: Full feature matrix (N, F)
            labels: Binary labels for decisive moves only
            label_indices: Indices into features where labels exist
            seq_len: Lookback window
        """
        self.seq_len = seq_len
        self.features = features
        self.labels = labels
        self.label_indices = label_indices

        # Only keep label indices where we have enough history
        self.valid_indices = label_indices[label_indices >= seq_len]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        label_idx = self.valid_indices[idx]
        x = self.features[label_idx - self.seq_len:label_idx]
        y = self.labels[np.where(self.label_indices == label_idx)[0][0]]
        return (
            torch.FloatTensor(x),
            torch.LongTensor([y]).squeeze()
        )


def normalize_features(features: np.ndarray,
                       mean: Optional[np.ndarray] = None,
                       std: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features. Returns (normalized, mean, std).
    """
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
        std[std < 1e-8] = 1.0

    return (features - mean) / std, mean, std


# ═══════════════════════════════════════════════════════════════════
#  Convenience: Load model or ensemble
# ═══════════════════════════════════════════════════════════════════

def load_ensemble(ticker: str, horizon: int = 5) -> EnsembleLSTM:
    """Load a trained ensemble for a specific ticker and horizon."""
    model_dir = Path(f"cache/lstm_weights/{ticker}_{horizon}d_ensemble")
    return EnsembleLSTM.load(str(model_dir))


if __name__ == "__main__":
    # Quick sanity check
    model = DirectionalLSTM()
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Features: {NUM_HYBRID_FEATURES} ({', '.join(HYBRID_FEATURE_NAMES)})")

    # Test with random input
    x = torch.randn(4, DMT_LSTM_SEQ_LEN, NUM_HYBRID_FEATURES)
    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Softmax: {torch.softmax(logits, dim=1)}")

    # Test ensemble
    ensemble = EnsembleLSTM([DirectionalLSTM() for _ in range(3)])
    direction, confidence = ensemble.predict_direction(x[:1])
    print(f"\nEnsemble prediction: {direction} ({confidence:.1%})")

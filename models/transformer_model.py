"""
Transformer Volatility Model — PyTorch Transformer encoder for volatility
prediction with attention-based feature importance extraction.

Architecture:
  Input (seq_len × n_features)
    → Linear projection → d_model
    → Positional Encoding (sinusoidal)
    → TransformerEncoder (N layers, H heads)
    → Global Average Pooling
    → Linear Head → 1 (predicted realized volatility)

Feature importance is extracted via:
  1. Attention weight averaging across heads/layers
  2. Gradient-based saliency as validation
"""
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, SEQ_LEN, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD,
    DROPOUT, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    BATCH_SIZE, WALK_FORWARD_TEST_DAYS, FEATURE_NAMES, NUM_FEATURES,
)


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class VolatilityDataset(Dataset):
    """
    Time series dataset that creates sliding windows of features
    paired with next-day realized volatility targets.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]  # (seq_len, n_features)
        y = self.targets[idx + self.seq_len]           # scalar
        return x, y


# ═══════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════
# Transformer Model
# ═══════════════════════════════════════════════════════════════════

class VolatilityTransformer(nn.Module):
    """
    Transformer encoder for volatility prediction.

    Produces:
      - Volatility prediction (scalar)
      - Attention weights for interpretability
    """

    def __init__(
        self,
        n_features: int = NUM_FEATURES,
        d_model: int = D_MODEL,
        n_head: int = N_HEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers

        # Input projection: n_features → d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Prediction head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Storage for attention weights (populated during forward pass)
        self._attention_weights = []

    def forward(self, x, return_attention: bool = False):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, n_features)
            return_attention: If True, also return attention weights

        Returns:
            pred: (batch, 1) volatility prediction
            attn_weights: list of attention weight tensors (if requested)
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        # Collect attention weights via hooks if requested
        if return_attention:
            self._attention_weights = []
            hooks = self._register_attention_hooks()

        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling over sequence dimension
        pooled = encoded.mean(dim=1)  # (batch, d_model)

        # Prediction
        pred = self.output_head(pooled)  # (batch, 1)

        if return_attention:
            # Remove hooks
            for h in hooks:
                h.remove()
            return pred, self._attention_weights

        return pred

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights."""
        hooks = []

        for layer in self.transformer_encoder.layers:
            def hook_fn(module, input, output, _layer=layer):
                # Re-run the self-attention to get weights
                # The TransformerEncoderLayer doesn't expose attention weights
                # directly, so we capture them via the self_attn module
                with torch.no_grad():
                    src = input[0]
                    _, attn_w = _layer.self_attn(
                        src, src, src,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    self._attention_weights.append(attn_w.detach())

            h = layer.register_forward_hook(hook_fn)
            hooks.append(h)

        return hooks


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class TransformerTrainer:
    """
    Walk-forward training and evaluation for the VolatilityTransformer.
    """

    def __init__(self, n_features: int = NUM_FEATURES):
        self.model = VolatilityTransformer(n_features=n_features).to(DEVICE)
        self.n_features = n_features
        self.training_history = []
        self.feature_importance = None

    def prepare_data(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align features and targets, handle NaNs.
        """
        # Align indices
        common_idx = features.index.intersection(targets.index)
        feat = features.loc[common_idx].values
        tgt = targets.loc[common_idx].values

        # Remove any remaining NaN rows
        valid = ~(np.isnan(feat).any(axis=1) | np.isnan(tgt))
        feat = feat[valid]
        tgt = tgt[valid]

        return feat, tgt

    def train(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        verbose: bool = True,
    ) -> dict:
        """
        Train the model using walk-forward validation.

        Args:
            features: Normalized feature DataFrame
            targets: Target volatility Series
            verbose: Print progress

        Returns:
            Dictionary with training metrics and feature importance
        """
        feat, tgt = self.prepare_data(features, targets)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Transformer Training")
            print(f"{'='*60}")
            print(f"Data: {len(feat)} samples, {feat.shape[1]} features")
            print(f"Device: {DEVICE}")

        # Walk-forward split
        test_size = WALK_FORWARD_TEST_DAYS
        train_size = len(feat) - test_size

        if train_size < SEQ_LEN * 2:
            raise ValueError(
                f"Not enough data for training. Need at least {SEQ_LEN * 2} samples, "
                f"got {train_size}"
            )

        train_feat, test_feat = feat[:train_size], feat[train_size:]
        train_tgt, test_tgt = tgt[:train_size], tgt[train_size:]

        if verbose:
            print(f"Train: {len(train_feat)} samples | Test: {len(test_feat)} samples")

        # Create datasets and dataloaders
        train_dataset = VolatilityDataset(train_feat, train_tgt, SEQ_LEN)
        test_dataset = VolatilityDataset(test_feat, test_tgt, SEQ_LEN)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            # ─── Train ──────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # ─── Validate ───────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE).unsqueeze(1)
                    pred = self.model(x_batch)
                    val_loss += criterion(pred, y_batch).item()
                    n_val += 1

            val_loss /= max(n_val, 1)
            scheduler.step()

            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Patience: {patience_counter}/{PATIENCE}"
                )

            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(DEVICE)

        # ─── Extract Feature Importance ─────────────────────────
        if verbose:
            print("\nExtracting feature importance from attention weights...")

        self.feature_importance = self._extract_feature_importance(
            test_feat, test_tgt
        )

        # Compute final metrics
        naive_baseline_mse = np.var(test_tgt[SEQ_LEN:])

        results = {
            "best_val_loss": best_val_loss,
            "naive_baseline_mse": naive_baseline_mse,
            "improvement_over_naive": (naive_baseline_mse - best_val_loss) / naive_baseline_mse * 100,
            "epochs_trained": len(self.training_history),
            "feature_importance": self.feature_importance,
        }

        if verbose:
            print(f"\n✓ Training complete")
            print(f"  Best validation MSE: {best_val_loss:.6f}")
            print(f"  Naive baseline MSE:  {naive_baseline_mse:.6f}")
            print(f"  Improvement:         {results['improvement_over_naive']:.1f}%")
            self.print_feature_importance()

        return results

    def _extract_feature_importance(
        self,
        test_feat: np.ndarray,
        test_tgt: np.ndarray,
    ) -> pd.DataFrame:
        """
        Extract feature importance using attention weights and gradient saliency.
        """
        self.model.eval()
        dataset = VolatilityDataset(test_feat, test_tgt, SEQ_LEN)
        loader = DataLoader(dataset, batch_size=min(len(dataset), 64), shuffle=False)

        all_attention_importance = []
        all_gradient_importance = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            x_batch.requires_grad_(True)

            # ─── Attention-based importance ──────────────────────
            pred, attn_weights = self.model(x_batch, return_attention=True)

            if attn_weights:
                # attn_weights: list of (batch, n_head, seq_len, seq_len)
                # Average across layers and heads → (batch, seq_len, seq_len)
                avg_attn = torch.stack(attn_weights).mean(dim=[0, 2])  # (batch, seq_len, seq_len)

                # Sum attention received by each time step
                # → (batch, seq_len) — how much attention each position receives
                temporal_importance = avg_attn.sum(dim=-2)  # (batch, seq_len)

                # Map back to feature space using input gradients
                # → (batch, seq_len) represents temporal importance
                all_attention_importance.append(temporal_importance.cpu().numpy())

            # ─── Gradient-based saliency ─────────────────────────
            if x_batch.grad is not None:
                x_batch.grad.zero_()
            pred_sum = pred.sum()
            pred_sum.backward(retain_graph=True)

            if x_batch.grad is not None:
                # Gradient saliency: |∂output/∂input|
                grad_importance = x_batch.grad.abs().mean(dim=1)  # (batch, n_features)
                all_gradient_importance.append(grad_importance.detach().cpu().numpy())

            x_batch.requires_grad_(False)

        # ─── Aggregate importance scores ─────────────────────────
        feature_names = FEATURE_NAMES[:self.n_features]

        # Gradient-based importance (primary method — more reliable for feature-level)
        if all_gradient_importance:
            grad_imp = np.concatenate(all_gradient_importance, axis=0)
            grad_mean = grad_imp.mean(axis=0)
            grad_std = grad_imp.std(axis=0)

            # Normalize to sum to 1
            grad_normalized = grad_mean / grad_mean.sum()
        else:
            grad_normalized = np.ones(self.n_features) / self.n_features
            grad_std = np.zeros(self.n_features)

        # Attention-based temporal importance
        if all_attention_importance:
            attn_imp = np.concatenate(all_attention_importance, axis=0)
            temporal_mean = attn_imp.mean(axis=0)
        else:
            temporal_mean = np.ones(SEQ_LEN) / SEQ_LEN

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": grad_normalized,
            "Importance_Std": grad_std / grad_mean.sum() if grad_mean.sum() > 0 else grad_std,
            "Rank": np.argsort(np.argsort(-grad_normalized)) + 1,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        # Add temporal importance as a separate attribute
        self._temporal_importance = temporal_mean

        return importance_df

    def get_temporal_importance(self) -> np.ndarray:
        """
        Return temporal importance scores.

        Shape: (seq_len,) — importance of each lagged time step.
        Day 0 = most recent, day seq_len-1 = oldest.
        """
        if hasattr(self, "_temporal_importance"):
            return self._temporal_importance
        return np.ones(SEQ_LEN) / SEQ_LEN

    def predict(self, features: np.ndarray) -> float:
        """
        Make a single prediction from the latest features.

        Args:
            features: (seq_len, n_features) array of recent feature values

        Returns:
            Predicted realized volatility (scalar)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
            pred = self.model(x)
            return pred.item()

    def print_feature_importance(self, top_n: int = 10):
        """Print ranked feature importance."""
        if self.feature_importance is None:
            print("No feature importance available. Run train() first.")
            return

        print(f"\n{'─'*50}")
        print(f"Top {top_n} Volatility Drivers (Transformer)")
        print(f"{'─'*50}")

        for i, row in self.feature_importance.head(top_n).iterrows():
            bar = "█" * int(row["Importance"] * 100)
            print(
                f"  #{int(row['Rank']):2d}  {row['Feature']:20s}  "
                f"{row['Importance']:.4f} ± {row['Importance_Std']:.4f}  {bar}"
            )

    def get_training_history(self) -> pd.DataFrame:
        """Return training loss history as DataFrame."""
        return pd.DataFrame(self.training_history)


if __name__ == "__main__":
    # Quick test with synthetic data
    print(f"Device: {DEVICE}")

    n_samples = 500
    n_features = NUM_FEATURES

    # Synthetic features and targets
    np.random.seed(42)
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=FEATURE_NAMES,
        index=pd.date_range("2022-01-01", periods=n_samples, freq="B"),
    )
    targets = pd.Series(
        np.abs(np.random.randn(n_samples)) * 0.01,
        index=features.index,
        name="target_vol",
    )

    trainer = TransformerTrainer(n_features=n_features)
    results = trainer.train(features, targets)

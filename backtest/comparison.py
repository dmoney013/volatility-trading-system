"""
Model Comparison — GARCH-only vs GARCH+Transformer integrated model.

Uses expanding-window cross-validation with multiple train/val/test folds
to provide robust, overfit-aware performance comparison.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, MAX_POSITION_PCT, COMMISSION_PER_CONTRACT, TRADING_DAYS,
)
from backtest.engine import straddle_price, Trade


# ═══════════════════════════════════════════════════════════════════
# Expanding-Window Cross-Validation Splitter
# ═══════════════════════════════════════════════════════════════════

def generate_cv_folds(n_samples, n_folds=4, val_pct=0.15, test_pct=0.15):
    """
    Generate expanding-window folds for time series cross-validation.
    
    Each fold has more training data than the last. The test window
    slides forward so each fold evaluates on different market conditions.
    """
    test_size = int(n_samples * test_pct)
    val_size = int(n_samples * val_pct)
    usable = n_samples - test_size  # Last chunk always available for test
    
    folds = []
    step = (usable - val_size - 200) // max(n_folds - 1, 1)  # Min 200 train
    
    for i in range(n_folds):
        test_end = n_samples - i * step
        test_start = test_end - test_size
        val_end = test_start
        val_start = val_end - val_size
        train_end = val_start
        
        if train_end < 200 or test_start < 0:
            break
            
        folds.append({
            "fold": i + 1,
            "train": (0, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
        })
    
    return folds


# ═══════════════════════════════════════════════════════════════════
# Core Backtest Engine (shared by both models)
# ═══════════════════════════════════════════════════════════════════

def _run_backtest(test_prices, signal_series, treasury, ticker,
                  initial_capital, holding_period, entry_threshold):
    """
    Run backtest using a pre-computed signal series.
    Signal > entry_threshold triggers a long straddle entry.
    """
    close = test_prices["Close"]
    common_idx = signal_series.index.intersection(close.index).sort_values()

    if not treasury.empty and "Close" in treasury.columns:
        rf = treasury["Close"].reindex(common_idx, method="ffill") / 100.0
    else:
        rf = pd.Series(0.04, index=common_idx)

    log_ret = np.log(close / close.shift(1))
    mkt_iv = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    mkt_iv = mkt_iv.reindex(common_idx, method="ffill")

    capital = initial_capital
    trades, equity = [], []
    in_position = False

    for i in range(len(common_idx)):
        date = common_idx[i]
        spot = close.loc[date]
        sig = signal_series.get(date, 0)
        iv = mkt_iv.get(date, 0.3)
        if np.isnan(iv): iv = 0.3
        r = rf.get(date, 0.04)
        equity.append({"date": date, "equity": capital})

        if in_position:
            if i - entry_idx >= holding_period:
                call_v = max(spot - ed["strike"], 0)
                put_v = max(ed["strike"] - spot, 0)
                exit_v = call_v + put_v
                pnl = (exit_v - ed["price"]) * ed["contracts"] * 100
                net = pnl - COMMISSION_PER_CONTRACT * 4
                capital += net
                trades.append(net)
                in_position = False
        else:
            if sig > entry_threshold and i + holding_period < len(common_idx):
                T = holding_period / TRADING_DAYS
                strike = round(spot, 0)
                price = straddle_price(spot, strike, T, r, iv)
                if price <= 0.01: continue
                cost = price * 100
                contracts = int((capital * MAX_POSITION_PCT - COMMISSION_PER_CONTRACT * 2) / cost)
                if contracts < 1: continue
                ed = {"strike": strike, "price": price, "contracts": contracts}
                entry_idx = i
                in_position = True

    # Compute stats
    if not trades:
        return {"return_pct": 0, "trades": 0, "win_rate": 0, "pf": 0,
                "sharpe": 0, "max_dd": 0, "final": capital}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    pf = total_win / total_loss if total_loss > 0 else float("inf")

    eq = pd.Series([e["equity"] for e in equity])
    dd = (eq / eq.cummax() - 1).min() * 100
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(TRADING_DAYS) if daily_ret.std() > 0 else 0

    return {
        "return_pct": (capital - initial_capital) / initial_capital * 100,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "pf": min(pf, 99.99),
        "sharpe": sharpe,
        "max_dd": dd,
        "final": capital,
    }


# ═══════════════════════════════════════════════════════════════════
# Signal Generators
# ═══════════════════════════════════════════════════════════════════

def garch_only_signal(cond_vol, prices, idx):
    """GARCH-only signal: RV forecast - market IV."""
    close = prices["Close"]
    log_ret = np.log(close / close.shift(1))
    mkt_iv = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    mkt_iv = mkt_iv.reindex(idx, method="ffill")
    cv = cond_vol.reindex(idx, method="ffill")
    return cv - mkt_iv


def garch_transformer_signal(cond_vol, prices, idx, transformer_preds):
    """
    Integrated GARCH+Transformer signal.
    
    The Transformer predicts next-day absolute return (vol proxy).
    We boost the GARCH signal when the Transformer ALSO predicts high vol,
    and dampen it when the Transformer disagrees.
    
    Signal = GARCH_spread × Transformer_confidence
    
    Transformer_confidence:
      - > 1.0 when Transformer predicts vol above median (confirms GARCH)
      - < 1.0 when Transformer predicts vol below median (disagrees)
    """
    garch_sig = garch_only_signal(cond_vol, prices, idx)
    
    # Normalize transformer predictions to a confidence multiplier
    preds = transformer_preds.reindex(idx, method="ffill")
    pred_median = preds.rolling(21, min_periods=5).median()
    pred_std = preds.rolling(21, min_periods=5).std().replace(0, 1e-8)
    
    # Z-score of transformer prediction
    z = (preds - pred_median) / pred_std
    
    # Convert to confidence multiplier: sigmoid-like mapping
    # z > 0 → confidence > 1 (Transformer confirms high vol)
    # z < 0 → confidence < 1 (Transformer says low vol)
    confidence = 1.0 / (1.0 + np.exp(-z))  # Sigmoid: range (0, 1)
    confidence = 0.5 + confidence  # Shift to range (0.5, 1.5)
    
    return garch_sig * confidence


# ═══════════════════════════════════════════════════════════════════
# Transformer Training for a Given Fold
# ═══════════════════════════════════════════════════════════════════

def train_transformer_for_fold(prices, train_idx, val_idx, test_idx):
    """
    Train Transformer on training data and generate predictions for all periods.
    Returns predictions as a pd.Series indexed by date.
    """
    from data.fetcher import fetch_vix, fetch_treasury_yield
    from data.feature_engineer import build_features, normalize_features, build_target
    from models.transformer_model import TransformerTrainer

    vix_data = fetch_vix()
    treasury = fetch_treasury_yield()
    options = {"calls": pd.DataFrame(), "puts": pd.DataFrame()}

    features = build_features(prices, vix_data, treasury, options)
    normed = normalize_features(features)
    targets = build_target(prices)

    trainer = TransformerTrainer(n_features=normed.shape[1])

    # Align data
    common = normed.index.intersection(targets.dropna().index)
    feat_arr = normed.loc[common].values
    tgt_arr = targets.loc[common].values

    # Split for transformer training using the fold indices
    n = len(feat_arr)
    t_end = min(train_idx[1], n)
    v_end = min(val_idx[1], n)

    if t_end < 100:
        # Not enough data — return flat predictions
        return pd.Series(0.01, index=prices.index)

    train_feat = feat_arr[:t_end]
    train_tgt = tgt_arr[:t_end]

    # Train
    from models.transformer_model import VolatilityDataset
    from torch.utils.data import DataLoader
    import torch
    from config import DEVICE, SEQ_LEN, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE

    train_ds = VolatilityDataset(train_feat, train_tgt, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = trainer.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(min(EPOCHS, 50)):  # Cap at 50 for speed
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Generate predictions for entire dataset
    model.eval()
    predictions = []
    with torch.no_grad():
        for j in range(SEQ_LEN, len(feat_arr)):
            x = torch.FloatTensor(feat_arr[j-SEQ_LEN:j]).unsqueeze(0).to(DEVICE)
            p = model(x).item()
            predictions.append((common[j], p))

    pred_series = pd.Series(
        [p[1] for p in predictions],
        index=[p[0] for p in predictions],
        name="transformer_pred"
    )
    return pred_series


# ═══════════════════════════════════════════════════════════════════
# Main Comparison Runner
# ═══════════════════════════════════════════════════════════════════

def run_comparison(ticker, initial_capital=INITIAL_CAPITAL, holding_period=10,
                   entry_threshold=0.02, n_folds=4, verbose=True):
    """
    Compare GARCH-only vs GARCH+Transformer across multiple CV folds.
    Reports per-fold results, overfitting metrics, and aggregate comparison.
    """
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    if verbose:
        print(f"\n{'#'*60}")
        print(f"MODEL COMPARISON: {ticker}")
        print(f"GARCH-Only vs GARCH+Transformer")
        print(f"{'#'*60}")

    prices = fetch_price_data(ticker)
    treasury = fetch_treasury_yield()
    n = len(prices)

    folds = generate_cv_folds(n, n_folds=n_folds)
    if verbose:
        print(f"\n📊 {len(folds)} cross-validation folds ({n} total trading days)")

    garch_results = []
    integrated_results = []
    overfit_metrics = []

    for fold in folds:
        fi = fold["fold"]
        tr = fold["train"]
        vl = fold["val"]
        ts = fold["test"]

        train_prices = prices.iloc[tr[0]:tr[1]]
        val_prices = prices.iloc[vl[0]:vl[1]]
        test_prices = prices.iloc[ts[0]:ts[1]]

        if verbose:
            print(f"\n{'─'*60}")
            print(f"Fold {fi}: Train [{train_prices.index[0].date()} → {train_prices.index[-1].date()}] "
                  f"Val [{val_prices.index[0].date()} → {val_prices.index[-1].date()}] "
                  f"Test [{test_prices.index[0].date()} → {test_prices.index[-1].date()}]")

        # ── Train GARCH on training set ──
        garch = GARCHVolatilityModel()
        garch.fit(train_prices, verbose=False)

        # Get conditional vol for all periods (expanding window)
        garch_full = GARCHVolatilityModel()
        garch_full.fit(prices.iloc[:ts[1]], verbose=False)
        cond_vol = garch_full.get_conditional_volatility()

        test_idx = test_prices.index
        val_idx_dates = val_prices.index

        # ── GARCH-only signal ──
        garch_sig = garch_only_signal(cond_vol, prices.iloc[:ts[1]], test_idx)
        g_result = _run_backtest(test_prices, garch_sig, treasury, ticker,
                                  initial_capital, holding_period, entry_threshold)
        garch_results.append({"fold": fi, **g_result})

        # ── GARCH-only on TRAINING set (for overfitting detection) ──
        train_idx_dates = train_prices.index
        garch_sig_train = garch_only_signal(cond_vol, prices.iloc[:tr[1]], train_idx_dates)
        g_train = _run_backtest(train_prices, garch_sig_train, treasury, ticker,
                                 initial_capital, holding_period, entry_threshold)

        # ── GARCH-only on VALIDATION set ──
        garch_sig_val = garch_only_signal(cond_vol, prices.iloc[:vl[1]], val_idx_dates)
        g_val = _run_backtest(val_prices, garch_sig_val, treasury, ticker,
                               initial_capital, holding_period, entry_threshold)

        # ── Train Transformer ──
        if verbose: print(f"  Training Transformer for fold {fi}...")
        transformer_preds = train_transformer_for_fold(prices.iloc[:ts[1]], tr, vl, ts)

        # ── Integrated signal ──
        int_sig = garch_transformer_signal(cond_vol, prices.iloc[:ts[1]], test_idx, transformer_preds)
        i_result = _run_backtest(test_prices, int_sig, treasury, ticker,
                                  initial_capital, holding_period, entry_threshold)
        integrated_results.append({"fold": fi, **i_result})

        # ── Overfitting metrics ──
        overfit_metrics.append({
            "fold": fi,
            "train_return": g_train["return_pct"],
            "val_return": g_val["return_pct"],
            "test_return_garch": g_result["return_pct"],
            "test_return_integrated": i_result["return_pct"],
            "overfit_gap": g_train["return_pct"] - g_result["return_pct"],
            "val_test_gap": g_val["return_pct"] - g_result["return_pct"],
        })

        if verbose:
            print(f"  GARCH-only:       {g_result['return_pct']:+8.1f}% | {g_result['trades']} trades | WR {g_result['win_rate']:.0f}% | PF {g_result['pf']:.2f}")
            print(f"  GARCH+Transformer: {i_result['return_pct']:+8.1f}% | {i_result['trades']} trades | WR {i_result['win_rate']:.0f}% | PF {i_result['pf']:.2f}")
            print(f"  Overfit gap:       Train {g_train['return_pct']:+.1f}% → Test {g_result['return_pct']:+.1f}% (Δ = {g_train['return_pct'] - g_result['return_pct']:.1f}pp)")

    # ── Aggregate Results ──
    garch_df = pd.DataFrame(garch_results)
    int_df = pd.DataFrame(integrated_results)
    overfit_df = pd.DataFrame(overfit_metrics)

    if verbose:
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS ACROSS {len(folds)} FOLDS")
        print(f"{'='*60}")
        print(f"\n{'Metric':<25} {'GARCH-Only':>15} {'GARCH+Trans':>15}")
        print(f"{'─'*55}")

        for col, label, fmt in [
            ("return_pct", "Avg Return", "{:+.1f}%"),
            ("win_rate", "Avg Win Rate", "{:.1f}%"),
            ("pf", "Avg Profit Factor", "{:.2f}"),
            ("sharpe", "Avg Sharpe Ratio", "{:.2f}"),
            ("trades", "Avg Trades", "{:.1f}"),
            ("max_dd", "Avg Max Drawdown", "{:.1f}%"),
        ]:
            gm = garch_df[col].mean()
            im = int_df[col].mean()
            print(f"  {label:<23} {fmt.format(gm):>15} {fmt.format(im):>15}")

        better = "GARCH+Transformer" if int_df["return_pct"].mean() > garch_df["return_pct"].mean() else "GARCH-Only"
        print(f"\n  🏆 Winner: {better}")

        # Overfitting analysis
        print(f"\n{'='*60}")
        print(f"OVERFITTING ANALYSIS")
        print(f"{'='*60}")
        print(f"\n  {'Fold':<6} {'Train':>10} {'Val':>10} {'Test(G)':>10} {'Test(G+T)':>10} {'Overfit':>10}")
        print(f"  {'─'*56}")
        for _, row in overfit_df.iterrows():
            print(f"  {int(row['fold']):<6} "
                  f"{row['train_return']:>+9.1f}% "
                  f"{row['val_return']:>+9.1f}% "
                  f"{row['test_return_garch']:>+9.1f}% "
                  f"{row['test_return_integrated']:>+9.1f}% "
                  f"{row['overfit_gap']:>+9.1f}pp")

        avg_gap = overfit_df["overfit_gap"].mean()
        print(f"\n  Avg Train→Test gap: {avg_gap:+.1f}pp")
        if abs(avg_gap) < 50:
            print(f"  ✅ Low overfitting — model generalizes well")
        elif abs(avg_gap) < 200:
            print(f"  ⚠️ Moderate overfitting — some performance degradation OOS")
        else:
            print(f"  🚨 High overfitting — train performance significantly inflated")

        val_test_gap = overfit_df["val_test_gap"].mean()
        print(f"  Avg Val→Test gap:   {val_test_gap:+.1f}pp")
        if abs(val_test_gap) < 50:
            print(f"  ✅ Validation is a good proxy for test performance")
        else:
            print(f"  ⚠️ Validation doesn't perfectly predict test results (market regime shift)")

    return {
        "ticker": ticker,
        "garch_results": garch_df,
        "integrated_results": int_df,
        "overfit_metrics": overfit_df,
        "winner": better,
    }


if __name__ == "__main__":
    run_comparison("MARA", initial_capital=150.0)

"""
Hybrid Scanner — Unified GARCH + Beta + Index LSTM directional scanner.

This REPLACES the standalone DMT and original Hybrid scanner.

Flow per ticker:
  1. Fetch prices → fit GARCH → get persistence, spread, dampened
  2. If dampened or spread <= 0 → skip
  3. Find best fit index via Beta Model (e.g., ARKK, SPY)
  4. If best R^2 < 0.15 → skip (too idiosyncratic)
  5. Fetch index features and run Index LSTM ensemble → P(UP), P(DOWN) for the index
  6. Apply beta multiplier → Stock direction (if beta < 0, reverse direction)
  7. If index confidence < threshold → skip
  8. Find affordable option contract (ATM, ≥14 days expiry)
  9. Score by: index_confidence × abs(beta) × garch_spread
  10. Return top_n ranked candidates
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.garch_model import GARCHVolatilityModel
from models.lstm_model import (
    EnsembleLSTM, compute_features, normalize_features, NUM_HYBRID_FEATURES
)
from models.beta_model import find_best_fit_index, INDEX_BENCHMARKS, _fetch_benchmark
from data.fetcher import fetch_price_data
from config import (
    DEVICE, TRADING_DAYS, MIN_EXPIRY_TRADING_DAYS, REJECT_DAMPENED,
    DMT_BUDGET, DMT_CONFIDENCE_THRESHOLD, DMT_LSTM_SEQ_LEN
)
from signals.scanner import SCAN_UNIVERSE


# ─── Paths ────────────────────────────────────────────────────────
WEIGHTS_DIR = Path("cache/index_lstm_weights")


# Cache for index data to avoid re-fetching and re-computing features
_INDEX_CACHE = {}


def _load_ensemble(symbol: str, horizon: int = 5) -> Optional[EnsembleLSTM]:
    model_dir = WEIGHTS_DIR / f"{symbol}_{horizon}d_ensemble"
    if not model_dir.exists():
        return None
    try:
        return EnsembleLSTM.load(str(model_dir), input_dim=NUM_HYBRID_FEATURES)
    except FileNotFoundError:
        return None


def _load_normalization_stats(symbol: str, horizon: int = 5):
    stats_file = WEIGHTS_DIR / f"{symbol}_{horizon}d_ensemble" / "stats.npz"
    if stats_file.exists():
        stats = np.load(stats_file)
        return stats['mean'], stats['std']
    return None, None


def _get_index_features(index_sym: str, vix: pd.DataFrame):
    """Fetch index data, compute GARCH, and return features DataFrame."""
    if index_sym in _INDEX_CACHE:
        return _INDEX_CACHE[index_sym]
        
    prices = _fetch_benchmark(index_sym)
    if prices is None or len(prices) < 200:
        _INDEX_CACHE[index_sym] = None
        return None
        
    try:
        garch = GARCHVolatilityModel()
        garch.fit(prices, verbose=False)
        persistence = garch.persistence
        dampened = garch.dampened
        
        forecast_df = garch.forecast(horizon=5)
        predicted = forecast_df['Annualized Vol'].iloc[-1]
        recent_rv = prices['Close'].pct_change().rolling(21).std().iloc[-1] * np.sqrt(252)
        spread = 0.0
        if recent_rv > 0:
            spread = (predicted - recent_rv) / recent_rv
            
        log_returns = np.log(prices['Close'] / prices['Close'].shift(1))
        cond_vol = log_returns.rolling(21).std() * np.sqrt(252)
        
        features_df = compute_features(
            prices, vix=vix,
            garch_persistence=persistence,
            garch_spread=spread,
            garch_dampened=dampened,
            garch_cond_vol=cond_vol
        )
        _INDEX_CACHE[index_sym] = features_df
        return features_df
    except Exception as e:
        print(f"  ❌ Error computing features for {index_sym}: {e}")
        _INDEX_CACHE[index_sym] = None
        return None


def scan_hybrid_opportunities(budget: float = DMT_BUDGET,
                              top_n: int = 3,
                              horizon: int = 5,
                              tickers: Optional[List[str]] = None,
                              verbose: bool = True) -> List[Dict]:
    scan_tickers = tickers or SCAN_UNIVERSE
    min_calendar_days = int(MIN_EXPIRY_TRADING_DAYS * 7 / 5)
    min_expiry_date = datetime.now() + timedelta(days=min_calendar_days)

    if verbose:
        print(f"\n🔮 Hybrid Scanner (Index-Beta Translation)")
        print(f"   Budget: ${budget:.0f} | Horizon: {horizon}d | "
              f"Threshold: {DMT_CONFIDENCE_THRESHOLD:.0%}")
        print(f"   Scanning {len(scan_tickers)} tickers...\n")

    # Fetch VIX once
    try:
        vix = yf.download("^VIX", period="2y", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
    except Exception:
        vix = None

    results = []

    for sym in scan_tickers:
        try:
            # ─── 1. Fetch stock prices & GARCH ───────────────────────────
            prices = fetch_price_data(sym)
            if prices is None or len(prices) < 300:
                continue

            spot = prices['Close'].iloc[-1]

            garch = GARCHVolatilityModel()
            garch.fit(prices, verbose=False)

            # Dampened rejection
            if REJECT_DAMPENED and garch.dampened:
                if verbose:
                    print(f"  ⚠️  {sym}: dampened — skipping")
                continue

            # Calculate spread
            forecast_df = garch.forecast(horizon=5)
            predicted_vol = forecast_df['Annualized Vol'].iloc[-1]
            log_ret = np.log(prices['Close'] / prices['Close'].shift(1))
            hist_vol_30d = (log_ret.rolling(30).std().iloc[-1] * np.sqrt(TRADING_DAYS))
            
            if np.isnan(hist_vol_30d):
                continue
                
            garch_rv = garch.get_conditional_volatility().iloc[-1]
            spread = garch_rv - hist_vol_30d

            if spread <= 0:
                if verbose:
                    print(f"  ⚠️  {sym}: negative GARCH spread ({spread:+.1%}) — skipping")
                continue

            # ─── 2. Find best fit index via Beta ─────────────────────────
            # Only test against our trainable core indices
            core_benchmarks = {k: INDEX_BENCHMARKS[k] for k in ['ARKK', 'BITO', 'SPY', 'QQQ', 'IWM', 'DIA']}
            fit = find_best_fit_index(prices, benchmarks=core_benchmarks)
            
            if not fit or fit['best_r2'] < 0.15:
                if verbose:
                    r2_str = f"{fit['best_r2']:.2f}" if fit else "N/A"
                    print(f"  ⚠️  {sym}: R^2 too low ({r2_str} < 0.15) — skipping")
                continue
                
            best_idx = fit['best_benchmark']
            beta = fit['best_beta']
            r2 = fit['best_r2']

            # ─── 3. Run Index LSTM Inference ─────────────────────────────
            ensemble = _load_ensemble(best_idx, horizon)
            if ensemble is None:
                if verbose:
                    print(f"  ⚠️  {sym}: No saved LSTM for {best_idx} — skipping")
                continue
                
            idx_features = _get_index_features(best_idx, vix)
            if idx_features is None or len(idx_features) < DMT_LSTM_SEQ_LEN:
                continue
                
            feat_np = idx_features.values.astype(np.float32)
            mean, std = _load_normalization_stats(best_idx, horizon)
            if mean is None:
                if verbose:
                    print(f"  ⚠️  {sym}: No stats.npz for {best_idx} — skipping")
                continue
                
            feat_norm, _, _ = normalize_features(feat_np, mean=mean, std=std)
            seq = feat_norm[-DMT_LSTM_SEQ_LEN:]
            x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

            idx_direction, idx_confidence = ensemble.predict_direction(
                x, confidence_threshold=DMT_CONFIDENCE_THRESHOLD
            )
            
            if idx_confidence < DMT_CONFIDENCE_THRESHOLD:
                if verbose:
                    print(f"  ⚠️  {sym}: {best_idx} index confidence too low ({idx_confidence:.0%})")
                continue

            # ─── 4. Translate Index Direction to Stock Direction ─────────
            if beta < 0:
                stock_direction = 'DOWN' if idx_direction == 'UP' else 'UP'
            else:
                stock_direction = idx_direction

            # ─── 5. Find Options & Score ─────────────────────────────────
            option = _find_option(sym, spot, stock_direction, budget, min_expiry_date)
            if option is None:
                if verbose:
                    print(f"  ⚠️  {sym}: no affordable option found")
                continue

            garch_spread_pct = spread / max(hist_vol_30d, 0.01)
            score = idx_confidence * abs(beta) * garch_spread_pct

            result = {
                'ticker': sym,
                'direction': stock_direction,
                'option_type': 'CALL' if stock_direction == 'UP' else 'PUT',
                'index': best_idx,
                'idx_dir': idx_direction,
                'beta': round(beta, 2),
                'r2': round(r2, 2),
                'confidence': round(idx_confidence, 4),
                'spot': round(spot, 2),
                'strike': option['strike'],
                'expiry': option['expiry'],
                'option_price': option['price'],
                'total_cost': option['total_cost'],
                'contracts': option['contracts'],
                'garch_spread': round(spread, 4),
                'signal_strength': round(garch_spread_pct, 3),
                'score': round(score, 4)
            }
            results.append(result)

            if verbose:
                print(f"  ✅ {sym}: {stock_direction} (via {best_idx} {idx_direction} | β={beta:.2f}) | "
                      f"Conf: {idx_confidence:.0%} | Score: {score:.3f} | "
                      f"${option['strike']} {result['option_type']} ${option['total_cost']:.2f}")

        except Exception as e:
            if verbose:
                print(f"  ❌ {sym}: {e}")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    top = results[:top_n]

    if verbose:
        print(f"\n{'='*75}")
        print(f"  TOP {len(top)} HYBRID OPPORTUNITIES (INDEX-BETA)")
        print(f"{'='*75}")
        for i, r in enumerate(top):
            print(f"  {i+1}. {r['ticker']} {r['option_type']} ${r['strike']} exp {r['expiry']}")
            print(f"     └─ Score: {r['score']:.3f} | Index: {r['index']} {r['idx_dir']} "
                  f"(β={r['beta']:.2f}, R²={r['r2']:.2f}) | Conf: {r['confidence']:.0%}")
            print(f"     └─ GARCH: +{r['signal_strength']:.1%} | Cost: ${r['total_cost']:.2f}")

    return top


def _find_option(ticker: str, spot: float, direction: str,
                 budget: float, min_expiry_date: datetime) -> Optional[Dict]:
    """
    Find an affordable, quality option contract for a directional trade.

    v2 quality gates applied:
      - Minimum price $0.30 (no penny options)
      - Minimum volume 50
      - Max bid-ask spread 20%
      - Cost must be within budget
    """
    # Import HAT v2 quality thresholds if available, else use safe defaults
    try:
        from broker.hat import MIN_OPTION_PRICE, MIN_OPTION_VOLUME, MAX_BID_ASK_SPREAD_PCT
    except ImportError:
        MIN_OPTION_PRICE = 0.30
        MIN_OPTION_VOLUME = 50
        MAX_BID_ASK_SPREAD_PCT = 0.20

    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None

        valid_exps = [e for e in exps if datetime.strptime(e, '%Y-%m-%d') >= min_expiry_date]
        if not valid_exps:
            return None

        best_exp = min(valid_exps, key=lambda x: datetime.strptime(x, '%Y-%m-%d') - min_expiry_date)
        chain = tk.option_chain(best_exp)
        options = chain.calls if direction == 'UP' else chain.puts
        atm = round(spot)

        for strike in [atm, atm - 1, atm + 1]:
            row = options[options['strike'] == strike]
            if row.empty:
                continue
            row = row.iloc[0]

            bid = float(row.get('bid', 0) or 0)
            ask = float(row.get('ask', 0) or 0)
            price = row['lastPrice']
            if price < 0.03:
                price = (bid + ask) / 2
            if price < 0.03:
                continue

            # v2 quality gate: minimum option price
            if price < MIN_OPTION_PRICE:
                continue

            # v2 quality gate: minimum volume
            vol = int(row.get('volume', 0) or 0)
            if vol < MIN_OPTION_VOLUME:
                continue

            # v2 quality gate: bid-ask spread
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else price
            if mid > 0 and bid > 0 and ask > 0:
                spread_pct = (ask - bid) / mid
                if spread_pct > MAX_BID_ASK_SPREAD_PCT:
                    continue

            total_cost = price * 100 + 0.65
            if total_cost > budget or total_cost < 5:
                continue

            contracts = int((budget - 0.65) / (price * 100))
            if contracts < 1:
                continue

            return {
                'strike': int(strike),
                'expiry': best_exp,
                'price': round(price, 2),
                'total_cost': round(total_cost * contracts + 0.65, 2),
                'contracts': contracts
            }
        return None
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Scanner — Index-Beta Translation")
    parser.add_argument("--budget", type=float, default=DMT_BUDGET)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    tickers = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]

    scan_hybrid_opportunities(
        budget=args.budget,
        top_n=args.top,
        horizon=args.horizon,
        tickers=tickers,
    )

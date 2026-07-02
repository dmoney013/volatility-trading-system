"""
Futures Bracket Backtester (fv3) — Tests the GARCH bracket breakout
strategy on daily SPY OHLC data as a proxy for /MES.

Backtest logic (simplified signal-quality test):
  - GARCH model sets bracket_upper and bracket_lower from Open each day
  - If High >= bracket_upper → profitable trade (WIN)
  - If Low  <= bracket_lower → profitable trade (WIN)
  - If Open == High or Open == Low → skip day (degenerate bar)
  - If neither bracket hit → LOSS (GARCH predicted vol expansion that
    didn't materialize)

Uses train / validation / test split with most recent data for training
(calibrates to current regime), middle for validation, oldest for test.

Each split uses unique non-overlapping data. The GARCH model is re-fit
on a rolling window within each split's data only — no data leakage.
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.garch_futures_model import GARCHFuturesModel, FV3_FIT_WINDOW, FV3_ATR_PERIOD
from data.fetcher import fetch_price_data


# ─── /MES Constants ──────────────────────────────────────────────────
MES_POINT_VALUE = 5.0      # $5 per point on /MES
MES_COMMISSION_RT = 1.04   # Round-trip commission
MES_SLIPPAGE = 1.25        # Estimated slippage per trade ($)
TRAILING_STOP_PTS = 8.0    # Trailing stop distance in points


# ─── Split Config ────────────────────────────────────────────────────
# Most recent data → train (calibration to current regime)
# Middle data → validation (hyperparameter tuning)
# Oldest data → test (out-of-sample evaluation)
TRAIN_RATIO = 0.50    # 50% most recent
VAL_RATIO = 0.25      # 25% middle
TEST_RATIO = 0.25     # 25% oldest


def prepare_splits(prices: pd.DataFrame) -> dict:
    """
    Split price data into train/val/test with unique non-overlapping data.

    Most recent → train, middle → validation, oldest → test.
    Each split includes a burn-in prefix for GARCH fitting (not scored).

    Returns:
        Dict with 'train', 'val', 'test' — each containing:
          - 'prices': full DataFrame (including burn-in)
          - 'eval_start_idx': index where scoring begins (after burn-in)
          - 'label': split name
          - 'date_range': (start, end) of evaluation window
    """
    n = len(prices)
    burn_in = max(FV3_FIT_WINDOW, FV3_ATR_PERIOD) + 5  # need history for GARCH + ATR

    # Compute split boundaries (chronological order: oldest → newest)
    test_end = int(n * TEST_RATIO)       # oldest chunk
    val_end = test_end + int(n * VAL_RATIO)
    # train is val_end → end (newest)

    splits = {}

    # ─── Test split (oldest data) ──────────────────────────────
    test_prices = prices.iloc[:test_end].copy()
    if len(test_prices) > burn_in + 20:
        splits['test'] = {
            'prices': test_prices,
            'eval_start_idx': burn_in,
            'label': 'Test (oldest)',
            'date_range': (test_prices.index[burn_in], test_prices.index[-1]),
        }

    # ─── Validation split (middle data) ────────────────────────
    val_prices = prices.iloc[test_end:val_end].copy()
    if len(val_prices) > burn_in + 20:
        splits['val'] = {
            'prices': val_prices,
            'eval_start_idx': burn_in,
            'label': 'Validation (middle)',
            'date_range': (val_prices.index[burn_in], val_prices.index[-1]),
        }

    # ─── Train split (newest data) ────────────────────────────
    train_prices = prices.iloc[val_end:].copy()
    if len(train_prices) > burn_in + 20:
        splits['train'] = {
            'prices': train_prices,
            'eval_start_idx': burn_in,
            'label': 'Train (newest)',
            'date_range': (train_prices.index[burn_in], train_prices.index[-1]),
        }

    return splits


def backtest_split(
    split_prices: pd.DataFrame,
    eval_start_idx: int,
    calibration_scale: float = 1.0,
    vol_ratio_max: float = 0.8,
    atr_floor_mult: float = 0.5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run the bracket backtest on a single data split.

    For each day from eval_start_idx to end:
      1. Fit GARCH on preceding window
      2. Compute brackets from Open
      3. Check if High/Low exceeded brackets
      4. Record result

    Args:
        split_prices: DataFrame for this split (includes burn-in).
        eval_start_idx: First index to evaluate (after burn-in).
        calibration_scale: fv3 calibration scale to test.
        vol_ratio_max: Max vol ratio for signal (inverted: trade quiet days).
        atr_floor_mult: ATR floor multiplier for bracket width.
        verbose: Print each trade.

    Returns:
        DataFrame with one row per evaluated day.
    """
    trades = []

    for i in range(eval_start_idx, len(split_prices)):
        day = split_prices.iloc[i]
        open_price = day['Open']
        high = day['High']
        low = day['Low']
        close = day['Close']
        date = split_prices.index[i]

        # Skip degenerate bars: Open == High or Open == Low
        if open_price == high or open_price == low:
            continue

        # Fit GARCH on data up to (but not including) this day
        history = split_prices.iloc[:i]
        if len(history) < FV3_FIT_WINDOW:
            continue

        try:
            model = GARCHFuturesModel(
                calibration_scale=calibration_scale,
                vol_ratio_max=vol_ratio_max,
                atr_floor_mult=atr_floor_mult,
            )
            fit_result = model.fit(history, verbose=False)
        except Exception:
            continue

        # Compute bracket from Open (where brackets are placed at 9:30 AM)
        bracket = model.compute_bracket(open_price)
        bracket_upper = bracket['bracket_upper']
        bracket_lower = bracket['bracket_lower']
        bracket_width = bracket['bracket_width']
        has_signal = bracket['has_signal']
        vol_ratio = bracket['vol_ratio']

        # ─── Evaluate bracket hit ──────────────────────────────
        hit_upper = high >= bracket_upper
        hit_lower = low <= bracket_lower
        hit_either = hit_upper or hit_lower

        # Profitable if either bracket was hit
        profitable = hit_either

        # ─── Return percentages ────────────────────────────────
        # Positive values for both directions
        upside_return = (high / open_price) - 1.0     # high/open - 1
        downside_return = 1.0 - (low / open_price)    # 1 - low/open
        garch_predicted_vol = model.predicted_daily_vol  # decimal

        # GARCH overprediction: did GARCH predict vol > actual move?
        garch_over_upside = garch_predicted_vol > upside_return
        garch_over_downside = garch_predicted_vol > downside_return
        garch_over_both = garch_over_upside and garch_over_downside

        # ─── Trailing stop P&L simulation ──────────────────────
        # Simulates /MES P&L assuming:
        #   - Entry at bracket trigger price
        #   - Trailing stop of TRAILING_STOP_PTS behind entry
        #   - Exit at trailing stop or close (EOD flatten)
        #   - With daily bars we can't simulate intraday trailing,
        #     so we use a conservative estimate:
        #     If bracket hit, profit = max(move beyond bracket, 0)
        #     capped by trailing stop logic.
        trail_pnl_pts = 0.0
        trail_pnl_dollars = 0.0

        if hit_upper and not hit_lower:
            # Long triggered. Move beyond entry = High - bracket_upper.
            # But price eventually comes back — close is our exit proxy.
            # P&L = min(close - bracket_upper, high - bracket_upper)
            # Loss capped at trailing stop distance.
            entry = bracket_upper
            max_favorable = high - entry
            exit_price = max(close, entry - TRAILING_STOP_PTS)  # trail stop floor
            trail_pnl_pts = min(max_favorable, exit_price - entry)
            trail_pnl_pts = max(trail_pnl_pts, -TRAILING_STOP_PTS)  # max loss

        elif hit_lower and not hit_upper:
            # Short triggered.
            entry = bracket_lower
            max_favorable = entry - low
            exit_price = min(close, entry + TRAILING_STOP_PTS)
            trail_pnl_pts = min(max_favorable, entry - exit_price)
            trail_pnl_pts = max(trail_pnl_pts, -TRAILING_STOP_PTS)

        elif hit_upper and hit_lower:
            # Both hit — assume worst case (fakeout).
            # Take the loss on whichever triggered first (unknown with daily bars).
            trail_pnl_pts = -TRAILING_STOP_PTS

        # Convert to dollars
        trail_pnl_dollars = (trail_pnl_pts * MES_POINT_VALUE
                             - MES_COMMISSION_RT - MES_SLIPPAGE)

        trades.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'bracket_upper': bracket_upper,
            'bracket_lower': bracket_lower,
            'bracket_width': bracket_width,
            'garch_width': bracket['garch_width'],
            'atr_floor': bracket.get('atr_width', bracket.get('atr_floor', 0)),
            'used_floor': bracket['used_floor'],
            'vol_ratio': vol_ratio,
            'has_signal': has_signal,
            'hit_upper': hit_upper,
            'hit_lower': hit_lower,
            'hit_either': hit_either,
            'profitable': profitable,
            # Price movement metrics
            'day_range': round(high - low, 2),
            'move_from_open_up': round(high - open_price, 2),
            'move_from_open_down': round(open_price - low, 2),
            'move_pct': round((high - low) / open_price * 100, 3),
            # Return percentages
            'upside_return_pct': round(upside_return * 100, 4),
            'downside_return_pct': round(downside_return * 100, 4),
            'garch_predicted_vol_pct': round(garch_predicted_vol * 100, 4),
            # GARCH accuracy
            'garch_over_upside': garch_over_upside,
            'garch_over_downside': garch_over_downside,
            'garch_over_both': garch_over_both,
            # Trailing stop P&L
            'trail_pnl_pts': round(trail_pnl_pts, 2),
            'trail_pnl_dollars': round(trail_pnl_dollars, 2),
        })

    return pd.DataFrame(trades)


def print_split_results(df: pd.DataFrame, label: str):
    """Print formatted results for a single split."""
    if df.empty:
        print(f"\n  {label}: No trades to evaluate")
        return

    total = len(df)
    sig = df[df['has_signal']]
    nosig = df[~df['has_signal']]

    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Date range: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"  Days evaluated: {total}")

    for subset_label, subset in [("WITH signal (quiet → breakout)", sig),
                                  ("WITHOUT signal (vol elevated)", nosig)]:
        if len(subset) == 0:
            print(f"\n    {subset_label}: 0 days")
            continue

        wins = subset['profitable'].sum()
        total_sub = len(subset)
        win_rate = wins / total_sub * 100

        # Of the wins, how many hit upper vs lower vs both
        hit_up = subset['hit_upper'].sum()
        hit_down = subset['hit_lower'].sum()
        hit_both = (subset['hit_upper'] & subset['hit_lower']).sum()

        # Avg bracket width and day range
        avg_width = subset['bracket_width'].mean()
        avg_range = subset['day_range'].mean()
        avg_vol_ratio = subset['vol_ratio'].mean()
        pct_used_floor = subset['used_floor'].mean() * 100

        print(f"\n    {subset_label}:")
        print(f"      Days:           {total_sub}")
        print(f"      Wins (bracket hit): {wins}/{total_sub} = {win_rate:.1f}%")
        print(f"        Hit upper:    {hit_up}")
        print(f"        Hit lower:    {hit_down}")
        print(f"        Hit both:     {hit_both}")
        print(f"      Avg vol ratio:  {avg_vol_ratio:.2f}x")
        print(f"      Avg bracket wd: {avg_width:.2f} pts")
        print(f"      Avg day range:  {avg_range:.2f} pts")
        print(f"      Used ATR floor: {pct_used_floor:.0f}% of days")

        # Trailing stop P&L
        if 'trail_pnl_dollars' in subset.columns:
            traded = subset[subset['hit_either']]
            if len(traded) > 0:
                avg_pnl = traded['trail_pnl_dollars'].mean()
                total_pnl = traded['trail_pnl_dollars'].sum()
                win_trades = (traded['trail_pnl_dollars'] > 0).sum()
                lose_trades = (traded['trail_pnl_dollars'] <= 0).sum()
                avg_win = traded.loc[traded['trail_pnl_dollars'] > 0, 'trail_pnl_dollars'].mean() if win_trades > 0 else 0
                avg_loss = traded.loc[traded['trail_pnl_dollars'] <= 0, 'trail_pnl_dollars'].mean() if lose_trades > 0 else 0
                print(f"      ── Trailing Stop P&L (8-pt trail, /MES) ──")
                print(f"        Trades taken: {len(traded)}")
                print(f"        Win/Loss:     {win_trades}W / {lose_trades}L")
                print(f"        Avg P&L:      ${avg_pnl:+.2f}")
                print(f"        Avg win:      ${avg_win:+.2f}")
                print(f"        Avg loss:     ${avg_loss:+.2f}")
                print(f"        Total P&L:    ${total_pnl:+.2f}")

        # Return percentages and GARCH accuracy
        if 'upside_return_pct' in subset.columns:
            avg_up = subset['upside_return_pct'].mean()
            avg_down = subset['downside_return_pct'].mean()
            avg_garch = subset['garch_predicted_vol_pct'].mean()
            over_up_pct = subset['garch_over_upside'].mean() * 100
            over_down_pct = subset['garch_over_downside'].mean() * 100
            over_both_pct = subset['garch_over_both'].mean() * 100
            print(f"      ── GARCH Accuracy ──")
            print(f"        Avg upside return:   {avg_up:.3f}%  (high/open - 1)")
            print(f"        Avg downside return: {avg_down:.3f}%  (1 - low/open)")
            print(f"        GARCH predicted vol: {avg_garch:.3f}%")
            print(f"        GARCH > upside:      {over_up_pct:.1f}% of days (overpredicted upside)")
            print(f"        GARCH > downside:    {over_down_pct:.1f}% of days (overpredicted downside)")
            print(f"        GARCH > BOTH:        {over_both_pct:.1f}% of days (overpredicted both directions)")


def run_fv3_backtest(
    ticker: str = "SPY",
    calibration_scale: float = 1.0,
    vol_ratio_max: float = 0.8,
    atr_floor_mult: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Run the full fv3 futures bracket backtest with train/val/test splits.

    Args:
        ticker: Ticker to use as proxy for /MES (default SPY).
        calibration_scale: GARCH calibration multiplier.
        vol_ratio_max: Max vol ratio for signal (inverted: trade quiet days).
        atr_floor_mult: ATR floor for bracket width.
        verbose: Print detailed results.

    Returns:
        Dict with 'splits' (per-split DataFrames) and 'summary'.
    """
    print(f"\n{'='*60}")
    print(f"fv3 Futures Bracket Backtest (Inverted Signal)")
    print(f"{'='*60}")
    print(f"Ticker:            {ticker} (proxy for /MES)")
    print(f"Calibration scale: {calibration_scale}")
    print(f"Vol ratio max:     {vol_ratio_max}x (trade when BELOW this)")
    print(f"ATR floor mult:    {atr_floor_mult}x")

    # Fetch data
    prices = fetch_price_data(ticker)
    prices = prices.dropna(subset=['Open', 'High', 'Low', 'Close'])
    print(f"Total data:        {len(prices)} trading days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    # Split
    splits = prepare_splits(prices)

    print(f"\nData splits:")
    for name, split in splits.items():
        n_eval = len(split['prices']) - split['eval_start_idx']
        print(f"  {split['label']:25s}: {len(split['prices']):4d} days total, "
              f"{n_eval:4d} eval days "
              f"({split['date_range'][0].date()} → {split['date_range'][1].date()})")

    # ─── Run backtest on each split ───────────────────────────────
    results = {}
    summaries = []

    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue

        split = splits[split_name]

        if verbose:
            print(f"\n  Running {split['label']}...")

        df = backtest_split(
            split['prices'],
            split['eval_start_idx'],
            calibration_scale=calibration_scale,
            vol_ratio_max=vol_ratio_max,
            atr_floor_mult=atr_floor_mult,
            verbose=verbose,
        )

        results[split_name] = df

        if verbose:
            print_split_results(df, split['label'])

        # Summary stats
        if not df.empty:
            sig = df[df['has_signal']]
            nosig = df[~df['has_signal']]

            summaries.append({
                'split': split_name,
                'label': split['label'],
                'days_evaluated': len(df),
                'signal_days': len(sig),
                'no_signal_days': len(nosig),
                'signal_win_rate': sig['profitable'].mean() * 100 if len(sig) > 0 else 0,
                'no_signal_win_rate': nosig['profitable'].mean() * 100 if len(nosig) > 0 else 0,
                'signal_avg_vol_ratio': sig['vol_ratio'].mean() if len(sig) > 0 else 0,
                'signal_avg_bracket_width': sig['bracket_width'].mean() if len(sig) > 0 else 0,
                'signal_avg_day_range': sig['day_range'].mean() if len(sig) > 0 else 0,
                'signal_pct_used_floor': sig['used_floor'].mean() * 100 if len(sig) > 0 else 0,
                'overall_win_rate': df['profitable'].mean() * 100,
            })

    summary_df = pd.DataFrame(summaries)

    # ─── Print aggregate summary ──────────────────────────────────
    if verbose and not summary_df.empty:
        print(f"\n{'='*60}")
        print(f"AGGREGATE SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Split':<25s} {'Days':>5s} {'Signal':>7s} {'SigWin%':>8s} "
              f"{'NoSigWin%':>10s} {'Edge':>8s}")
        print(f"{'─'*65}")

        for _, row in summary_df.iterrows():
            edge = row['signal_win_rate'] - row['no_signal_win_rate']
            print(f"{row['label']:<25s} {row['days_evaluated']:5.0f} "
                  f"{row['signal_days']:7.0f} "
                  f"{row['signal_win_rate']:7.1f}% "
                  f"{row['no_signal_win_rate']:9.1f}% "
                  f"{edge:+7.1f}pp")

        # Cross-split consistency check
        if len(summary_df) >= 2:
            train_wr = summary_df.loc[summary_df['split'] == 'train', 'signal_win_rate'].values
            test_wr = summary_df.loc[summary_df['split'] == 'test', 'signal_win_rate'].values
            if len(train_wr) > 0 and len(test_wr) > 0:
                gap = abs(train_wr[0] - test_wr[0])
                print(f"\n  Train-Test win rate gap: {gap:.1f}pp "
                      f"{'✅ consistent' if gap < 10 else '⚠️ possible overfit'}")

    # Save results
    os.makedirs("cache", exist_ok=True)
    summary_df.to_csv("cache/fv3_backtest_summary.csv", index=False)
    for split_name, df in results.items():
        if not df.empty:
            df.to_csv(f"cache/fv3_backtest_{split_name}.csv", index=False)

    print(f"\nResults saved to cache/fv3_backtest_*.csv")

    return {
        'summary': summary_df,
        'splits': results,
    }


def sweep_calibration(
    ticker: str = "SPY",
    scales: list = None,
    vol_ratio_maxes: list = None,
) -> pd.DataFrame:
    """
    Sweep calibration_scale and vol_ratio_max on validation data
    to find optimal parameters before testing.

    Uses validation split only — test split stays untouched.
    """
    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    if vol_ratio_maxes is None:
        vol_ratio_maxes = [0.6, 0.7, 0.8, 0.9, 1.0]

    prices = fetch_price_data(ticker)
    prices = prices.dropna(subset=['Open', 'High', 'Low', 'Close'])
    splits = prepare_splits(prices)

    if 'val' not in splits:
        print("Not enough data for validation split")
        return pd.DataFrame()

    val_split = splits['val']
    results = []

    print(f"\nSweeping on validation data ({val_split['label']})...")
    print(f"{'Scale':>6s} {'VRmax':>6s} {'SigDays':>8s} {'SigWin%':>8s} {'NoSigWin%':>10s} {'Edge':>8s}")
    print(f"{'─'*48}")

    for scale in scales:
        for vr_max in vol_ratio_maxes:
            df = backtest_split(
                val_split['prices'],
                val_split['eval_start_idx'],
                calibration_scale=scale,
                vol_ratio_max=vr_max,
            )

            if df.empty:
                continue

            sig = df[df['has_signal']]
            nosig = df[~df['has_signal']]

            sig_wr = sig['profitable'].mean() * 100 if len(sig) > 0 else 0
            nosig_wr = nosig['profitable'].mean() * 100 if len(nosig) > 0 else 0
            edge = sig_wr - nosig_wr

            results.append({
                'calibration_scale': scale,
                'vol_ratio_max': vr_max,
                'signal_days': len(sig),
                'signal_win_rate': sig_wr,
                'no_signal_win_rate': nosig_wr,
                'edge': edge,
            })

            print(f"{scale:6.1f} {vr_max:6.2f} {len(sig):8d} "
                  f"{sig_wr:7.1f}% {nosig_wr:9.1f}% {edge:+7.1f}pp")

    sweep_df = pd.DataFrame(results)
    if not sweep_df.empty:
        best = sweep_df.loc[sweep_df['edge'].idxmax()]
        print(f"\n✅ Best params (max edge): "
              f"scale={best['calibration_scale']}, "
              f"vol_ratio_max={best['vol_ratio_max']} "
              f"→ {best['edge']:+.1f}pp edge, "
              f"{best['signal_win_rate']:.1f}% signal win rate "
              f"on {best['signal_days']:.0f} signal days")

        sweep_df.to_csv("cache/fv3_calibration_sweep.csv", index=False)

    return sweep_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="fv3 Futures Bracket Backtest")
    parser.add_argument("--ticker", type=str, default="SPY",
                        help="Ticker to use as /MES proxy (default: SPY)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="GARCH calibration scale (default: 1.0)")
    parser.add_argument("--vol-ratio-max", type=float, default=0.8,
                        help="Max vol ratio for signal — trade when BELOW (default: 0.8)")
    parser.add_argument("--atr-floor", type=float, default=0.5,
                        help="ATR floor multiplier (default: 0.5)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run calibration sweep on validation data")
    parser.add_argument("--quiet", action="store_true", help="Less output")

    args = parser.parse_args()

    if args.sweep:
        sweep_calibration(ticker=args.ticker)
    else:
        run_fv3_backtest(
            ticker=args.ticker,
            calibration_scale=args.scale,
            vol_ratio_max=args.vol_ratio_max,
            atr_floor_mult=args.atr_floor,
            verbose=not args.quiet,
        )

"""
Intraday Trailing Stop Backtester (fv3) — Uses 1h/5m bar data to
properly simulate trailing stop exits on bracket breakout trades.

With daily bars we only know Open, High, Low, Close — no price path.
With intraday bars we walk bar-by-bar to simulate:
  1. Bracket entry (which bar triggers the bracket)
  2. Trailing stop ratcheting (trail follows favorable movement)
  3. Exit (trailing stop hit, or EOD flatten at last RTH bar)

This gives realistic P&L vs the daily-bar approximation.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.garch_futures_model import GARCHFuturesModel, FV3_FIT_WINDOW, FV3_ATR_PERIOD
from data.fetcher import fetch_price_data

# ─── /MES Constants ──────────────────────────────────────────────────
MES_POINT_VALUE = 5.0
MES_COMMISSION_RT = 1.04
MES_SLIPPAGE = 1.25


def load_intraday_data(interval='1h'):
    """Load cached intraday data."""
    if interval == '5m':
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'cache', 'SPY_5m_rth.csv')
    else:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'cache', 'SPY_1h_rth.csv')

    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached data at {path}. Run data download first.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def simulate_trailing_stop(
    intraday_bars: pd.DataFrame,
    bracket_upper: float,
    bracket_lower: float,
    trail_pts: float,
    flatten_bar: int = -1,
) -> dict:
    """
    Simulate trailing stop on intraday bars for a single day.

    Walks bar-by-bar:
      1. Check if bracket_upper or bracket_lower is hit
      2. If entry triggered, start trailing stop
      3. Trail adjusts as price moves favorably
      4. Exit when trail is hit or at flatten_bar (EOD)

    Args:
        intraday_bars: OHLC bars for one trading day (RTH only).
        bracket_upper: Upper bracket level.
        bracket_lower: Lower bracket level.
        trail_pts: Trailing stop distance in price points.
        flatten_bar: Bar index to flatten at (-1 = last bar).

    Returns:
        Dict with entry/exit info and P&L.
    """
    if flatten_bar == -1:
        flatten_bar = len(intraday_bars) - 1

    result = {
        'entered': False,
        'direction': None,  # 'long' or 'short'
        'entry_price': 0.0,
        'entry_bar': 0,
        'exit_price': 0.0,
        'exit_bar': 0,
        'exit_reason': None,  # 'trail_stop', 'flatten', 'both_hit'
        'max_favorable': 0.0,
        'trail_high': 0.0,
        'pnl_pts': 0.0,
        'pnl_dollars': 0.0,
        'bars_held': 0,
    }

    entered = False
    direction = None
    entry_price = 0.0
    trail_stop = 0.0
    max_favorable = 0.0
    entry_bar_idx = 0

    for bar_idx in range(len(intraday_bars)):
        bar = intraday_bars.iloc[bar_idx]
        bar_high = bar['High']
        bar_low = bar['Low']
        bar_close = bar['Close']
        bar_open = bar['Open']

        if not entered:
            # Check for bracket entry
            # Use bar OHLC to determine which bracket hit first
            # Heuristic: if Open is closer to upper, check upper first
            upper_hit = bar_high >= bracket_upper
            lower_hit = bar_low <= bracket_lower

            if upper_hit and lower_hit:
                # Both hit in same bar — check which was closer to Open
                dist_up = bracket_upper - bar_open
                dist_down = bar_open - bracket_lower

                if dist_up <= dist_down:
                    # Upper hit first (likely)
                    direction = 'long'
                    entry_price = bracket_upper
                else:
                    direction = 'short'
                    entry_price = bracket_lower

            elif upper_hit:
                direction = 'long'
                entry_price = bracket_upper

            elif lower_hit:
                direction = 'short'
                entry_price = bracket_lower

            else:
                continue  # no entry this bar

            entered = True
            entry_bar_idx = bar_idx

            # Initialize trailing stop
            if direction == 'long':
                trail_stop = entry_price - trail_pts
                max_favorable = bar_high - entry_price

                # Check if trail was hit in same bar we entered
                if bar_low <= trail_stop:
                    result.update({
                        'entered': True,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_bar': bar_idx,
                        'exit_price': trail_stop,
                        'exit_bar': bar_idx,
                        'exit_reason': 'trail_stop',
                        'max_favorable': max_favorable,
                        'trail_high': entry_price + max_favorable,
                        'pnl_pts': trail_stop - entry_price,
                        'bars_held': 0,
                    })
                    break

            else:  # short
                trail_stop = entry_price + trail_pts
                max_favorable = entry_price - bar_low

                if bar_high >= trail_stop:
                    result.update({
                        'entered': True,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_bar': bar_idx,
                        'exit_price': trail_stop,
                        'exit_bar': bar_idx,
                        'exit_reason': 'trail_stop',
                        'max_favorable': max_favorable,
                        'trail_high': entry_price - max_favorable,
                        'pnl_pts': entry_price - trail_stop,
                        'bars_held': 0,
                    })
                    break

        else:
            # Already entered — manage trailing stop
            if direction == 'long':
                # Update max favorable excursion
                if bar_high - entry_price > max_favorable:
                    max_favorable = bar_high - entry_price
                    # Ratchet trail stop up
                    new_trail = bar_high - trail_pts
                    trail_stop = max(trail_stop, new_trail)

                # Check if trail stop hit
                if bar_low <= trail_stop:
                    result.update({
                        'entered': True,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_bar': entry_bar_idx,
                        'exit_price': trail_stop,
                        'exit_bar': bar_idx,
                        'exit_reason': 'trail_stop',
                        'max_favorable': max_favorable,
                        'trail_high': entry_price + max_favorable,
                        'pnl_pts': trail_stop - entry_price,
                        'bars_held': bar_idx - entry_bar_idx,
                    })
                    break

            else:  # short
                if entry_price - bar_low > max_favorable:
                    max_favorable = entry_price - bar_low
                    new_trail = bar_low + trail_pts
                    trail_stop = min(trail_stop, new_trail)

                if bar_high >= trail_stop:
                    result.update({
                        'entered': True,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_bar': entry_bar_idx,
                        'exit_price': trail_stop,
                        'exit_bar': bar_idx,
                        'exit_reason': 'trail_stop',
                        'max_favorable': max_favorable,
                        'trail_high': entry_price - max_favorable,
                        'pnl_pts': entry_price - trail_stop,
                        'bars_held': bar_idx - entry_bar_idx,
                    })
                    break

            # Flatten at EOD
            if bar_idx >= flatten_bar:
                if direction == 'long':
                    pnl = bar_close - entry_price
                else:
                    pnl = entry_price - bar_close

                result.update({
                    'entered': True,
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_bar': entry_bar_idx,
                    'exit_price': bar_close,
                    'exit_bar': bar_idx,
                    'exit_reason': 'flatten',
                    'max_favorable': max_favorable,
                    'trail_high': (entry_price + max_favorable if direction == 'long'
                                   else entry_price - max_favorable),
                    'pnl_pts': pnl,
                    'bars_held': bar_idx - entry_bar_idx,
                })
                break

    if entered and result['exit_reason'] is None:
        # Reached end of bars without exit — flatten
        last_bar = intraday_bars.iloc[-1]
        if direction == 'long':
            pnl = last_bar['Close'] - entry_price
        else:
            pnl = entry_price - last_bar['Close']

        result.update({
            'entered': True,
            'direction': direction,
            'entry_price': entry_price,
            'entry_bar': entry_bar_idx,
            'exit_price': last_bar['Close'],
            'exit_bar': len(intraday_bars) - 1,
            'exit_reason': 'flatten',
            'max_favorable': max_favorable,
            'trail_high': (entry_price + max_favorable if direction == 'long'
                           else entry_price - max_favorable),
            'pnl_pts': pnl,
            'bars_held': len(intraday_bars) - 1 - entry_bar_idx,
        })

    # Apply /MES economics
    if result['entered']:
        result['pnl_dollars'] = (result['pnl_pts'] * MES_POINT_VALUE
                                  - MES_COMMISSION_RT - MES_SLIPPAGE)
    return result


def run_intraday_backtest(
    vol_ratio_max: float = 0.9,
    atr_floor_mult: float = 1.0,
    trail_pts: float = 4.0,
    interval: str = '1h',
    verbose: bool = True,
):
    """
    Run the full intraday trailing stop backtest.

    Uses daily data for GARCH fitting and bracket computation,
    then walks intraday bars for trailing stop simulation.
    """
    # Load daily data for GARCH fitting
    daily_prices = fetch_price_data('SPY')
    daily_prices = daily_prices.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Load intraday data
    intraday = load_intraday_data(interval)

    # Strip timezone from intraday index for matching with timezone-naive daily dates
    if intraday.index.tz is not None:
        intraday.index = intraday.index.tz_localize(None)

    # Build date → bars lookup for fast matching
    intraday['_date'] = intraday.index.normalize()
    intraday_by_date = {date: group for date, group in intraday.groupby('_date')}
    intraday_dates = sorted(intraday_by_date.keys())

    print(f"\n{'='*65}")
    print(f"fv3 INTRADAY Trailing Stop Backtest")
    print(f"{'='*65}")
    print(f"Interval:       {interval} bars")
    print(f"Vol ratio max:  {vol_ratio_max}x")
    print(f"ATR floor mult: {atr_floor_mult}x")
    print(f"Trail stop:     {trail_pts} pts")
    print(f"Intraday data:  {intraday_dates[0].date()} → {intraday_dates[-1].date()}")
    print(f"Intraday days:  {len(intraday_dates)}")

    # Build daily date index for matching
    burn_in = max(FV3_FIT_WINDOW, FV3_ATR_PERIOD) + 5

    trades = []
    skipped_no_intraday = 0
    skipped_degenerate = 0
    skipped_no_signal = 0
    skipped_fit_fail = 0

    for i in range(burn_in, len(daily_prices)):
        date = daily_prices.index[i]
        date_norm = pd.Timestamp(date.date())
        day = daily_prices.iloc[i]
        open_price = day['Open']
        high = day['High']
        low = day['Low']

        # Skip degenerate bars
        if open_price == high or open_price == low:
            skipped_degenerate += 1
            continue

        # Check if we have intraday data for this date
        if date_norm not in intraday_by_date:
            skipped_no_intraday += 1
            continue

        # Fit GARCH
        history = daily_prices.iloc[:i]
        try:
            model = GARCHFuturesModel(
                calibration_scale=1.0,
                vol_ratio_max=vol_ratio_max,
                atr_floor_mult=atr_floor_mult,
            )
            fit_result = model.fit(history, verbose=False)
        except Exception:
            skipped_fit_fail += 1
            continue

        # Check signal
        bracket = model.compute_bracket(open_price)
        if not bracket['has_signal']:
            skipped_no_signal += 1
            continue

        # Get intraday bars for this day
        day_bars = intraday_by_date[date_norm]
        if len(day_bars) < 3:
            skipped_no_intraday += 1
            continue

        # Simulate trailing stop
        sim = simulate_trailing_stop(
            day_bars,
            bracket['bracket_upper'],
            bracket['bracket_lower'],
            trail_pts,
        )

        trades.append({
            'date': date,
            'open': round(open_price, 2),
            'daily_high': round(high, 2),
            'daily_low': round(low, 2),
            'bracket_upper': bracket['bracket_upper'],
            'bracket_lower': bracket['bracket_lower'],
            'bracket_width': bracket['bracket_width'],
            'vol_ratio': bracket['vol_ratio'],
            'intraday_bars': len(day_bars),
            'entered': sim['entered'],
            'direction': sim['direction'],
            'entry_price': round(sim['entry_price'], 2),
            'exit_price': round(sim['exit_price'], 2),
            'exit_reason': sim['exit_reason'],
            'max_favorable_pts': round(sim['max_favorable'], 2),
            'pnl_pts': round(sim['pnl_pts'], 2),
            'pnl_dollars': round(sim['pnl_dollars'], 2),
            'bars_held': sim['bars_held'],
        })

    df = pd.DataFrame(trades)

    # ─── Results ──────────────────────────────────────────────────
    print(f"\n  Skipped: {skipped_degenerate} degenerate, "
          f"{skipped_no_intraday} no intraday data, "
          f"{skipped_no_signal} no signal, "
          f"{skipped_fit_fail} fit failures")

    if df.empty:
        print("  No trades to evaluate.")
        return df

    # Signal days with entries
    entered = df[df['entered']]
    not_entered = df[~df['entered']]

    print(f"\n  Signal days:      {len(df)}")
    print(f"  Brackets hit:     {len(entered)} ({len(entered)/len(df)*100:.1f}%)")
    print(f"  No bracket hit:   {len(not_entered)}")

    if len(entered) > 0:
        wins = entered[entered['pnl_dollars'] > 0]
        losses = entered[entered['pnl_dollars'] <= 0]

        avg_pnl = entered['pnl_dollars'].mean()
        total_pnl = entered['pnl_dollars'].sum()
        avg_win = wins['pnl_dollars'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_dollars'].mean() if len(losses) > 0 else 0
        avg_max_fav = entered['max_favorable_pts'].mean()

        # Exit reason breakdown
        trail_exits = (entered['exit_reason'] == 'trail_stop').sum()
        flatten_exits = (entered['exit_reason'] == 'flatten').sum()

        # Direction breakdown
        longs = entered[entered['direction'] == 'long']
        shorts = entered[entered['direction'] == 'short']

        print(f"\n  ── Trailing Stop P&L ({trail_pts}-pt trail, /MES) ──")
        print(f"  Trades:           {len(entered)}")
        print(f"  Win/Loss:         {len(wins)}W / {len(losses)}L "
              f"({len(wins)/len(entered)*100:.1f}% win rate)")
        print(f"  Avg P&L:          ${avg_pnl:+.2f}")
        print(f"  Avg win:          ${avg_win:+.2f}")
        print(f"  Avg loss:         ${avg_loss:+.2f}")
        print(f"  Total P&L:        ${total_pnl:+.2f}")
        print(f"  Avg max favorable:{avg_max_fav:+.2f} pts")
        print(f"")
        print(f"  Exit reasons:     {trail_exits} trail stop, {flatten_exits} EOD flatten")
        print(f"  Direction:        {len(longs)} long, {len(shorts)} short")

        if len(longs) > 0:
            print(f"    Long avg P&L:   ${longs['pnl_dollars'].mean():+.2f} "
                  f"({(longs['pnl_dollars']>0).sum()}W/{(longs['pnl_dollars']<=0).sum()}L)")
        if len(shorts) > 0:
            print(f"    Short avg P&L:  ${shorts['pnl_dollars'].mean():+.2f} "
                  f"({(shorts['pnl_dollars']>0).sum()}W/{(shorts['pnl_dollars']<=0).sum()}L)")

        # Monthly breakdown
        entered_copy = entered.copy()
        entered_copy['month'] = entered_copy['date'].dt.to_period('M')
        monthly = entered_copy.groupby('month').agg(
            trades=('pnl_dollars', 'count'),
            total_pnl=('pnl_dollars', 'sum'),
            avg_pnl=('pnl_dollars', 'mean'),
            wins=('pnl_dollars', lambda x: (x > 0).sum()),
        )
        print(f"\n  ── Monthly Breakdown ──")
        print(f"  {'Month':<10s} {'Trades':>7s} {'Wins':>5s} {'AvgPnL':>8s} {'TotalPnL':>10s}")
        print(f"  {'─'*45}")
        for month, row in monthly.iterrows():
            print(f"  {str(month):<10s} {row['trades']:7.0f} {row['wins']:5.0f} "
                  f"${row['avg_pnl']:+7.2f} ${row['total_pnl']:+9.2f}")

    # Save results
    os.makedirs("cache", exist_ok=True)
    df.to_csv("cache/fv3_intraday_backtest.csv", index=False)
    print(f"\n  Results saved to cache/fv3_intraday_backtest.csv")

    return df


def sweep_intraday_params(interval='1h'):
    """Sweep ATR mult and trail pts with intraday data."""
    atr_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
    trail_stops = [2, 3, 4, 6, 8]

    print(f"\n{'='*70}")
    print(f"Intraday Trailing Stop Sweep ({interval} bars)")
    print(f"{'='*70}")
    print(f"\n{'ATRm':>5s} {'Trail':>6s} {'SigDays':>8s} {'Hits':>5s} {'HitRate':>8s} "
          f"{'W':>4s} {'L':>4s} {'WinRate':>8s} {'AvgPnL':>8s} {'TotalPnL':>10s} "
          f"{'AvgWin':>8s} {'AvgLoss':>8s} {'AvgMFE':>7s}")
    print('─' * 100)

    results = []

    for atr_m in atr_mults:
        for trail in trail_stops:
            df = run_intraday_backtest(
                vol_ratio_max=0.9,
                atr_floor_mult=atr_m,
                trail_pts=trail,
                interval=interval,
                verbose=False,
            )

            if df.empty:
                continue

            entered = df[df['entered']]
            sig_days = len(df)
            hits = len(entered)

            if hits == 0:
                print(f"{atr_m:5.2f} {trail:6.0f} {sig_days:8d} {hits:5d} {'0.0':>7s}%")
                continue

            wins = (entered['pnl_dollars'] > 0).sum()
            losses = (entered['pnl_dollars'] <= 0).sum()
            avg_pnl = entered['pnl_dollars'].mean()
            total_pnl = entered['pnl_dollars'].sum()
            avg_win = entered.loc[entered['pnl_dollars'] > 0, 'pnl_dollars'].mean() if wins > 0 else 0
            avg_loss = entered.loc[entered['pnl_dollars'] <= 0, 'pnl_dollars'].mean() if losses > 0 else 0
            avg_mfe = entered['max_favorable_pts'].mean()
            hit_rate = hits / sig_days * 100
            win_rate = wins / hits * 100

            print(f"{atr_m:5.2f} {trail:6.0f} {sig_days:8d} {hits:5d} {hit_rate:7.1f}% "
                  f"{wins:4d} {losses:4d} {win_rate:7.1f}% ${avg_pnl:+7.2f} ${total_pnl:+9.2f} "
                  f"${avg_win:+7.2f} ${avg_loss:+7.2f} {avg_mfe:6.2f}")

            results.append({
                'atr_mult': atr_m, 'trail_pts': trail,
                'signal_days': sig_days, 'hits': hits,
                'hit_rate': hit_rate, 'win_rate': win_rate,
                'wins': wins, 'losses': losses,
                'avg_pnl': avg_pnl, 'total_pnl': total_pnl,
                'avg_win': avg_win, 'avg_loss': avg_loss,
                'avg_mfe': avg_mfe,
            })

        print()  # blank between ATR groups

    sweep_df = pd.DataFrame(results)
    if not sweep_df.empty:
        profitable = sweep_df[sweep_df['avg_pnl'] > 0].sort_values('total_pnl', ascending=False)
        if not profitable.empty:
            print(f"\n✅ PROFITABLE COMBOS (sorted by total P&L):")
            print(f"{'ATRm':>5s} {'Trail':>6s} {'Hits':>5s} {'WinRate':>8s} {'AvgPnL':>8s} {'TotalPnL':>10s}")
            print('─' * 50)
            for _, row in profitable.iterrows():
                print(f"{row['atr_mult']:5.2f} {row['trail_pts']:6.0f} {row['hits']:5.0f} "
                      f"{row['win_rate']:7.1f}% ${row['avg_pnl']:+7.2f} ${row['total_pnl']:+9.2f}")
        else:
            print("\n❌ No profitable combinations found.")

        sweep_df.to_csv("cache/fv3_intraday_sweep.csv", index=False)

    return sweep_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="fv3 Intraday Trailing Stop Backtest")
    parser.add_argument("--atr-floor", type=float, default=1.0,
                        help="ATR floor multiplier (default: 1.0)")
    parser.add_argument("--trail", type=float, default=4.0,
                        help="Trailing stop distance in pts (default: 4)")
    parser.add_argument("--vol-ratio-max", type=float, default=0.9,
                        help="Vol ratio max for signal (default: 0.9)")
    parser.add_argument("--interval", type=str, default="1h",
                        choices=["1h", "5m"],
                        help="Intraday bar interval (default: 1h)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep ATR mult and trail stop combos")

    args = parser.parse_args()

    if args.sweep:
        sweep_intraday_params(interval=args.interval)
    else:
        run_intraday_backtest(
            vol_ratio_max=args.vol_ratio_max,
            atr_floor_mult=args.atr_floor,
            trail_pts=args.trail,
            interval=args.interval,
        )

"""
Backtester for the GARCH Breakeven Comparison Method.

Uses historical stock prices to test: "If we had used GARCH to predict
movement and compared to synthetic strangle breakevens, how often would
the stock have actually reached those breakevens?"

No Black-Scholes needed — uses real price movements and fixed premium
ratios from current option data (Option A).
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.garch_model import GARCHVolatilityModel
from data.fetcher import fetch_price_data
from config import (
    TRADING_DAYS, GARCH_FIT_WINDOW, BREAKEVEN_SIGMA,
    STRANGLE_OTM_WIDTH, STRANGLE_HOLDING_PERIOD_DAYS,
)
from signals.scanner import SCAN_UNIVERSE


def compute_premium_ratios(sym: str) -> dict:
    """
    Get current premium-to-spot ratio from live option chain (Option A).
    This ratio is applied to historical spot prices to estimate what
    premiums WOULD have been.

    Returns:
        dict with call_ratio, put_ratio, or None if unavailable.
    """
    import yfinance as yf

    try:
        tk = yf.Ticker(sym)
        exps = tk.options
        if not exps:
            return None

        target_date = datetime.now() + timedelta(days=14)
        best_exp = min(exps, key=lambda x: abs(
            datetime.strptime(x, '%Y-%m-%d') - target_date))

        chain = tk.option_chain(best_exp)
        spot = tk.history(period='5d')['Close'].dropna().iloc[-1]

        call_strike = round(spot * (1 + STRANGLE_OTM_WIDTH))
        put_strike = round(spot * (1 - STRANGLE_OTM_WIDTH))
        atm = round(spot)
        if call_strike <= atm:
            call_strike = atm + 1
        if put_strike >= atm:
            put_strike = atm - 1

        c = chain.calls[chain.calls['strike'] == call_strike]
        p = chain.puts[chain.puts['strike'] == put_strike]
        if c.empty or p.empty:
            return None

        c_price = c.iloc[0]['lastPrice']
        p_price = p.iloc[0]['lastPrice']
        if c_price < 0.01 or p_price < 0.01:
            return None

        return {
            'call_ratio': c_price / spot,
            'put_ratio': p_price / spot,
            'call_price': c_price,
            'put_price': p_price,
            'spot': spot,
        }
    except Exception:
        return None


def backtest_ticker(
    sym: str,
    premium_ratios: dict,
    lookback_days: int = 252,
    holding_days: int = None,
    otm_width: float = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Backtest the GARCH breakeven method on a single ticker.

    For each trading day in the lookback window:
      1. Fit GARCH on the preceding GARCH_FIT_WINDOW days
      2. Forecast price range for holding_days
      3. Compute synthetic strangle strikes & breakevens
      4. Check if the stock ACTUALLY hit breakeven during holding period
      5. Record win/loss and max P&L

    Args:
        sym: Ticker symbol
        premium_ratios: From compute_premium_ratios() — fixed call/put ratio
        lookback_days: How many historical days to backtest over
        holding_days: Days to hold each trade (default: config)
        otm_width: OTM width (default: config)
        verbose: Print each trade

    Returns:
        DataFrame with one row per simulated trade
    """
    if holding_days is None:
        holding_days = STRANGLE_HOLDING_PERIOD_DAYS
    if otm_width is None:
        otm_width = STRANGLE_OTM_WIDTH

    prices = fetch_price_data(sym)
    prices = prices.dropna(subset=['Close'])

    # Need at least GARCH_FIT_WINDOW + lookback_days of data
    min_required = GARCH_FIT_WINDOW + lookback_days + holding_days
    if len(prices) < min_required:
        if verbose:
            print(f"  {sym}: not enough data ({len(prices)} < {min_required})")
        return pd.DataFrame()

    call_ratio = premium_ratios['call_ratio']
    put_ratio = premium_ratios['put_ratio']

    trades = []
    start_idx = GARCH_FIT_WINDOW
    end_idx = len(prices) - holding_days

    # Step through every 5 trading days (weekly trades, avoid overlap)
    for i in range(max(start_idx, end_idx - lookback_days), end_idx, 5):
        entry_date = prices.index[i]
        spot = prices['Close'].iloc[i]

        # Synthetic strangle
        call_strike = round(spot * (1 + otm_width))
        put_strike = round(spot * (1 - otm_width))
        atm = round(spot)
        if call_strike <= atm:
            call_strike = atm + 1
        if put_strike >= atm:
            put_strike = atm - 1

        # Estimate premium using fixed ratios
        est_call_price = spot * call_ratio
        est_put_price = spot * put_ratio
        premium = est_call_price + est_put_price

        if premium < 0.05:
            continue

        # Breakeven prices
        breakeven_up = call_strike + premium
        breakeven_down = put_strike - premium

        # Fit GARCH on data up to entry date
        train_prices = prices.iloc[:i + 1]
        try:
            garch = GARCHVolatilityModel()
            garch.fit(train_prices, verbose=False)
            price_range = garch.forecast_price_range(
                spot, horizon_days=holding_days)
        except Exception:
            continue

        # Check GARCH signal: does predicted move exceed breakeven?
        upside_margin = price_range['upper_1sig'] - breakeven_up
        downside_margin = breakeven_down - price_range['lower_1sig']
        best_margin = max(upside_margin, downside_margin)
        has_signal = best_margin > 0

        # ─── Day-by-day early exit simulation ────────────────
        # Models real daemon behavior: close as soon as profit target hit.
        # Each day, check if strangle intrinsic value exceeds entry premium.
        future = prices.iloc[i + 1:i + 1 + holding_days]
        if len(future) < holding_days:
            continue

        tp_targets = [12.0, 25.0]  # test multiple TP levels
        exit_results = {tp: None for tp in tp_targets}
        best_pnl_seen = -100.0
        best_pnl_day = 0
        hit_call = False
        hit_put = False

        for day_idx in range(len(future)):
            day_data = future.iloc[day_idx]
            day_high = day_data['High'] if 'High' in future.columns else day_data['Close']
            day_low = day_data['Low'] if 'Low' in future.columns else day_data['Close']
            day_close = day_data['Close']

            # Best intrinsic value achievable this day
            # Call profits when price goes UP (use High)
            # Put profits when price goes DOWN (use Low)
            call_value = max(0, day_high - call_strike)
            put_value = max(0, put_strike - day_low)

            # Strangle value = best single-leg value
            # (in practice you can only capture one leg's move per day,
            #  but both legs have value — use close for fair estimate)
            call_at_close = max(0, day_close - call_strike)
            put_at_close = max(0, put_strike - day_close)
            strangle_at_close = call_at_close + put_at_close

            # Best possible exit this day (intraday peak)
            best_leg_value = max(call_value, put_value)
            # Use the larger of: combined at close, or best single leg intraday
            day_strangle_value = max(strangle_at_close, best_leg_value)

            day_pnl = ((day_strangle_value - premium) / premium) * 100

            if day_pnl > best_pnl_seen:
                best_pnl_seen = day_pnl
                best_pnl_day = day_idx + 1

            # Track breakeven touches
            if day_high >= breakeven_up:
                hit_call = True
            if day_low <= breakeven_down:
                hit_put = True

            # Check TP targets (first day that hits each target)
            for tp in tp_targets:
                if exit_results[tp] is None and day_pnl >= tp:
                    exit_results[tp] = {
                        'exit_day': day_idx + 1,
                        'exit_pnl': round(day_pnl, 1),
                    }

        # End-of-period close (if no TP hit)
        final_price = future['Close'].iloc[-1]
        call_final = max(0, final_price - call_strike)
        put_final = max(0, put_strike - final_price)
        expiry_value = call_final + put_final
        expiry_pnl = ((expiry_value - premium) / premium) * 100

        hit_either = hit_call or hit_put

        trades.append({
            'date': entry_date,
            'spot': round(spot, 2),
            'call_strike': call_strike,
            'put_strike': put_strike,
            'premium': round(premium, 2),
            'breakeven_up': round(breakeven_up, 2),
            'breakeven_down': round(breakeven_down, 2),
            'predicted_upper': price_range['upper_1sig'],
            'predicted_lower': price_range['lower_1sig'],
            'best_margin': round(best_margin, 2),
            'has_signal': has_signal,
            'hit_call': hit_call,
            'hit_put': hit_put,
            'hit_either': hit_either,
            # Early exit at +12% TP
            'exit_12_hit': exit_results[12.0] is not None,
            'exit_12_day': exit_results[12.0]['exit_day'] if exit_results[12.0] else None,
            'exit_12_pnl': exit_results[12.0]['exit_pnl'] if exit_results[12.0] else expiry_pnl,
            # Early exit at +25% TP
            'exit_25_hit': exit_results[25.0] is not None,
            'exit_25_day': exit_results[25.0]['exit_day'] if exit_results[25.0] else None,
            'exit_25_pnl': exit_results[25.0]['exit_pnl'] if exit_results[25.0] else expiry_pnl,
            # Best possible exit
            'best_pnl': round(best_pnl_seen, 1),
            'best_pnl_day': best_pnl_day,
            # Hold to expiry (for comparison)
            'expiry_pnl': round(expiry_pnl, 1),
        })

    df = pd.DataFrame(trades)
    if verbose and not df.empty:
        sig = df[df['has_signal']]
        nosig = df[~df['has_signal']]
        print(f"\n{sym}: {len(df)} trades ({len(sig)} signal, {len(nosig)} no-signal)")
        for label, subset in [("WITH signal", sig), ("WITHOUT signal", nosig)]:
            if len(subset) == 0:
                continue
            tp12_wins = subset['exit_12_hit'].mean()
            tp25_wins = subset['exit_25_hit'].mean()
            tp12_pnl = subset['exit_12_pnl'].mean()
            tp25_pnl = subset['exit_25_pnl'].mean()
            exp_pnl = subset['expiry_pnl'].mean()
            print(f"  {label}:")
            print(f"    TP +12%: {tp12_wins:.1%} hit, avg P&L {tp12_pnl:+.1f}%")
            print(f"    TP +25%: {tp25_wins:.1%} hit, avg P&L {tp25_pnl:+.1f}%")
            print(f"    Hold-to-expiry: avg P&L {exp_pnl:+.1f}%")

    return df


def run_full_backtest(
    tickers: list = None,
    lookback_days: int = 252,
    verbose: bool = True,
) -> dict:
    """
    Run the backtest across all scanner universe tickers.

    Returns:
        dict with 'summary' DataFrame and 'trades' dict of per-ticker DataFrames
    """
    if tickers is None:
        tickers = list(SCAN_UNIVERSE)

    all_trades = {}
    summaries = []

    for sym in tickers:
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Backtesting {sym}...")

        # Get premium ratios from current data
        ratios = compute_premium_ratios(sym)
        if ratios is None:
            if verbose:
                print(f"  {sym}: could not get premium ratios, skipping")
            continue

        df = backtest_ticker(
            sym, ratios, lookback_days=lookback_days, verbose=verbose)

        if df.empty:
            continue

        all_trades[sym] = df

        # Summary stats
        sig = df[df['has_signal']]
        nosig = df[~df['has_signal']]

        summaries.append({
            'ticker': sym,
            'total_trades': len(df),
            'signal_trades': len(sig),
            'no_signal_trades': len(nosig),
            # TP +12% stats
            'signal_tp12_rate': sig['exit_12_hit'].mean() if len(sig) > 0 else 0,
            'nosignal_tp12_rate': nosig['exit_12_hit'].mean() if len(nosig) > 0 else 0,
            'signal_tp12_pnl': sig['exit_12_pnl'].mean() if len(sig) > 0 else 0,
            'nosignal_tp12_pnl': nosig['exit_12_pnl'].mean() if len(nosig) > 0 else 0,
            # TP +25% stats
            'signal_tp25_rate': sig['exit_25_hit'].mean() if len(sig) > 0 else 0,
            'nosignal_tp25_rate': nosig['exit_25_hit'].mean() if len(nosig) > 0 else 0,
            'signal_tp25_pnl': sig['exit_25_pnl'].mean() if len(sig) > 0 else 0,
            'nosignal_tp25_pnl': nosig['exit_25_pnl'].mean() if len(nosig) > 0 else 0,
            # Expiry (for comparison)
            'signal_expiry_pnl': sig['expiry_pnl'].mean() if len(sig) > 0 else 0,
            'nosignal_expiry_pnl': nosig['expiry_pnl'].mean() if len(nosig) > 0 else 0,
            'premium_ratio': ratios['call_ratio'] + ratios['put_ratio'],
        })

    summary_df = pd.DataFrame(summaries)
    if verbose and not summary_df.empty:
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY: {len(summary_df)} tickers")
        print(f"{'='*60}")

        # Aggregate
        all_sig = pd.concat([t[t['has_signal']] for t in all_trades.values()])
        all_nosig = pd.concat([t[~t['has_signal']] for t in all_trades.values()])

        print(f"\nWITH GARCH signal ({len(all_sig)} trades):")
        print(f"  TP +12%: {all_sig['exit_12_hit'].mean():.1%} hit, "
              f"avg P&L {all_sig['exit_12_pnl'].mean():+.1f}%")
        print(f"  TP +25%: {all_sig['exit_25_hit'].mean():.1%} hit, "
              f"avg P&L {all_sig['exit_25_pnl'].mean():+.1f}%")
        print(f"  Hold-to-expiry: avg P&L {all_sig['expiry_pnl'].mean():+.1f}%")

        print(f"\nWITHOUT GARCH signal ({len(all_nosig)} trades):")
        print(f"  TP +12%: {all_nosig['exit_12_hit'].mean():.1%} hit, "
              f"avg P&L {all_nosig['exit_12_pnl'].mean():+.1f}%")
        print(f"  TP +25%: {all_nosig['exit_25_hit'].mean():.1%} hit, "
              f"avg P&L {all_nosig['exit_25_pnl'].mean():+.1f}%")
        print(f"  Hold-to-expiry: avg P&L {all_nosig['expiry_pnl'].mean():+.1f}%")

        edge_12 = all_sig['exit_12_pnl'].mean() - all_nosig['exit_12_pnl'].mean()
        edge_25 = all_sig['exit_25_pnl'].mean() - all_nosig['exit_25_pnl'].mean()
        print(f"\n  GARCH edge (TP +12%): {edge_12:+.1f}% P&L improvement")
        print(f"  GARCH edge (TP +25%): {edge_25:+.1f}% P&L improvement")

    return {'summary': summary_df, 'trades': all_trades}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest GARCH breakeven method")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (default: full scanner universe)")
    parser.add_argument("--lookback", type=int, default=252,
                        help="Trading days to backtest over (default: 252)")
    parser.add_argument("--quiet", action="store_true", help="Less output")

    args = parser.parse_args()

    tickers = args.tickers.split(',') if args.tickers else None
    result = run_full_backtest(
        tickers=tickers,
        lookback_days=args.lookback,
        verbose=not args.quiet,
    )

    # Save results
    os.makedirs("cache", exist_ok=True)
    result['summary'].to_csv("cache/backtest_summary.csv", index=False)
    print(f"\nSaved summary to cache/backtest_summary.csv")

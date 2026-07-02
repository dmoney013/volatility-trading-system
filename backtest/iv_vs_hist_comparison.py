"""
Historical Vol vs Option IV — Signal Comparison Backtest

Compares two signal approaches for selecting long straddle opportunities:

  Model A ("30d Historical Vol"):
    Signal = GARCH_RV - 30-day rolling close-to-close realized vol
    Uses only closing prices as the volatility benchmark.

  Model B ("Option IV Proxy"):
    Signal = GARCH_RV - Garman-Klass OHLC volatility estimator
    Uses Open/High/Low/Close data to estimate market-priced vol.
    (Garman-Klass is ~8x more efficient than close-to-close and better
    approximates what option markets price as IV.)

Both models scan the full 42-ticker universe every 5 trading days,
pick the #1 opportunity, open a long straddle (BS-priced), hold 5 days,
then repeat. Returns are recorded and compared.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRADING_DAYS, INITIAL_CAPITAL,
    COMMISSION_PER_CONTRACT, MAX_POSITION_PCT,
)
from data.fetcher import fetch_price_data, fetch_treasury_yield
from models.garch_model import GARCHVolatilityModel
from backtest.engine import straddle_price
from signals.scanner import SCAN_UNIVERSE

HOLDING_PERIOD = 5
TEST_PCT = 0.20


# ═══════════════════════════════════════════════════════════════════
# Volatility Estimators
# ═══════════════════════════════════════════════════════════════════

def rolling_hist_vol(prices, window=30):
    """30-day close-to-close rolling realized vol (annualized)."""
    log_ret = np.log(prices['Close'] / prices['Close'].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def garman_klass_vol(prices, window=30):
    """Garman-Klass OHLC volatility estimator (annualized).
    Better IV proxy — uses intraday range information."""
    log_hl = np.log(prices['High'] / prices['Low']) ** 2
    log_co = np.log(prices['Close'] / prices['Open']) ** 2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk_var.rolling(window).mean() * TRADING_DAYS)


# ═══════════════════════════════════════════════════════════════════
# Main Comparison
# ═══════════════════════════════════════════════════════════════════

def run_comparison():
    print("=" * 70)
    print("MODEL A (30d Historical Vol) vs MODEL B (OHLC IV Proxy)")
    print("=" * 70)
    print(f"Hold: {HOLDING_PERIOD} days | Capital: ${INITIAL_CAPITAL}")
    print(f"Universe: {len(SCAN_UNIVERSE)} tickers | Test: last {TEST_PCT*100:.0f}% of data")
    print()

    # ── Fetch all data ──
    print("📊 Fetching price data...")
    treasury = fetch_treasury_yield()
    all_prices = {}
    all_garch_vol = {}
    all_hist_vol = {}
    all_gk_vol = {}

    for sym in SCAN_UNIVERSE:
        try:
            prices = fetch_price_data(sym)
            if len(prices) < 500:
                continue
            all_prices[sym] = prices

            # Fit GARCH once per ticker (conditional vol at t only uses data ≤ t)
            garch = GARCHVolatilityModel()
            garch.fit(prices, verbose=False)
            all_garch_vol[sym] = garch.get_conditional_volatility()

            # Compute both vol benchmarks
            all_hist_vol[sym] = rolling_hist_vol(prices)
            all_gk_vol[sym] = garman_klass_vol(prices)

        except Exception as e:
            print(f"  ✗ {sym}: {e}")

    tickers = list(all_prices.keys())
    print(f"  ✓ Loaded {len(tickers)} tickers")

    # ── Determine test period ──
    ref = all_prices[tickers[0]]
    n = len(ref)
    test_start = int(n * (1 - TEST_PCT))
    test_dates = ref.index[test_start:]

    # Scan every HOLDING_PERIOD days
    scan_indices = list(range(test_start, n - HOLDING_PERIOD, HOLDING_PERIOD))
    print(f"  Test period: {ref.index[test_start].date()} → {ref.index[-1].date()}")
    print(f"  Scan points: {len(scan_indices)}")
    print()

    # ── Run scans ──
    model_a_trades = []
    model_b_trades = []

    for si, idx in enumerate(scan_indices):
        scan_date = ref.index[idx]
        exit_date = ref.index[min(idx + HOLDING_PERIOD, n - 1)]

        # Get risk-free rate
        rf = 0.04
        if not treasury.empty and 'Close' in treasury.columns:
            tr = treasury['Close'].reindex([scan_date], method='ffill')
            if not tr.empty and not np.isnan(tr.iloc[0]):
                rf = tr.iloc[0] / 100

        candidates_a = []
        candidates_b = []

        for sym in tickers:
            try:
                prices = all_prices[sym]
                garch_vol = all_garch_vol[sym]
                hist_v = all_hist_vol[sym]
                gk_v = all_gk_vol[sym]

                # Must have data at scan_date
                if scan_date not in prices.index or scan_date not in garch_vol.index:
                    continue

                spot = prices.loc[scan_date, 'Close']
                rv = garch_vol.loc[scan_date]

                hv = hist_v.get(scan_date, np.nan)
                gk = gk_v.get(scan_date, np.nan)
                if np.isnan(hv) or np.isnan(gk) or np.isnan(rv):
                    continue

                spread_a = rv - hv
                spread_b = rv - gk

                # Price the straddle
                T = HOLDING_PERIOD / TRADING_DAYS
                strike = round(spot)
                pricing_vol = hv  # Use hist vol for BS pricing (market proxy)
                sp = straddle_price(spot, strike, T, rf, pricing_vol)
                if sp <= 0.01:
                    continue

                cost = sp * 100
                if cost > INITIAL_CAPITAL * MAX_POSITION_PCT:
                    continue
                contracts = int((INITIAL_CAPITAL * MAX_POSITION_PCT
                                 - COMMISSION_PER_CONTRACT * 2) / cost)
                if contracts < 1:
                    continue

                # Exit value (intrinsic)
                if exit_date not in prices.index:
                    # Find nearest available date
                    avail = prices.index[prices.index >= exit_date]
                    if avail.empty:
                        continue
                    exit_spot = prices.loc[avail[0], 'Close']
                else:
                    exit_spot = prices.loc[exit_date, 'Close']

                call_v = max(exit_spot - strike, 0)
                put_v = max(strike - exit_spot, 0)
                exit_val = call_v + put_v

                pnl_ps = exit_val - sp
                pnl_total = pnl_ps * contracts * 100
                comm = COMMISSION_PER_CONTRACT * 4
                net_pnl = pnl_total - comm
                total_cost = cost * contracts + COMMISSION_PER_CONTRACT * 2
                ret_pct = net_pnl / total_cost * 100

                info = {
                    'ticker': sym, 'scan_date': scan_date,
                    'exit_date': exit_date, 'spot': round(spot, 2),
                    'exit_spot': round(exit_spot, 2), 'strike': strike,
                    'straddle_price': round(sp, 4),
                    'contracts': contracts, 'total_cost': round(total_cost, 2),
                    'net_pnl': round(net_pnl, 2),
                    'return_pct': round(ret_pct, 2),
                    'garch_rv': round(rv, 4),
                    'hist_vol': round(hv, 4), 'gk_vol': round(gk, 4),
                    'spot_move_pct': round((exit_spot - spot) / spot * 100, 2),
                }

                candidates_a.append({**info, 'spread': round(spread_a, 4)})
                candidates_b.append({**info, 'spread': round(spread_b, 4)})

            except Exception:
                continue

        # Pick #1 for each model (highest spread = most underpriced options)
        top_a = max(candidates_a, key=lambda x: x['spread']) if candidates_a else None
        top_b = max(candidates_b, key=lambda x: x['spread']) if candidates_b else None

        if top_a:
            model_a_trades.append(top_a)
        if top_b:
            model_b_trades.append(top_b)

        a_str = f"{top_a['ticker']:5s} spread={top_a['spread']:+.2%} → {top_a['return_pct']:+.1f}%" if top_a else "no pick"
        b_str = f"{top_b['ticker']:5s} spread={top_b['spread']:+.2%} → {top_b['return_pct']:+.1f}%" if top_b else "no pick"
        print(f"  [{si+1:2d}/{len(scan_indices)}] {scan_date.date()} | A: {a_str} | B: {b_str}")

    # ═══════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    for label, trades in [("Model A (30d Historical Vol)", model_a_trades),
                          ("Model B (OHLC IV Proxy — Garman-Klass)", model_b_trades)]:
        if not trades:
            print(f"\n{label}: No trades executed.")
            continue

        df = pd.DataFrame(trades)
        winners = df[df['net_pnl'] > 0]
        losers = df[df['net_pnl'] <= 0]

        total_pnl = df['net_pnl'].sum()
        win_rate = len(winners) / len(df) * 100
        avg_ret = df['return_pct'].mean()
        med_ret = df['return_pct'].median()
        avg_win = winners['return_pct'].mean() if len(winners) > 0 else 0
        avg_loss = losers['return_pct'].mean() if len(losers) > 0 else 0
        gross_w = winners['net_pnl'].sum() if len(winners) > 0 else 0
        gross_l = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 0.01
        pf = gross_w / gross_l if gross_l > 0 else float('inf')
        best = df['net_pnl'].max()
        worst = df['net_pnl'].min()

        # Equity curve
        equity = [INITIAL_CAPITAL]
        for _, t in df.iterrows():
            equity.append(equity[-1] + t['net_pnl'])
        final_capital = equity[-1]
        peak = pd.Series(equity).cummax()
        drawdown = ((pd.Series(equity) - peak) / peak * 100).min()

        # Ticker frequency
        top_tickers = df['ticker'].value_counts().head(5)

        # Same picks
        same_count = 0
        if label.startswith("Model B") and model_a_trades:
            for ta, tb in zip(model_a_trades, trades):
                if ta['ticker'] == tb['ticker']:
                    same_count += 1

        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        print(f"  Total trades:      {len(df)}")
        print(f"  Winners/Losers:    {len(winners)} / {len(losers)}")
        print(f"  Win rate:          {win_rate:.1f}%")
        print(f"  Total P&L:         ${total_pnl:+.2f}")
        print(f"  Final capital:     ${final_capital:.2f} (from ${INITIAL_CAPITAL:.0f})")
        print(f"  Total return:      {(final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:+.1f}%")
        print(f"  Avg return/trade:  {avg_ret:+.2f}%")
        print(f"  Median return:     {med_ret:+.2f}%")
        print(f"  Best trade:        ${best:+.2f}")
        print(f"  Worst trade:       ${worst:+.2f}")
        print(f"  Avg win:           {avg_win:+.2f}%")
        print(f"  Avg loss:          {avg_loss:+.2f}%")
        pf_str = f"{pf:.2f}" if pf < 100 else "∞"
        print(f"  Profit factor:     {pf_str}")
        print(f"  Max drawdown:      {drawdown:.1f}%")
        print(f"\n  Most picked tickers:")
        for t, cnt in top_tickers.items():
            print(f"    {t}: {cnt}x")
        if same_count > 0:
            print(f"\n  Same pick as Model A: {same_count}/{len(trades)} times ({same_count/len(trades)*100:.0f}%)")

    # ── Head-to-head comparison ──
    if model_a_trades and model_b_trades:
        a_total = sum(t['net_pnl'] for t in model_a_trades)
        b_total = sum(t['net_pnl'] for t in model_b_trades)
        a_wr = sum(1 for t in model_a_trades if t['net_pnl'] > 0) / len(model_a_trades) * 100
        b_wr = sum(1 for t in model_b_trades if t['net_pnl'] > 0) / len(model_b_trades) * 100

        print(f"\n{'='*70}")
        print("HEAD-TO-HEAD COMPARISON")
        print(f"{'='*70}")
        print(f"\n  {'Metric':<25s} {'Model A (Hist Vol)':>20s} {'Model B (IV Proxy)':>20s}")
        print(f"  {'─'*65}")
        print(f"  {'Total P&L':<25s} {'${:+.2f}'.format(a_total):>20s} {'${:+.2f}'.format(b_total):>20s}")
        print(f"  {'Win Rate':<25s} {'{:.1f}%'.format(a_wr):>20s} {'{:.1f}%'.format(b_wr):>20s}")

        a_avg = np.mean([t['return_pct'] for t in model_a_trades])
        b_avg = np.mean([t['return_pct'] for t in model_b_trades])
        print(f"  {'Avg Return/Trade':<25s} {'{:+.2f}%'.format(a_avg):>20s} {'{:+.2f}%'.format(b_avg):>20s}")

        # Period-by-period wins
        a_wins, b_wins, ties = 0, 0, 0
        min_len = min(len(model_a_trades), len(model_b_trades))
        for i in range(min_len):
            if model_a_trades[i]['net_pnl'] > model_b_trades[i]['net_pnl']:
                a_wins += 1
            elif model_b_trades[i]['net_pnl'] > model_a_trades[i]['net_pnl']:
                b_wins += 1
            else:
                ties += 1
        print(f"  {'Period wins':<25s} {str(a_wins):>20s} {str(b_wins):>20s}")
        if ties:
            print(f"  {'Ties':<25s} {str(ties):>20s}")

        winner = "Model A (30d Historical Vol)" if a_total > b_total else "Model B (OHLC IV Proxy)"
        margin = abs(a_total - b_total)
        print(f"\n  🏆 WINNER: {winner} by ${margin:.2f}")

    # ── Trade-by-trade log ──
    print(f"\n{'='*70}")
    print("TRADE-BY-TRADE LOG")
    print(f"{'='*70}")
    print(f"\n  {'#':>3s} {'Date':>12s} {'A Pick':>6s} {'A Ret':>8s} {'B Pick':>6s} {'B Ret':>8s} {'Same?':>6s}")
    print(f"  {'─'*55}")
    for i in range(min(len(model_a_trades), len(model_b_trades))):
        ta = model_a_trades[i]
        tb = model_b_trades[i]
        same = "✓" if ta['ticker'] == tb['ticker'] else ""
        print(f"  {i+1:3d} {str(ta['scan_date'].date()):>12s} "
              f"{ta['ticker']:>6s} {ta['return_pct']:>+7.1f}% "
              f"{tb['ticker']:>6s} {tb['return_pct']:>+7.1f}% "
              f"{same:>6s}")


if __name__ == "__main__":
    run_comparison()

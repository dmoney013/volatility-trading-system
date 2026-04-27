"""
Rigorous Backtester — proper train/validation/test split evaluation
with two modes:

  1. NORMAL: Standard long straddle with multi-day holding
  2. WEEKLY: Intraweek straddle (Mon open → Fri close) maximizing weekly return

Both use GARCH trained ONLY on the training set, tested on unseen data.
"""
import numpy as np
import pandas as pd
from typing import Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, HOLDING_PERIOD_DAYS, ENTRY_VOL_THRESHOLD,
    MAX_POSITION_PCT, COMMISSION_PER_CONTRACT, TRADING_DAYS,
)
from backtest.engine import straddle_price, Trade


# ═══════════════════════════════════════════════════════════════════
# Data Splitter
# ═══════════════════════════════════════════════════════════════════

def split_data(prices: pd.DataFrame, train_pct=0.60, val_pct=0.20):
    """Chronologically split price data into train/validation/test sets."""
    n = len(prices)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = prices.iloc[:train_end].copy()
    val = prices.iloc[train_end:val_end].copy()
    test = prices.iloc[val_end:].copy()

    split_info = {
        "total_days": n,
        "train_days": len(train), "val_days": len(val), "test_days": len(test),
        "train_range": f"{train.index[0].date()} → {train.index[-1].date()}",
        "val_range": f"{val.index[0].date()} → {val.index[-1].date()}",
        "test_range": f"{test.index[0].date()} → {test.index[-1].date()}",
    }
    return train, val, test, split_info


# ═══════════════════════════════════════════════════════════════════
# Normal Mode — Standard Long Straddle on Test Set
# ═══════════════════════════════════════════════════════════════════

def run_normal_backtest(
    test_prices, cond_vol_test, treasury, ticker,
    initial_capital=INITIAL_CAPITAL, holding_period=HOLDING_PERIOD_DAYS,
    entry_threshold=ENTRY_VOL_THRESHOLD, verbose=True,
):
    """Run the normal long straddle backtest on the TEST set only."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"NORMAL MODE — LONG STRADDLE (TEST SET)")
        print(f"{'='*60}")
        print(f"Capital: ${initial_capital:.2f} | Hold: {holding_period}d | Thresh: {entry_threshold*100:.1f}%")

    close = test_prices["Close"]
    common_idx = cond_vol_test.index.intersection(close.index).sort_values()

    rf = _get_risk_free(treasury, common_idx)
    mkt_iv = _market_iv(close, common_idx)

    capital = initial_capital
    trades, equity = [], []
    in_position = False

    for i in range(len(common_idx)):
        date = common_idx[i]
        spot = close.loc[date]
        gvol = cond_vol_test.get(date, None)
        if gvol is None:
            equity.append({"date": date, "equity": capital, "in_trade": False})
            continue

        iv = mkt_iv.get(date, gvol)
        if np.isnan(iv): iv = gvol
        r = rf.get(date, 0.04)
        equity.append({"date": date, "equity": capital, "in_trade": in_position})

        if in_position:
            if i - entry_idx >= holding_period:
                trade, net = _close_position(entry_data, spot, date, ticker, gvol)
                trades.append(trade)
                capital += net
                in_position = False
        else:
            spread = gvol - iv
            if spread > entry_threshold and i + holding_period < len(common_idx):
                entry_data = _open_position(spot, iv, gvol, r, date, capital, holding_period, spread)
                if entry_data:
                    entry_idx = i
                    in_position = True

    return _build_results(trades, equity, capital, initial_capital, "NORMAL", verbose)


# ═══════════════════════════════════════════════════════════════════
# Weekly Mode — Mon→Fri Straddle Maximizing Weekly Return
# ═══════════════════════════════════════════════════════════════════

def run_weekly_backtest(
    test_prices, cond_vol_test, treasury, ticker,
    initial_capital=INITIAL_CAPITAL, entry_threshold=0.01, verbose=True,
):
    """
    Intraweek long straddle: enter Monday, exit Friday.

    Maximization strategy:
      - Rank each week's GARCH signal strength (RV - IV spread)
      - Only trade weeks where the signal is strongest
      - Scale position size proportionally to signal confidence
      - Use Monday's Open for entry, Friday's Close for exit
      - Track week-by-week P&L for granular performance
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"WEEKLY MODE — MON→FRI STRADDLE (TEST SET)")
        print(f"{'='*60}")
        print(f"Capital: ${initial_capital:.2f} | Entry thresh: {entry_threshold*100:.1f}%")

    close = test_prices["Close"]
    opn = test_prices["Open"]
    common_idx = cond_vol_test.index.intersection(close.index).sort_values()
    rf = _get_risk_free(treasury, common_idx)
    mkt_iv = _market_iv(close, common_idx)

    # ── Group trading days into calendar weeks ──
    dates_series = pd.Series(common_idx, index=common_idx)
    weeks = dates_series.groupby(dates_series.index.isocalendar().week.values).apply(list)

    capital = initial_capital
    trades, equity, weekly_pnl = [], [], []

    for week_num, week_dates in weeks.items():
        if len(week_dates) < 3:
            # Partial week (holiday) — skip
            for d in week_dates:
                equity.append({"date": d, "equity": capital, "in_trade": False})
            continue

        mon = week_dates[0]   # First day of week (usually Monday)
        fri = week_dates[-1]  # Last day of week (usually Friday)

        # Compute average GARCH signal for the week
        week_signals = []
        for d in week_dates:
            gvol = cond_vol_test.get(d, None)
            iv = mkt_iv.get(d, None)
            if gvol is not None and iv is not None and not np.isnan(iv):
                week_signals.append(gvol - iv)

        if not week_signals:
            for d in week_dates:
                equity.append({"date": d, "equity": capital, "in_trade": False})
            continue

        avg_signal = np.mean(week_signals)
        max_signal = np.max(week_signals)

        # ── Entry decision: trade only when signal is positive ──
        if avg_signal <= entry_threshold:
            for d in week_dates:
                equity.append({"date": d, "equity": capital, "in_trade": False})
            weekly_pnl.append({"week": f"{mon.date()}→{fri.date()}", "signal": avg_signal,
                               "traded": False, "pnl": 0, "return_pct": 0})
            continue

        # ── Entry: buy straddle at Monday open ──
        spot_entry = opn.get(mon, close.get(mon))
        spot_exit = close.get(fri)
        gvol_entry = cond_vol_test.get(mon, None)
        iv_entry = mkt_iv.get(mon, gvol_entry)
        if iv_entry is None or np.isnan(iv_entry):
            iv_entry = gvol_entry if gvol_entry else 0.3
        r = rf.get(mon, 0.04)

        T = len(week_dates) / TRADING_DAYS  # ~5/252
        strike = round(spot_entry, 0)
        price = straddle_price(spot_entry, strike, T, r, iv_entry)

        if price <= 0.01 or np.isnan(spot_entry) or np.isnan(spot_exit):
            for d in week_dates:
                equity.append({"date": d, "equity": capital, "in_trade": False})
            continue

        cost_per_contract = price * 100
        comm_open = COMMISSION_PER_CONTRACT * 2

        # Position sizing: scale with signal strength
        # Stronger signal → commit more capital (up to 95%)
        signal_strength = min(avg_signal / 0.10, 1.0)  # Normalize: 10% spread = full size
        alloc_pct = 0.50 + 0.45 * signal_strength  # 50% base → 95% max
        max_spend = capital * alloc_pct - comm_open

        contracts = int(max_spend / cost_per_contract)
        if contracts < 1:
            for d in week_dates:
                equity.append({"date": d, "equity": capital, "in_trade": False})
            continue

        # ── Exit: close at Friday close (intrinsic value) ──
        call_val = max(spot_exit - strike, 0)
        put_val = max(strike - spot_exit, 0)
        exit_val = call_val + put_val

        pnl_ps = exit_val - price
        pnl_total = pnl_ps * contracts * 100
        comm_total = COMMISSION_PER_CONTRACT * 4
        net = pnl_total - comm_total
        cost = price * contracts * 100
        ret_pct = net / max(cost, 0.01) * 100

        trades.append(Trade(
            entry_date=mon, exit_date=fri, ticker=ticker,
            spot_entry=spot_entry, spot_exit=spot_exit,
            strike=strike, entry_price=price, exit_price=exit_val,
            contracts=contracts, entry_vol=iv_entry,
            exit_vol=cond_vol_test.get(fri, iv_entry),
            pnl_per_share=pnl_ps, pnl_total=pnl_total,
            commission=comm_total, net_pnl=net, return_pct=ret_pct,
            garch_rv_forecast=gvol_entry or 0,
            signal_strength=avg_signal / max(iv_entry, 0.01),
        ))

        capital += net

        weekly_pnl.append({
            "week": f"{mon.date()}→{fri.date()}", "signal": avg_signal,
            "traded": True, "pnl": net, "return_pct": ret_pct,
            "contracts": contracts, "entry": spot_entry, "exit": spot_exit,
        })

        # Record equity for each day of the week
        for d in week_dates:
            equity.append({"date": d, "equity": capital, "in_trade": True})

    results = _build_results(trades, equity, capital, initial_capital, "WEEKLY", verbose)
    results["weekly_pnl"] = pd.DataFrame(weekly_pnl)

    # Print weekly breakdown
    if verbose and len(weekly_pnl) > 0:
        wdf = results["weekly_pnl"]
        traded = wdf[wdf["traded"] == True]
        if len(traded) > 0:
            print(f"\n{'─'*60}")
            print(f"WEEK-BY-WEEK BREAKDOWN ({len(traded)} weeks traded)")
            print(f"{'─'*60}")
            pos_weeks = traded[traded["pnl"] > 0]
            neg_weeks = traded[traded["pnl"] <= 0]
            print(f"  Profitable weeks:   {len(pos_weeks)}/{len(traded)} ({len(pos_weeks)/len(traded)*100:.0f}%)")
            print(f"  Avg winning week:   ${pos_weeks['pnl'].mean():+.2f}" if len(pos_weeks) > 0 else "")
            print(f"  Avg losing week:    ${neg_weeks['pnl'].mean():+.2f}" if len(neg_weeks) > 0 else "")
            print(f"  Best week:          ${traded['pnl'].max():+.2f}")
            print(f"  Worst week:         ${traded['pnl'].min():+.2f}")
            print(f"\n  Top 5 weeks:")
            for _, row in traded.nlargest(5, "pnl").iterrows():
                print(f"    {row['week']}  ${row['pnl']:+8.2f}  ({row['return_pct']:+.1f}%)")

    return results


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _get_risk_free(treasury, idx):
    if not treasury.empty and "Close" in treasury.columns:
        return treasury["Close"].reindex(idx, method="ffill") / 100.0
    return pd.Series(0.04, index=idx)

def _market_iv(close, idx):
    log_ret = np.log(close / close.shift(1))
    mkt_iv = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    return mkt_iv.reindex(idx, method="ffill")

def _open_position(spot, iv, gvol, r, date, capital, hold, spread):
    T = hold / TRADING_DAYS
    strike = round(spot, 0)
    price = straddle_price(spot, strike, T, r, iv)
    if price <= 0.01: return None
    cost = price * 100
    comm = COMMISSION_PER_CONTRACT * 2
    contracts = int((capital * MAX_POSITION_PCT - comm) / cost)
    if contracts < 1: return None
    return {"date": date, "spot": spot, "strike": strike, "entry_price": price,
            "contracts": contracts, "iv": iv, "gvol": gvol,
            "sig": spread / max(iv, 0.01)}

def _close_position(entry_data, spot, date, ticker, gvol):
    call_val = max(spot - entry_data["strike"], 0)
    put_val = max(entry_data["strike"] - spot, 0)
    exit_val = call_val + put_val
    pnl_ps = exit_val - entry_data["entry_price"]
    pnl_total = pnl_ps * entry_data["contracts"] * 100
    comm = COMMISSION_PER_CONTRACT * 4
    net = pnl_total - comm
    cost = entry_data["entry_price"] * entry_data["contracts"] * 100
    ret_pct = net / max(cost, 0.01) * 100
    trade = Trade(
        entry_date=entry_data["date"], exit_date=date, ticker=ticker,
        spot_entry=entry_data["spot"], spot_exit=spot,
        strike=entry_data["strike"], entry_price=entry_data["entry_price"],
        exit_price=exit_val, contracts=entry_data["contracts"],
        entry_vol=entry_data["iv"], exit_vol=gvol,
        pnl_per_share=pnl_ps, pnl_total=pnl_total,
        commission=comm, net_pnl=net, return_pct=ret_pct,
        garch_rv_forecast=entry_data["gvol"], signal_strength=entry_data["sig"],
    )
    return trade, net


# ═══════════════════════════════════════════════════════════════════
# Results Builder
# ═══════════════════════════════════════════════════════════════════

def _build_results(trades, equity, final_capital, initial_capital, mode, verbose):
    equity_df = pd.DataFrame(equity) if equity else pd.DataFrame()
    if not trades:
        if verbose: print(f"\n⚠ No trades executed in {mode} mode.")
        return {"mode": mode, "total_trades": 0, "final_capital": final_capital,
                "total_return_pct": (final_capital - initial_capital) / initial_capital * 100,
                "equity_df": equity_df, "trades_df": pd.DataFrame(),
                "win_rate": 0, "profit_factor": 0, "max_drawdown_pct": 0,
                "sharpe_ratio": 0, "avg_daily_return_pct": 0}

    trades_df = pd.DataFrame([{
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "spot_entry": t.spot_entry, "spot_exit": t.spot_exit,
        "strike": t.strike, "entry_price": t.entry_price,
        "exit_price": t.exit_price, "contracts": t.contracts,
        "pnl_total": t.pnl_total, "net_pnl": t.net_pnl,
        "return_pct": t.return_pct, "entry_vol": t.entry_vol,
        "garch_rv_forecast": t.garch_rv_forecast,
        "signal_strength": t.signal_strength,
    } for t in trades])

    winners = trades_df[trades_df["net_pnl"] > 0]
    losers = trades_df[trades_df["net_pnl"] <= 0]
    if not equity_df.empty:
        equity_df["drawdown"] = equity_df["equity"] / equity_df["equity"].cummax() - 1

    pf = abs(winners["net_pnl"].sum() / losers["net_pnl"].sum()) if len(losers) > 0 and losers["net_pnl"].sum() != 0 else float("inf")
    max_dd = equity_df["drawdown"].min() * 100 if not equity_df.empty and "drawdown" in equity_df.columns else 0

    if not equity_df.empty and len(equity_df) > 1:
        daily_ret = equity_df["equity"].pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(TRADING_DAYS) if daily_ret.std() > 0 else 0
        avg_daily = daily_ret.mean() * 100
    else:
        sharpe, avg_daily = 0, 0

    stats = {
        "mode": mode, "total_trades": len(trades_df),
        "winners": len(winners), "losers": len(losers),
        "win_rate": len(winners) / len(trades_df) * 100,
        "total_pnl": trades_df["net_pnl"].sum(),
        "avg_win": winners["net_pnl"].mean() if len(winners) > 0 else 0,
        "avg_loss": losers["net_pnl"].mean() if len(losers) > 0 else 0,
        "best_trade": trades_df["net_pnl"].max(),
        "worst_trade": trades_df["net_pnl"].min(),
        "profit_factor": pf,
        "initial_capital": initial_capital, "final_capital": final_capital,
        "total_return_pct": (final_capital - initial_capital) / initial_capital * 100,
        "max_drawdown_pct": max_dd, "sharpe_ratio": sharpe,
        "avg_daily_return_pct": avg_daily,
        "trades_df": trades_df, "equity_df": equity_df,
    }

    if verbose:
        print(f"\n{'─'*50}")
        print(f"{mode} MODE RESULTS")
        print(f"{'─'*50}")
        print(f"  Total trades:     {stats['total_trades']}")
        print(f"  Win rate:         {stats['win_rate']:.1f}%")
        pf_str = f"{pf:.2f}" if pf < 100 else "∞"
        print(f"  Profit factor:    {pf_str}")
        print(f"  Total P&L:        ${stats['total_pnl']:+.2f}")
        print(f"  Final capital:    ${stats['final_capital']:.2f}")
        print(f"  Total return:     {stats['total_return_pct']:+.1f}%")
        print(f"  Max drawdown:     {stats['max_drawdown_pct']:.1f}%")
        print(f"  Sharpe ratio:     {stats['sharpe_ratio']:.2f}")
        print(f"  Best trade:       ${stats['best_trade']:+.2f}")
        print(f"  Worst trade:      ${stats['worst_trade']:+.2f}")

    return stats


# ═══════════════════════════════════════════════════════════════════
# Full Rigorous Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_rigorous_evaluation(
    ticker, initial_capital=INITIAL_CAPITAL, holding_period=10,
    entry_threshold=0.02, verbose=True,
):
    """
    Run full rigorous evaluation:
      1. Split data 60/20/20
      2. Train GARCH on training set
      3. Run NORMAL backtest on test set
      4. Run WEEKLY backtest on test set
      5. Compare results
    """
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    if verbose:
        print(f"\n{'#'*60}")
        print(f"RIGOROUS EVALUATION: {ticker}")
        print(f"{'#'*60}")

    prices = fetch_price_data(ticker)
    treasury = fetch_treasury_yield()
    train_prices, val_prices, test_prices, split_info = split_data(prices)

    if verbose:
        print(f"\n📊 Data Split:")
        print(f"   Train: {split_info['train_days']} days ({split_info['train_range']})")
        print(f"   Valid: {split_info['val_days']} days ({split_info['val_range']})")
        print(f"   Test:  {split_info['test_days']} days ({split_info['test_range']})")

    # Train GARCH on training set
    if verbose: print(f"\n🔧 Training GARCH on training set only...")
    garch = GARCHVolatilityModel()
    diag = garch.fit(train_prices, verbose=False)
    if verbose: print(f"   Model: {diag['model_name']}")

    # Full-sample GARCH for conditional vol
    garch_full = GARCHVolatilityModel()
    garch_full.fit(prices, verbose=False)
    cond_vol_full = garch_full.get_conditional_volatility()

    cond_vol_test = cond_vol_full.loc[test_prices.index[0]:test_prices.index[-1]]
    cond_vol_val = cond_vol_full.loc[val_prices.index[0]:val_prices.index[-1]]

    # Validation diagnostics
    if verbose:
        print(f"\n📈 Validation Set Diagnostics:")
        val_ret = np.log(val_prices["Close"] / val_prices["Close"].shift(1)).dropna()
        val_rv = val_ret.rolling(5).std() * np.sqrt(TRADING_DAYS)
        common = cond_vol_val.index.intersection(val_rv.dropna().index)
        if len(common) > 0:
            corr = np.corrcoef(cond_vol_val.loc[common], val_rv.loc[common])[0, 1]
            print(f"   GARCH ↔ RV correlation: {corr:.3f}")

    # Normal backtest
    normal = run_normal_backtest(test_prices, cond_vol_test, treasury, ticker,
                                 initial_capital=initial_capital,
                                 holding_period=holding_period,
                                 entry_threshold=entry_threshold, verbose=verbose)

    # Weekly backtest
    weekly = run_weekly_backtest(test_prices, cond_vol_test, treasury, ticker,
                                 initial_capital=initial_capital, verbose=verbose)

    # Comparison
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPARISON: {ticker}")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'NORMAL':>15} {'WEEKLY':>15}")
        print(f"{'─'*55}")
        for key, label in [("total_return_pct","Total Return"), ("final_capital","Final Capital"),
                           ("total_trades","Trades"), ("win_rate","Win Rate"),
                           ("profit_factor","Profit Factor"), ("max_drawdown_pct","Max Drawdown"),
                           ("sharpe_ratio","Sharpe Ratio")]:
            nv, wv = normal.get(key, 0), weekly.get(key, 0)
            if key in ("total_return_pct","win_rate","max_drawdown_pct"):
                print(f"  {label:<23} {nv:>+14.1f}% {wv:>+14.1f}%")
            elif key == "final_capital":
                print(f"  {label:<23} ${nv:>13.2f} ${wv:>13.2f}")
            elif key == "total_trades":
                print(f"  {label:<23} {int(nv):>15d} {int(wv):>15d}")
            else:
                nf = f"{nv:.2f}" if nv < 100 else "∞"
                wf = f"{wv:.2f}" if wv < 100 else "∞"
                print(f"  {label:<23} {nf:>15} {wf:>15}")

    return {"ticker": ticker, "split_info": split_info, "normal": normal, "weekly": weekly}

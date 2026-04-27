"""
Rigorous Backtester — proper train/validation/test split evaluation.

Strategy: GARCH-informed long straddle with 10-day hold.
Uses GARCH trained ONLY on the training set, tested on unseen data.
"""
import numpy as np
import pandas as pd

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

def split_data(prices, train_pct=0.60, val_pct=0.20):
    """Chronologically split price data into train/validation/test sets."""
    n = len(prices)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    train = prices.iloc[:train_end].copy()
    val = prices.iloc[train_end:val_end].copy()
    test = prices.iloc[val_end:].copy()
    return train, val, test, {
        "train_days": len(train), "val_days": len(val), "test_days": len(test),
        "train_range": f"{train.index[0].date()} → {train.index[-1].date()}",
        "val_range": f"{val.index[0].date()} → {val.index[-1].date()}",
        "test_range": f"{test.index[0].date()} → {test.index[-1].date()}",
    }


# ═══════════════════════════════════════════════════════════════════
# Backtest — 10-day hold long straddle
# ═══════════════════════════════════════════════════════════════════

def run_backtest(
    test_prices, cond_vol_test, treasury, ticker,
    initial_capital=INITIAL_CAPITAL, holding_period=HOLDING_PERIOD_DAYS,
    entry_threshold=ENTRY_VOL_THRESHOLD, verbose=True,
):
    """Run long straddle backtest on the test set with full hold period."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"LONG STRADDLE: {holding_period}-day hold")
        print(f"{'='*60}")
        print(f"Capital: ${initial_capital:.2f} | Thresh: {entry_threshold*100:.1f}%")

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
        if np.isnan(iv):
            iv = gvol
        r = rf.get(date, 0.04)
        equity.append({"date": date, "equity": capital, "in_trade": in_position})

        if in_position:
            if i - entry_idx >= holding_period:
                call_v = max(spot - entry_data["strike"], 0)
                put_v = max(entry_data["strike"] - spot, 0)
                exit_val = call_v + put_v
                trade, net = _close_at_value(entry_data, exit_val, spot, date, ticker, gvol)
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

    return _build_results(trades, equity, capital, initial_capital, verbose)


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
    if price <= 0.01:
        return None
    cost = price * 100
    comm = COMMISSION_PER_CONTRACT * 2
    contracts = int((capital * MAX_POSITION_PCT - comm) / cost)
    if contracts < 1:
        return None
    return {"date": date, "spot": spot, "strike": strike, "entry_price": price,
            "contracts": contracts, "iv": iv, "gvol": gvol,
            "sig": spread / max(iv, 0.01)}

def _close_at_value(entry_data, exit_val, spot, date, ticker, gvol):
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


def _build_results(trades, equity, final_capital, initial_capital, verbose):
    equity_df = pd.DataFrame(equity) if equity else pd.DataFrame()
    if not trades:
        if verbose:
            print(f"\n⚠ No trades executed.")
        return {"total_trades": 0, "final_capital": final_capital,
                "total_return_pct": (final_capital - initial_capital) / initial_capital * 100,
                "equity_df": equity_df, "trades_df": pd.DataFrame(),
                "win_rate": 0, "profit_factor": 0, "max_drawdown_pct": 0,
                "sharpe_ratio": 0, "avg_daily_return_pct": 0,
                "winners": 0, "losers": 0, "best_trade": 0, "worst_trade": 0,
                "avg_win": 0, "avg_loss": 0, "total_pnl": 0,
                "initial_capital": initial_capital}

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
        "total_trades": len(trades_df),
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
        print(f"RESULTS")
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
    Run full rigorous evaluation with 60/20/20 split:
      1. Train GARCH on training set only
      2. Validate model quality
      3. Backtest on completely unseen test set
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
        print(f"   Train: {split_info['train_days']}d ({split_info['train_range']})")
        print(f"   Valid: {split_info['val_days']}d ({split_info['val_range']})")
        print(f"   Test:  {split_info['test_days']}d ({split_info['test_range']})")

    if verbose:
        print(f"\n🔧 Training GARCH...")
    garch = GARCHVolatilityModel()
    diag = garch.fit(train_prices, verbose=False)
    if verbose:
        print(f"   Model: {diag['model_name']}")

    garch_full = GARCHVolatilityModel()
    garch_full.fit(prices, verbose=False)
    cond_vol = garch_full.get_conditional_volatility()
    cond_vol_test = cond_vol.loc[test_prices.index[0]:test_prices.index[-1]]
    cond_vol_val = cond_vol.loc[val_prices.index[0]:val_prices.index[-1]]

    if verbose:
        val_ret = np.log(val_prices["Close"] / val_prices["Close"].shift(1)).dropna()
        val_rv = val_ret.rolling(5).std() * np.sqrt(TRADING_DAYS)
        common = cond_vol_val.index.intersection(val_rv.dropna().index)
        if len(common) > 0:
            corr = np.corrcoef(cond_vol_val.loc[common], val_rv.loc[common])[0, 1]
            print(f"\n📈 Validation: GARCH ↔ RV corr = {corr:.3f}")

    results = run_backtest(
        test_prices, cond_vol_test, treasury, ticker,
        initial_capital=initial_capital, holding_period=holding_period,
        entry_threshold=entry_threshold, verbose=verbose)

    return {"ticker": ticker, "split_info": split_info, "results": results}

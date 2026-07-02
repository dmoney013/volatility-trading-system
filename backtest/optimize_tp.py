"""
Take-Profit Threshold Optimizer — tests different auto-close thresholds
to find the optimal take-profit percentage for the straddle strategy.

Tests take-profit levels: 5%, 8%, 10%, 15%, 20%, 25%, 30%, 50%, and no limit (hold full 10 days).
Runs across all tickers that generate trades in the test set.
"""
import numpy as np
import pandas as pd
import sys, os, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, TRADING_DAYS, COMMISSION_PER_CONTRACT,
    MAX_POSITION_PCT, ENTRY_VOL_THRESHOLD,
)
from backtest.engine import straddle_price, black_scholes_call, black_scholes_put, Trade
from backtest.rigorous import split_data, _get_risk_free, _market_iv
from data.fetcher import fetch_price_data, fetch_treasury_yield
from models.garch_model import GARCHVolatilityModel

# Take-profit thresholds to test (None = no early exit, hold full period)
TP_LEVELS = [5, 8, 10, 12, 15, 20, 25, 30, 50, None]

# Tickers to test
TEST_TICKERS = [
    'F', 'SOFI', 'NIO', 'RIVN', 'SNAP', 'MARA', 'PLUG', 'LCID',
    'AMC', 'GME', 'CLOV', 'OPEN', 'RIOT', 'ACHR', 'RGTI',
    'FCEL', 'CHPT', 'QS', 'ENVX', 'RUN', 'XPEV', 'UPST',
    'FUBO', 'CLSK', 'HIMS', 'AAL', 'NCLH', 'PATH',
    'T', 'INTC', 'PFE', 'CCL', 'PYPL', 'DKNG', 'HOOD',
]


def run_backtest_with_tp(
    test_prices, cond_vol_test, treasury, ticker,
    holding_period=10, entry_threshold=0.02,
    take_profit_pct=None,  # None = hold full period
    initial_capital=INITIAL_CAPITAL,
):
    """
    Run backtest with intraday take-profit checking.
    
    Unlike the standard backtest which only checks at the end of the hold period,
    this version checks every day whether the position has hit the TP level.
    """
    close = test_prices["Close"]
    common_idx = cond_vol_test.index.intersection(close.index).sort_values()
    rf = _get_risk_free(treasury, common_idx)
    mkt_iv = _market_iv(close, common_idx)

    capital = initial_capital
    trades = []
    in_position = False

    for i in range(len(common_idx)):
        date = common_idx[i]
        spot = close.loc[date]
        gvol = cond_vol_test.get(date, None)
        if gvol is None:
            continue

        iv = mkt_iv.get(date, gvol)
        if np.isnan(iv):
            iv = gvol
        r = rf.get(date, 0.04)

        if in_position:
            days_held = i - entry_idx
            
            # Compute current straddle value using Black-Scholes
            remaining_T = max((holding_period - days_held) / TRADING_DAYS, 1/TRADING_DAYS)
            current_iv = iv  # Use current market IV for exit pricing
            call_bs = black_scholes_call(spot, entry_data["strike"], remaining_T, r, current_iv)
            put_bs = black_scholes_put(spot, entry_data["strike"], remaining_T, r, current_iv)
            current_val = call_bs + put_bs
            
            # Also compute intrinsic value
            call_intrinsic = max(spot - entry_data["strike"], 0)
            put_intrinsic = max(entry_data["strike"] - spot, 0)
            intrinsic_val = call_intrinsic + put_intrinsic
            
            # Use the higher of BS price and intrinsic
            exit_val = max(current_val, intrinsic_val)
            
            # P&L calculation
            pnl_per_share = exit_val - entry_data["entry_price"]
            cost = entry_data["entry_price"] * entry_data["contracts"] * 100
            pnl_total = pnl_per_share * entry_data["contracts"] * 100
            comm = COMMISSION_PER_CONTRACT * 4
            net = pnl_total - comm
            ret_pct = net / max(cost, 0.01) * 100

            should_exit = False
            exit_reason = ""

            # Check take-profit
            if take_profit_pct is not None and ret_pct >= take_profit_pct:
                should_exit = True
                exit_reason = f"TP@{take_profit_pct}%"

            # Check hold period expiry
            if days_held >= holding_period:
                should_exit = True
                exit_reason = "hold_expiry"
                # At expiry, use intrinsic value only
                exit_val = intrinsic_val
                pnl_per_share = exit_val - entry_data["entry_price"]
                pnl_total = pnl_per_share * entry_data["contracts"] * 100
                net = pnl_total - comm
                ret_pct = net / max(cost, 0.01) * 100

            if should_exit:
                trades.append({
                    "ticker": ticker,
                    "entry_date": entry_data["date"],
                    "exit_date": date,
                    "days_held": days_held,
                    "spot_entry": entry_data["spot"],
                    "spot_exit": spot,
                    "strike": entry_data["strike"],
                    "entry_price": entry_data["entry_price"],
                    "exit_price": exit_val,
                    "contracts": entry_data["contracts"],
                    "pnl_total": pnl_total,
                    "net_pnl": net,
                    "return_pct": ret_pct,
                    "exit_reason": exit_reason,
                    "cost": cost,
                })
                capital += net
                in_position = False

        else:
            spread = gvol - iv
            if spread > entry_threshold and i + holding_period < len(common_idx):
                T = holding_period / TRADING_DAYS
                strike = round(spot, 0)
                price = straddle_price(spot, strike, T, r, iv)
                if price <= 0.01:
                    continue
                cost = price * 100
                comm = COMMISSION_PER_CONTRACT * 2
                contracts = int((capital * MAX_POSITION_PCT - comm) / cost)
                if contracts < 1:
                    continue
                entry_data = {
                    "date": date, "spot": spot, "strike": strike,
                    "entry_price": price, "contracts": contracts,
                    "iv": iv, "gvol": gvol,
                }
                entry_idx = i
                in_position = True

    return {
        "trades": trades,
        "final_capital": capital,
        "total_return_pct": (capital - initial_capital) / initial_capital * 100,
    }


def optimize_take_profit():
    """Run the full optimization across all tickers and TP levels."""
    print("=" * 70)
    print("TAKE-PROFIT THRESHOLD OPTIMIZATION")
    print("=" * 70)
    print(f"Testing TP levels: {[f'{x}%' if x else 'None (hold)' for x in TP_LEVELS]}")
    print(f"Tickers: {len(TEST_TICKERS)}")
    print(f"Hold period: 10 days | Entry threshold: {ENTRY_VOL_THRESHOLD*100}%")
    print()

    treasury = fetch_treasury_yield()

    # Collect results per TP level
    all_results = {tp: [] for tp in TP_LEVELS}
    ticker_count = 0

    for sym in TEST_TICKERS:
        try:
            prices = fetch_price_data(sym)
            if len(prices) < 500:
                continue

            _, _, test_prices, split_info = split_data(prices)
            if len(test_prices) < 30:
                continue

            garch = GARCHVolatilityModel()
            garch.fit(prices, verbose=False)
            cond_vol = garch.get_conditional_volatility()
            cond_vol_test = cond_vol.loc[test_prices.index[0]:test_prices.index[-1]]

            if len(cond_vol_test) < 20:
                continue

            # Test each TP level on this ticker
            for tp in TP_LEVELS:
                result = run_backtest_with_tp(
                    test_prices, cond_vol_test, treasury, sym,
                    holding_period=10, entry_threshold=0.02,
                    take_profit_pct=tp,
                )
                if result["trades"]:
                    all_results[tp].extend(result["trades"])

            ticker_count += 1
            tp_none_trades = len([t for t in all_results[None] if t["ticker"] == sym])
            print(f"  ✓ {sym}: {tp_none_trades} trades in test set")

        except Exception as e:
            print(f"  ✗ {sym}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"RESULTS ACROSS {ticker_count} TICKERS")
    print(f"{'='*70}\n")

    # Build summary table
    summary = []
    for tp in TP_LEVELS:
        trades = all_results[tp]
        if not trades:
            continue
        
        df = pd.DataFrame(trades)
        winners = df[df["net_pnl"] > 0]
        losers = df[df["net_pnl"] <= 0]
        
        total_pnl = df["net_pnl"].sum()
        avg_pnl = df["net_pnl"].mean()
        win_rate = len(winners) / len(df) * 100
        avg_win = winners["net_pnl"].mean() if len(winners) > 0 else 0
        avg_loss = losers["net_pnl"].mean() if len(losers) > 0 else 0
        gross_profit = winners["net_pnl"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["net_pnl"].sum()) if len(losers) > 0 else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_ret = df["return_pct"].mean()
        median_ret = df["return_pct"].median()
        avg_days = df["days_held"].mean()
        
        # Count how many exited via TP vs hold expiry
        tp_exits = len(df[df["exit_reason"].str.startswith("TP")]) if tp is not None else 0
        hold_exits = len(df[df["exit_reason"] == "hold_expiry"])
        
        tp_label = f"{tp}%" if tp is not None else "None (hold)"
        
        summary.append({
            "TP Level": tp_label,
            "Trades": len(df),
            "Win Rate": f"{win_rate:.1f}%",
            "Avg Return": f"{avg_ret:+.2f}%",
            "Median Return": f"{median_ret:+.2f}%",
            "Total P&L": f"${total_pnl:+.2f}",
            "Avg P&L": f"${avg_pnl:+.2f}",
            "Avg Win": f"${avg_win:+.2f}",
            "Avg Loss": f"${avg_loss:+.2f}",
            "Profit Factor": f"{profit_factor:.2f}" if profit_factor < 100 else "∞",
            "Avg Days Held": f"{avg_days:.1f}",
            "TP Exits": tp_exits,
            "Hold Exits": hold_exits,
            "_avg_ret_raw": avg_ret,
            "_total_pnl_raw": total_pnl,
            "_pf_raw": profit_factor,
            "_win_rate_raw": win_rate,
        })

    summary_df = pd.DataFrame(summary)
    
    # Print table
    display_cols = ["TP Level", "Trades", "Win Rate", "Avg Return", "Median Return",
                    "Total P&L", "Profit Factor", "Avg Days Held", "TP Exits", "Hold Exits"]
    print(summary_df[display_cols].to_string(index=False))
    
    # Find optimal
    print(f"\n{'─'*70}")
    best_by_return = max(summary, key=lambda x: x["_avg_ret_raw"])
    best_by_pnl = max(summary, key=lambda x: x["_total_pnl_raw"])
    best_by_pf = max(summary, key=lambda x: x["_pf_raw"] if x["_pf_raw"] < 100 else 0)
    best_by_wr = max(summary, key=lambda x: x["_win_rate_raw"])
    
    print(f"\n🏆 OPTIMAL BY METRIC:")
    print(f"  Best Avg Return:    {best_by_return['TP Level']} → {best_by_return['Avg Return']}")
    print(f"  Best Total P&L:     {best_by_pnl['TP Level']} → {best_by_pnl['Total P&L']}")
    print(f"  Best Profit Factor: {best_by_pf['TP Level']} → {best_by_pf['Profit Factor']}")
    print(f"  Best Win Rate:      {best_by_wr['TP Level']} → {best_by_wr['Win Rate']}")
    
    return summary_df


if __name__ == "__main__":
    df = optimize_take_profit()

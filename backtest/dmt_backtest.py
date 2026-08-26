"""
DMT Backtester — simulates LSTM-based directional options trading on historical data.

Simulates buying a call option (for UP predictions) or a put option (for DOWN predictions)
using the actual walk-forward predictions generated during training.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DMT_BUDGET, DMT_TP_PCT, DMT_SL_PCT, MIN_EXPIRY_TRADING_DAYS,
    COMMISSION_PER_CONTRACT, REJECT_DAMPENED
)
from models.train_lstm import prepare_ticker_data, train_walk_forward
from models.garch_model import GARCHVolatilityModel


def run_backtest(ticker: str, horizon: int = 5, budget: float = DMT_BUDGET):
    """
    Run backtest using walk-forward predictions.
    """
    print(f"\n{'='*60}")
    print(f"🎬 Running DMT Backtest for {ticker} | Horizon: {horizon}d | Budget: ${budget}")
    print(f"{'='*60}")

    # Generate walk-forward predictions
    wf_results = train_walk_forward(ticker, horizon=horizon, verbose=False)
    if not wf_results:
        print("  ❌ Failed to generate walk-forward predictions.")
        return None

    # Load price data
    data_prep = prepare_ticker_data(ticker, horizon=horizon)
    if not data_prep:
        return None
    _, _, prices = data_prep

    predictions = wf_results['predictions']
    dates = pd.to_datetime(predictions['dates'])
    pred_classes = predictions['predicted']
    actual_classes = predictions['actual']
    confs = predictions['confidence']

    pred_df = pd.DataFrame({
        'predicted': pred_classes,
        'actual': actual_classes,
        'confidence': confs
    }, index=dates)

    # Reindex prices to match predictions
    prices = prices.loc[pred_df.index[0]:pred_df.index[-1]]

    # Simulation state
    cash = budget
    position = None  # None or dict: {'type': 'CALL'|'PUT', 'entry_price': float, 'entry_date': datetime, 'contracts': int, 'premium': float}
    trades = []
    equity_curve = []

    for i in range(len(pred_df)):
        current_date = pred_df.index[i]
        current_row = pred_df.iloc[i]
        spot = prices.loc[current_date, 'Close']

        # Check existing position for exit (TP/SL or expiration)
        if position is not None:
            days_held = (current_date - position['entry_date']).days * 5 / 7  # estimate trading days
            # Fetch options performance using spot changes as proxy
            # We estimate the option price using intrinsic value changes
            pct_change = (spot - position['entry_spot']) / position['entry_spot']
            
            # Simple delta/leverage model: single-leg option has leverage of ~10x
            leverage = 8.0
            if position['type'] == 'CALL':
                opt_return = pct_change * leverage
            else:
                opt_return = -pct_change * leverage

            # Adjust for daily theta decay (~2% per trading day)
            opt_return -= 0.02 * days_held

            # TP/SL check
            exit_reason = None
            if opt_return * 100 >= DMT_TP_PCT:
                exit_reason = 'TP'
                opt_return = DMT_TP_PCT / 100.0
            elif opt_return * 100 <= DMT_SL_PCT:
                exit_reason = 'SL'
                opt_return = DMT_SL_PCT / 100.0
            elif days_held >= horizon:
                exit_reason = 'EXPIRY'

            if exit_reason:
                exit_value = position['cost'] * (1 + opt_return)
                cash += exit_value
                trades.append({
                    'ticker': ticker,
                    'type': position['type'],
                    'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'entry_spot': position['entry_spot'],
                    'exit_spot': spot,
                    'return_pct': opt_return * 100,
                    'pnl': exit_value - position['cost'],
                    'exit_reason': exit_reason
                })
                position = None

        # Check for new entry if no open position
        if position is None:
            pred = current_row['predicted']
            conf = current_row['confidence']

            # Only trade on high confidence directions (0=DOWN -> PUT, 2=UP -> CALL)
            if conf >= 0.65 and pred in [0, 2]:
                # Apply Dampening Check
                is_dampened = False
                if REJECT_DAMPENED:
                    # Fit GARCH once per month to speed up backtest
                    # (GARCH parameters change slowly day-to-day)
                    cache_key = f"{ticker}_{current_date.strftime('%Y-%m')}"
                    if not hasattr(run_backtest, '_garch_cache'):
                        run_backtest._garch_cache = {}
                    
                    if cache_key not in run_backtest._garch_cache:
                        garch = GARCHVolatilityModel()
                        try:
                            garch.fit(prices.loc[:current_date], verbose=False)
                            run_backtest._garch_cache[cache_key] = garch.dampened
                        except Exception:
                            run_backtest._garch_cache[cache_key] = False
                    
                    is_dampened = run_backtest._garch_cache[cache_key]
                
                if not is_dampened:
                    opt_type = 'CALL' if pred == 2 else 'PUT'
                    
                    # Assume options premium is roughly 4% of stock price
                    est_premium = spot * 0.04
                    contract_cost = est_premium * 100 + COMMISSION_PER_CONTRACT
                    
                    # Use full budget or reasonable portion
                    max_contracts = int((cash - COMMISSION_PER_CONTRACT) / contract_cost)
                    
                    if max_contracts > 0:
                        position_cost = (est_premium * 100 * max_contracts) + COMMISSION_PER_CONTRACT
                        cash -= position_cost
                        position = {
                            'type': opt_type,
                            'entry_date': current_date,
                            'entry_spot': spot,
                            'cost': position_cost,
                            'contracts': max_contracts
                        }

        # Track total portfolio equity
        current_val = cash
        if position is not None:
            # Estimate current value of option
            pct_change = (spot - position['entry_spot']) / position['entry_spot']
            leverage = 8.0
            days_held = (current_date - position['entry_date']).days * 5 / 7
            opt_return = (pct_change * leverage if position['type'] == 'CALL' else -pct_change * leverage) - (0.02 * days_held)
            current_val += position['cost'] * (1 + max(opt_return, -1.0))
        
        equity_curve.append(current_val)

    # ─── Metrics Calculation ───────────────────────────────
    trades_df = pd.DataFrame(trades)
    
    if trades_df.empty:
        print("  ⚠️  No trades executed during the backtest period.")
        return None

    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0.0
    total_pnl = trades_df['pnl'].sum()
    pct_return = (total_pnl / budget) * 100

    # Calculate Sharpe Ratio
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    avg_ret = returns.mean() * 252
    std_ret = returns.std() * np.sqrt(252)
    sharpe = avg_ret / std_ret if std_ret > 0 else 0.0

    print(f"\n📊 BACKTEST RESULTS SUMMARY ({ticker})")
    print(f"   Total trades:      {total_trades}")
    print(f"   Win rate:          {win_rate:.1%}")
    print(f"   Total P&L:         ${total_pnl:+.2f} ({pct_return:+.1f}%)")
    print(f"   Sharpe Ratio:      {sharpe:.2f}")
    print(f"   Final Equity:      ${equity_curve[-1]:.2f}")
    
    # Exit reasons breakdown
    reasons = trades_df['exit_reason'].value_counts()
    print("   Exit reasons:")
    for r_name, r_count in reasons.items():
        print(f"     - {r_name}: {r_count}")
        
    return {
        'ticker': ticker,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'pct_return': pct_return,
        'sharpe': sharpe,
        'final_equity': equity_curve[-1]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMT Backtest")
    parser.add_argument("--ticker", type=str, default="PYPL", help="Ticker symbol to backtest")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days")
    args = parser.parse_args()

    run_backtest(args.ticker, horizon=args.horizon)

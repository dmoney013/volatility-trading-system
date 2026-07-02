"""
Day-of-Week Straddle Backtest — tests long straddle entry on specific
weekdays with a +12% take-profit exit (no fixed holding period).

Strategy:
  - Entry: Buy ATM straddle on every Thursday (or Friday) where GARCH
    signals the vol spread exceeds the entry threshold
  - Exit: Close as soon as the straddle appreciates by +12% or more,
    checking daily. No fixed holding period.
  - Max hold: 15 trading days (safety valve — closes at market if
    +12% was never reached)
  - Budget: Constrained to $150 initial capital

Usage:
    python backtest/day_of_week_tp.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, ENTRY_VOL_THRESHOLD,
    MAX_POSITION_PCT, COMMISSION_PER_CONTRACT, TRADING_DAYS,
)
from signals.scanner import SCAN_UNIVERSE


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes Option Pricing
# ═══════════════════════════════════════════════════════════════════

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def straddle_price(S, K, T, r, sigma):
    return black_scholes_call(S, K, T, r, sigma) + black_scholes_put(S, K, T, r, sigma)


# ═══════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    spot_entry: float
    spot_exit: float
    strike: float
    entry_price: float
    exit_price: float
    contracts: int
    entry_vol: float
    exit_vol: float
    pnl_per_share: float
    pnl_total: float
    commission: float
    net_pnl: float
    return_pct: float
    garch_rv_forecast: float
    signal_strength: float
    days_held: int
    exit_reason: str  # "take_profit" or "max_hold"


# ═══════════════════════════════════════════════════════════════════
# Backtester with Take-Profit Exit
# ═══════════════════════════════════════════════════════════════════

class DayOfWeekTPBacktester:
    """
    Backtests a GARCH-informed long straddle strategy with:
      - Entry restricted to a specific weekday (0=Mon ... 4=Fri)
      - Exit when straddle value reaches +12% profit (take-profit)
      - No fixed holding period — checks daily for the TP threshold
      - Safety max hold of 15 trading days
    """

    def __init__(
        self,
        entry_weekday: int = 3,         # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
        take_profit_pct: float = 12.0,   # Close at +12%
        max_hold_days: int = 15,         # Safety valve
        initial_capital: float = INITIAL_CAPITAL,
        entry_threshold: float = ENTRY_VOL_THRESHOLD,
        max_position_pct: float = MAX_POSITION_PCT,
        commission: float = COMMISSION_PER_CONTRACT,
    ):
        self.entry_weekday = entry_weekday
        self.take_profit_pct = take_profit_pct
        self.max_hold_days = max_hold_days
        self.initial_capital = initial_capital
        self.entry_threshold = entry_threshold
        self.max_position_pct = max_position_pct
        self.commission = commission
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []

    def run(
        self,
        prices: pd.DataFrame,
        conditional_vol: pd.Series,
        treasury: pd.DataFrame,
        ticker: str = "SIM",
        verbose: bool = False,
    ) -> dict:
        """Run the backtest for a single ticker."""
        close = prices["Close"]
        common_idx = conditional_vol.index.intersection(close.index)
        if not treasury.empty:
            common_idx = common_idx.intersection(treasury.index)
        common_idx = common_idx.sort_values()

        capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        in_position = False
        entry_trade_data = None
        position_entry_idx = 0

        # Risk-free rate
        if not treasury.empty and "Close" in treasury.columns:
            rf_rate = treasury["Close"].reindex(common_idx, method="ffill") / 100.0
        else:
            rf_rate = pd.Series(0.04, index=common_idx)

        # Market IV proxy: 21-day rolling realized vol
        log_ret = np.log(close / close.shift(1))
        market_iv_proxy = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
        market_iv_proxy = market_iv_proxy.reindex(common_idx, method="ffill")

        weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][self.entry_weekday]

        for i in range(len(common_idx)):
            date = common_idx[i]
            spot = close.loc[date]
            garch_vol = conditional_vol.loc[date] if date in conditional_vol.index else None

            if garch_vol is None:
                self.equity_curve.append({"date": date, "equity": capital, "in_trade": False})
                continue

            mkt_iv = market_iv_proxy.loc[date] if date in market_iv_proxy.index and not np.isnan(market_iv_proxy.loc[date]) else garch_vol
            r = rf_rate.loc[date] if date in rf_rate.index else 0.04

            self.equity_curve.append({"date": date, "equity": capital, "in_trade": in_position})

            if in_position:
                days_held = i - position_entry_idx

                # Compute current straddle value using Black-Scholes
                T_remaining = max((self.max_hold_days - days_held) / TRADING_DAYS, 0.001)
                strike = entry_trade_data["strike"]
                current_straddle = straddle_price(spot, strike, T_remaining, r, garch_vol)

                # Also compute intrinsic value as a floor
                intrinsic = max(spot - strike, 0) + max(strike - spot, 0)
                exit_price = max(current_straddle, intrinsic)

                entry_price = entry_trade_data["entry_price"]
                current_return_pct = (exit_price - entry_price) / entry_price * 100

                # Check take-profit threshold
                exit_reason = None
                if current_return_pct >= self.take_profit_pct:
                    exit_reason = "take_profit"
                elif days_held >= self.max_hold_days:
                    exit_reason = "max_hold"

                if exit_reason:
                    trade = self._close_position(
                        entry_trade_data, date, spot, garch_vol, exit_price,
                        days_held, exit_reason, capital
                    )
                    capital += trade.net_pnl
                    self.trades.append(trade)
                    in_position = False

                    if verbose:
                        print(f"  {trade.entry_date.date()} → {trade.exit_date.date()} "
                              f"({days_held}d) | {exit_reason:12s} | "
                              f"P&L: ${trade.net_pnl:+.2f} ({trade.return_pct:+.1f}%) | "
                              f"Capital: ${capital:.2f}")
            else:
                # Only enter on the target weekday
                if date.weekday() != self.entry_weekday:
                    continue

                # Check GARCH signal
                vol_spread = garch_vol - mkt_iv
                signal_strength = vol_spread / max(mkt_iv, 0.01)

                if vol_spread > self.entry_threshold:
                    T = self.max_hold_days / TRADING_DAYS
                    strike = round(spot, 0)

                    entry_straddle_price = straddle_price(spot, strike, T, r, mkt_iv)
                    if entry_straddle_price <= 0.01:
                        continue

                    cost_per_contract = entry_straddle_price * 100
                    total_commission = self.commission * 2
                    max_spend = capital * self.max_position_pct
                    contracts = int((max_spend - total_commission) / cost_per_contract)

                    if contracts < 1:
                        continue

                    entry_trade_data = {
                        "entry_date": date,
                        "ticker": ticker,
                        "spot_entry": spot,
                        "strike": strike,
                        "entry_price": entry_straddle_price,
                        "contracts": contracts,
                        "entry_vol": mkt_iv,
                        "garch_rv_forecast": garch_vol,
                        "signal_strength": signal_strength,
                    }
                    in_position = True
                    position_entry_idx = i

        results = self._compute_stats(capital)
        return results

    def _close_position(self, entry_data, exit_date, spot_exit, exit_vol,
                        exit_price, days_held, exit_reason, capital):
        strike = entry_data["strike"]
        contracts = entry_data["contracts"]
        entry_price = entry_data["entry_price"]

        pnl_per_share = exit_price - entry_price
        pnl_total = pnl_per_share * contracts * 100
        commission = self.commission * 2 * 2
        net_pnl = pnl_total - commission
        cost_basis = entry_price * contracts * 100 + commission / 2
        return_pct = net_pnl / max(cost_basis, 0.01) * 100

        return Trade(
            entry_date=entry_data["entry_date"],
            exit_date=exit_date,
            ticker=entry_data["ticker"],
            spot_entry=entry_data["spot_entry"],
            spot_exit=spot_exit,
            strike=strike,
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=contracts,
            entry_vol=entry_data["entry_vol"],
            exit_vol=exit_vol,
            pnl_per_share=pnl_per_share,
            pnl_total=pnl_total,
            commission=commission,
            net_pnl=net_pnl,
            return_pct=return_pct,
            garch_rv_forecast=entry_data["garch_rv_forecast"],
            signal_strength=entry_data["signal_strength"],
            days_held=days_held,
            exit_reason=exit_reason,
        )

    def _compute_stats(self, final_capital) -> dict:
        if not self.trades:
            return {
                "total_trades": 0,
                "final_capital": final_capital,
                "total_return_pct": (final_capital - self.initial_capital) / self.initial_capital * 100,
            }

        trades_df = pd.DataFrame([{
            "entry_date": t.entry_date, "exit_date": t.exit_date,
            "ticker": t.ticker,
            "spot_entry": t.spot_entry, "spot_exit": t.spot_exit,
            "strike": t.strike, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "contracts": t.contracts,
            "pnl_total": t.pnl_total, "net_pnl": t.net_pnl,
            "return_pct": t.return_pct, "days_held": t.days_held,
            "exit_reason": t.exit_reason,
            "entry_vol": t.entry_vol, "garch_rv_forecast": t.garch_rv_forecast,
        } for t in self.trades])

        winners = trades_df[trades_df["net_pnl"] > 0]
        losers = trades_df[trades_df["net_pnl"] <= 0]
        tp_exits = trades_df[trades_df["exit_reason"] == "take_profit"]

        total_pnl = trades_df["net_pnl"].sum()
        win_rate = len(winners) / len(trades_df) * 100
        profit_factor = (abs(winners["net_pnl"].sum() / losers["net_pnl"].sum())
                         if len(losers) > 0 and losers["net_pnl"].sum() != 0
                         else float("inf"))

        equity_df = pd.DataFrame(self.equity_curve)
        max_dd = 0
        if not equity_df.empty:
            equity_df["drawdown"] = equity_df["equity"] / equity_df["equity"].cummax() - 1
            max_dd = equity_df["drawdown"].min() * 100

        return {
            "total_trades": len(trades_df),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "tp_exits": len(tp_exits),
            "tp_rate": len(tp_exits) / len(trades_df) * 100,
            "max_hold_exits": len(trades_df) - len(tp_exits),
            "total_pnl": total_pnl,
            "avg_pnl": trades_df["net_pnl"].mean(),
            "avg_win": winners["net_pnl"].mean() if len(winners) > 0 else 0,
            "avg_loss": losers["net_pnl"].mean() if len(losers) > 0 else 0,
            "best_trade": trades_df["net_pnl"].max(),
            "worst_trade": trades_df["net_pnl"].min(),
            "profit_factor": profit_factor,
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return_pct": (final_capital - self.initial_capital) / self.initial_capital * 100,
            "max_drawdown_pct": max_dd,
            "avg_days_held": trades_df["days_held"].mean(),
            "avg_days_held_tp": tp_exits["days_held"].mean() if len(tp_exits) > 0 else 0,
            "trades_df": trades_df,
            "equity_df": equity_df,
        }


# ═══════════════════════════════════════════════════════════════════
# Multi-Ticker Runner
# ═══════════════════════════════════════════════════════════════════

def run_day_of_week_backtest(entry_weekday: int, verbose: bool = True) -> dict:
    """
    Run the take-profit backtest across all SCAN_UNIVERSE tickers
    for a specific entry weekday.

    Args:
        entry_weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    """
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][entry_weekday]

    print(f"\n{'═'*70}")
    print(f"  LONG STRADDLE BACKTEST — {weekday_name.upper()} ENTRY + 12% TAKE-PROFIT EXIT")
    print(f"  Universe: {len(SCAN_UNIVERSE)} tickers | Budget: ${INITIAL_CAPITAL}")
    print(f"  Exit: Close immediately when P&L ≥ +12% (max hold: 15 days)")
    print(f"{'═'*70}\n")

    treasury = fetch_treasury_yield()

    all_trades = []
    ticker_results = []
    failed = []

    for sym in SCAN_UNIVERSE:
        try:
            prices = fetch_price_data(sym)
            garch = GARCHVolatilityModel()
            garch.fit(prices, verbose=False)
            cond_vol = garch.get_conditional_volatility()

            bt = DayOfWeekTPBacktester(
                entry_weekday=entry_weekday,
                take_profit_pct=12.0,
                max_hold_days=15,
            )
            results = bt.run(prices, cond_vol, treasury, ticker=sym, verbose=False)

            if results["total_trades"] > 0:
                ticker_results.append({
                    "ticker": sym,
                    "trades": results["total_trades"],
                    "win_rate": results["win_rate"],
                    "tp_rate": results["tp_rate"],
                    "total_pnl": results["total_pnl"],
                    "return_pct": results["total_return_pct"],
                    "avg_days": results["avg_days_held"],
                    "final_capital": results["final_capital"],
                })
                # Collect individual trades
                for t in bt.trades:
                    all_trades.append(t)

                if verbose:
                    print(f"  {sym:6s} | {results['total_trades']:3d} trades | "
                          f"Win: {results['win_rate']:5.1f}% | "
                          f"TP hits: {results['tp_exits']:3d}/{results['total_trades']:3d} ({results['tp_rate']:.0f}%) | "
                          f"P&L: ${results['total_pnl']:+8.2f} | "
                          f"Return: {results['total_return_pct']:+7.1f}% | "
                          f"Avg hold: {results['avg_days_held']:.1f}d")
        except Exception as e:
            failed.append((sym, str(e)))

    if not ticker_results:
        print("No trades generated across any ticker.")
        return {}

    # ─── Aggregate Stats ────────────────────────────────────────
    df = pd.DataFrame(ticker_results)
    total_trades = df["trades"].sum()
    total_pnl = df["total_pnl"].sum()
    avg_return = df["return_pct"].mean()
    avg_win_rate = df["win_rate"].mean()
    avg_tp_rate = df["tp_rate"].mean()
    avg_days = df["avg_days"].mean()
    profitable_tickers = len(df[df["total_pnl"] > 0])

    # Trade-level stats
    all_net_pnl = [t.net_pnl for t in all_trades]
    all_days = [t.days_held for t in all_trades]
    all_exit_reasons = [t.exit_reason for t in all_trades]
    tp_count = sum(1 for r in all_exit_reasons if r == "take_profit")

    print(f"\n{'─'*70}")
    print(f"  AGGREGATE RESULTS — {weekday_name.upper()} ENTRY")
    print(f"{'─'*70}")
    print(f"  Tickers tested:      {len(ticker_results)} / {len(SCAN_UNIVERSE)}")
    print(f"  Profitable tickers:  {profitable_tickers} / {len(ticker_results)}")
    print(f"  Total trades:        {total_trades}")
    print(f"  Overall win rate:    {avg_win_rate:.1f}%")
    print(f"  Take-profit hit:     {tp_count}/{len(all_trades)} ({tp_count/len(all_trades)*100:.1f}%)")
    print(f"  Avg holding period:  {avg_days:.1f} days")
    print(f"  Total P&L:           ${total_pnl:+.2f}")
    print(f"  Avg return/ticker:   {avg_return:+.1f}%")
    print(f"  Best ticker P&L:     {df.loc[df['total_pnl'].idxmax(), 'ticker']} "
          f"(${df['total_pnl'].max():+.2f})")
    print(f"  Worst ticker P&L:    {df.loc[df['total_pnl'].idxmin(), 'ticker']} "
          f"(${df['total_pnl'].min():+.2f})")
    if failed:
        print(f"  Failed tickers:      {len(failed)} ({', '.join(f[0] for f in failed[:5])})")
    print(f"{'─'*70}\n")

    return {
        "weekday": weekday_name,
        "entry_weekday": entry_weekday,
        "ticker_results": df,
        "all_trades": all_trades,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "avg_return_pct": avg_return,
        "avg_win_rate": avg_win_rate,
        "avg_tp_rate": avg_tp_rate,
        "avg_days_held": avg_days,
        "tp_count": tp_count,
        "profitable_tickers": profitable_tickers,
    }


# ═══════════════════════════════════════════════════════════════════
# Main — Run Thursday and Friday
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  DAY-OF-WEEK STRADDLE BACKTEST: THURSDAY vs FRIDAY ENTRY")
    print("  Exit: +12% take-profit (no fixed holding period)")
    print("=" * 70)

    # Run Thursday (weekday=3)
    thu_results = run_day_of_week_backtest(entry_weekday=3, verbose=True)

    # Run Friday (weekday=4)
    fri_results = run_day_of_week_backtest(entry_weekday=4, verbose=True)

    # ─── Comparison ─────────────────────────────────────────────
    if thu_results and fri_results:
        print("\n" + "═" * 70)
        print("  HEAD-TO-HEAD COMPARISON: THURSDAY vs FRIDAY")
        print("═" * 70)
        print(f"{'Metric':<30s} {'Thursday':>15s} {'Friday':>15s} {'Winner':>10s}")
        print("─" * 70)

        comparisons = [
            ("Total trades", thu_results["total_trades"], fri_results["total_trades"], None),
            ("Total P&L", f"${thu_results['total_pnl']:+.2f}", f"${fri_results['total_pnl']:+.2f}",
             "Thu" if thu_results["total_pnl"] > fri_results["total_pnl"] else "Fri"),
            ("Avg return/ticker", f"{thu_results['avg_return_pct']:+.1f}%", f"{fri_results['avg_return_pct']:+.1f}%",
             "Thu" if thu_results["avg_return_pct"] > fri_results["avg_return_pct"] else "Fri"),
            ("Win rate", f"{thu_results['avg_win_rate']:.1f}%", f"{fri_results['avg_win_rate']:.1f}%",
             "Thu" if thu_results["avg_win_rate"] > fri_results["avg_win_rate"] else "Fri"),
            ("TP hit rate", f"{thu_results['avg_tp_rate']:.1f}%", f"{fri_results['avg_tp_rate']:.1f}%",
             "Thu" if thu_results["avg_tp_rate"] > fri_results["avg_tp_rate"] else "Fri"),
            ("Avg days held", f"{thu_results['avg_days_held']:.1f}", f"{fri_results['avg_days_held']:.1f}",
             "Thu" if thu_results["avg_days_held"] < fri_results["avg_days_held"] else "Fri"),
            ("Profitable tickers", str(thu_results["profitable_tickers"]), str(fri_results["profitable_tickers"]),
             "Thu" if thu_results["profitable_tickers"] > fri_results["profitable_tickers"] else "Fri"),
        ]

        for label, thu_val, fri_val, winner in comparisons:
            w = f"  ← {winner}" if winner else ""
            print(f"  {label:<28s} {str(thu_val):>15s} {str(fri_val):>15s} {w}")

        print("─" * 70)

        # Save comparison
        comparison_data = {
            "thursday": {
                "total_trades": int(thu_results["total_trades"]),
                "total_pnl": round(float(thu_results["total_pnl"]), 2),
                "avg_return_pct": round(float(thu_results["avg_return_pct"]), 2),
                "avg_win_rate": round(float(thu_results["avg_win_rate"]), 2),
                "avg_tp_rate": round(float(thu_results["avg_tp_rate"]), 2),
                "avg_days_held": round(float(thu_results["avg_days_held"]), 2),
                "profitable_tickers": int(thu_results["profitable_tickers"]),
                "tp_count": int(thu_results["tp_count"]),
            },
            "friday": {
                "total_trades": int(fri_results["total_trades"]),
                "total_pnl": round(float(fri_results["total_pnl"]), 2),
                "avg_return_pct": round(float(fri_results["avg_return_pct"]), 2),
                "avg_win_rate": round(float(fri_results["avg_win_rate"]), 2),
                "avg_tp_rate": round(float(fri_results["avg_tp_rate"]), 2),
                "avg_days_held": round(float(fri_results["avg_days_held"]), 2),
                "profitable_tickers": int(fri_results["profitable_tickers"]),
                "tp_count": int(fri_results["tp_count"]),
            },
        }
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "results", "thu_vs_fri_tp12.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\n✓ Results saved to {out_path}")

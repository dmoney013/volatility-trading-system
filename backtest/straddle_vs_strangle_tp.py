"""
Straddle vs Strangle Backtest — +12% Take-Profit Exit (any day entry).

Compares:
  - Long Straddle: ATM call + ATM put
  - Long Strangle: OTM call (5% above) + OTM put (5% below)

Both use:
  - GARCH signal for entry (vol spread > threshold)
  - +12% take-profit exit — close the day the position appreciates ≥ +12%
  - 15 trading-day max hold (safety valve)
  - Entry on any weekday (no day-of-week filter)
  - $150 initial capital, compounding

Usage:
    python backtest/straddle_vs_strangle_tp.py
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
    STRANGLE_OTM_WIDTH, STRANGLE_ENTRY_VOL_THRESHOLD,
)
from signals.scanner import SCAN_UNIVERSE
from backtest.engine import black_scholes_call, black_scholes_put, straddle_price


# ═══════════════════════════════════════════════════════════════════
# Strangle Pricing
# ═══════════════════════════════════════════════════════════════════

def strangle_price_fn(S, K_call, K_put, T, r, sigma):
    """OTM call at K_call + OTM put at K_put."""
    return black_scholes_call(S, K_call, T, r, sigma) + black_scholes_put(S, K_put, T, r, sigma)


def select_strangle_strikes(spot, width=STRANGLE_OTM_WIDTH):
    call_strike = round(spot * (1 + width))
    put_strike = round(spot * (1 - width))
    if call_strike <= round(spot):
        call_strike = round(spot) + 1
    if put_strike >= round(spot):
        put_strike = round(spot) - 1
    return call_strike, put_strike


# ═══════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    strategy: str  # "straddle" or "strangle"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    spot_entry: float
    spot_exit: float
    strike: float           # ATM strike (straddle) or 0 (strangle)
    call_strike: float      # Same as strike for straddle
    put_strike: float       # Same as strike for straddle
    entry_price: float
    exit_price: float
    contracts: int
    entry_vol: float
    exit_vol: float
    net_pnl: float
    return_pct: float
    days_held: int
    exit_reason: str


# ═══════════════════════════════════════════════════════════════════
# Unified Backtester
# ═══════════════════════════════════════════════════════════════════

class TPBacktester:
    """
    Backtests either a long straddle or long strangle with +12% TP exit.
    Entry on any weekday where GARCH signal fires.
    """

    def __init__(
        self,
        strategy: str = "straddle",       # "straddle" or "strangle"
        take_profit_pct: float = 12.0,
        max_hold_days: int = 15,
        initial_capital: float = INITIAL_CAPITAL,
        entry_threshold: float = ENTRY_VOL_THRESHOLD,
        max_position_pct: float = MAX_POSITION_PCT,
        commission: float = COMMISSION_PER_CONTRACT,
        otm_width: float = STRANGLE_OTM_WIDTH,
        entry_weekday: int = None,        # None=any day, 0=Mon, 3=Thu, 4=Fri
    ):
        self.strategy = strategy
        self.take_profit_pct = take_profit_pct
        self.max_hold_days = max_hold_days
        self.initial_capital = initial_capital
        self.entry_threshold = entry_threshold
        self.max_position_pct = max_position_pct
        self.commission = commission
        self.otm_width = otm_width
        self.entry_weekday = entry_weekday
        self.trades: List[Trade] = []

    def run(
        self,
        prices: pd.DataFrame,
        conditional_vol: pd.Series,
        treasury: pd.DataFrame,
        ticker: str = "SIM",
    ) -> dict:
        close = prices["Close"]
        common_idx = conditional_vol.index.intersection(close.index)
        if not treasury.empty:
            common_idx = common_idx.intersection(treasury.index)
        common_idx = common_idx.sort_values()

        capital = self.initial_capital
        self.trades = []
        in_position = False
        entry_data = None
        entry_idx = 0

        # Risk-free rate
        if not treasury.empty and "Close" in treasury.columns:
            rf_rate = treasury["Close"].reindex(common_idx, method="ffill") / 100.0
        else:
            rf_rate = pd.Series(0.04, index=common_idx)

        # Market IV proxy
        log_ret = np.log(close / close.shift(1))
        market_iv_proxy = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
        market_iv_proxy = market_iv_proxy.reindex(common_idx, method="ffill")

        for i in range(len(common_idx)):
            date = common_idx[i]
            spot = close.loc[date]
            garch_vol = conditional_vol.loc[date] if date in conditional_vol.index else None
            if garch_vol is None:
                continue

            mkt_iv = market_iv_proxy.loc[date] if date in market_iv_proxy.index and not np.isnan(market_iv_proxy.loc[date]) else garch_vol
            r = rf_rate.loc[date] if date in rf_rate.index else 0.04

            if in_position:
                days_held = i - entry_idx

                # Price the position at current date
                T_remaining = max((self.max_hold_days - days_held) / TRADING_DAYS, 0.001)
                if self.strategy == "straddle":
                    strike = entry_data["strike"]
                    current_val = straddle_price(spot, strike, T_remaining, r, garch_vol)
                    intrinsic = max(spot - strike, 0) + max(strike - spot, 0)
                else:
                    ck = entry_data["call_strike"]
                    pk = entry_data["put_strike"]
                    current_val = strangle_price_fn(spot, ck, pk, T_remaining, r, garch_vol)
                    intrinsic = max(spot - ck, 0) + max(pk - spot, 0)

                exit_price = max(current_val, intrinsic)
                ep = entry_data["entry_price"]
                ret_pct = (exit_price - ep) / ep * 100

                exit_reason = None
                if ret_pct >= self.take_profit_pct:
                    exit_reason = "take_profit"
                elif days_held >= self.max_hold_days:
                    exit_reason = "max_hold"

                if exit_reason:
                    contracts = entry_data["contracts"]
                    pnl_total = (exit_price - ep) * contracts * 100
                    commission = self.commission * 2 * 2
                    net_pnl = pnl_total - commission
                    cost_basis = ep * contracts * 100 + commission / 2
                    final_ret = net_pnl / max(cost_basis, 0.01) * 100

                    trade = Trade(
                        strategy=self.strategy,
                        entry_date=entry_data["entry_date"],
                        exit_date=date,
                        ticker=ticker,
                        spot_entry=entry_data["spot_entry"],
                        spot_exit=spot,
                        strike=entry_data.get("strike", 0),
                        call_strike=entry_data["call_strike"],
                        put_strike=entry_data["put_strike"],
                        entry_price=ep,
                        exit_price=exit_price,
                        contracts=contracts,
                        entry_vol=entry_data["entry_vol"],
                        exit_vol=garch_vol,
                        net_pnl=net_pnl,
                        return_pct=final_ret,
                        days_held=days_held,
                        exit_reason=exit_reason,
                    )
                    capital += net_pnl
                    self.trades.append(trade)
                    in_position = False

            else:
                # Weekday filter (if set)
                if self.entry_weekday is not None and date.weekday() != self.entry_weekday:
                    continue

                # Check entry signal
                vol_spread = garch_vol - mkt_iv
                if vol_spread <= self.entry_threshold:
                    continue

                signal_strength = vol_spread / max(mkt_iv, 0.01)
                T = self.max_hold_days / TRADING_DAYS

                if self.strategy == "straddle":
                    strike = round(spot, 0)
                    entry_price = straddle_price(spot, strike, T, r, mkt_iv)
                    call_strike = strike
                    put_strike = strike
                else:
                    call_strike, put_strike = select_strangle_strikes(spot, self.otm_width)
                    entry_price = strangle_price_fn(spot, call_strike, put_strike, T, r, mkt_iv)
                    strike = 0

                if entry_price <= 0.01:
                    continue

                cost_per = entry_price * 100
                max_spend = capital * self.max_position_pct
                contracts = int((max_spend - self.commission * 2) / cost_per)
                if contracts < 1:
                    continue

                entry_data = {
                    "entry_date": date,
                    "ticker": ticker,
                    "spot_entry": spot,
                    "strike": strike if self.strategy == "straddle" else 0,
                    "call_strike": call_strike,
                    "put_strike": put_strike,
                    "entry_price": entry_price,
                    "contracts": contracts,
                    "entry_vol": mkt_iv,
                    "signal_strength": signal_strength,
                }
                in_position = True
                entry_idx = i

        return self._compute_stats(capital)

    def _compute_stats(self, final_capital) -> dict:
        if not self.trades:
            return {
                "total_trades": 0,
                "winners": 0, "losers": 0,
                "win_rate": 0, "tp_rate": 0,
                "total_pnl": 0, "avg_days_held": 0,
                "final_capital": final_capital,
                "total_return_pct": (final_capital - self.initial_capital) / self.initial_capital * 100,
            }

        net_pnls = [t.net_pnl for t in self.trades]
        days = [t.days_held for t in self.trades]
        tp_count = sum(1 for t in self.trades if t.exit_reason == "take_profit")
        winners = sum(1 for p in net_pnls if p > 0)
        losers = len(net_pnls) - winners
        total_pnl = sum(net_pnls)

        win_sum = sum(p for p in net_pnls if p > 0)
        loss_sum = sum(p for p in net_pnls if p <= 0)
        profit_factor = abs(win_sum / loss_sum) if loss_sum != 0 else float("inf")

        return {
            "total_trades": len(self.trades),
            "winners": winners,
            "losers": losers,
            "win_rate": winners / len(self.trades) * 100,
            "tp_exits": tp_count,
            "tp_rate": tp_count / len(self.trades) * 100,
            "max_hold_exits": len(self.trades) - tp_count,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(self.trades),
            "avg_days_held": sum(days) / len(days),
            "avg_days_tp": (sum(t.days_held for t in self.trades if t.exit_reason == "take_profit") /
                           max(tp_count, 1)),
            "profit_factor": profit_factor,
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return_pct": (final_capital - self.initial_capital) / self.initial_capital * 100,
        }


# ═══════════════════════════════════════════════════════════════════
# Multi-Ticker Runner
# ═══════════════════════════════════════════════════════════════════

def run_strategy_backtest(strategy: str, verbose: bool = True) -> dict:
    """Run backtest across all SCAN_UNIVERSE tickers for one strategy."""
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    label = strategy.upper()
    print(f"\n{'═'*70}")
    print(f"  LONG {label} — +12% TAKE-PROFIT EXIT (ANY DAY ENTRY)")
    print(f"  Universe: {len(SCAN_UNIVERSE)} tickers | Budget: ${INITIAL_CAPITAL}")
    print(f"{'═'*70}\n")

    treasury = fetch_treasury_yield()
    ticker_results = []
    all_trades = []
    failed = []

    for sym in SCAN_UNIVERSE:
        try:
            prices = fetch_price_data(sym)
            garch = GARCHVolatilityModel()
            garch.fit(prices, verbose=False)
            cond_vol = garch.get_conditional_volatility()

            bt = TPBacktester(
                strategy=strategy,
                take_profit_pct=12.0,
                max_hold_days=15,
            )
            results = bt.run(prices, cond_vol, treasury, ticker=sym)

            if results["total_trades"] > 0:
                ticker_results.append({
                    "ticker": sym,
                    "trades": results["total_trades"],
                    "win_rate": results["win_rate"],
                    "tp_rate": results["tp_rate"],
                    "total_pnl": results["total_pnl"],
                    "return_pct": results["total_return_pct"],
                    "avg_days": results["avg_days_held"],
                    "profit_factor": results["profit_factor"],
                })
                all_trades.extend(bt.trades)

                if verbose:
                    print(f"  {sym:6s} | {results['total_trades']:3d} trades | "
                          f"Win: {results['win_rate']:5.1f}% | "
                          f"TP: {results['tp_exits']:3d}/{results['total_trades']:3d} "
                          f"({results['tp_rate']:.0f}%) | "
                          f"Avg hold: {results['avg_days_held']:.1f}d")
        except Exception as e:
            failed.append((sym, str(e)))

    if not ticker_results:
        print(f"No trades generated for {strategy}.")
        return {}

    df = pd.DataFrame(ticker_results)
    total_trades = int(df["trades"].sum())
    tp_count = sum(1 for t in all_trades if t.exit_reason == "take_profit")
    winners = sum(1 for t in all_trades if t.net_pnl > 0)

    print(f"\n{'─'*70}")
    print(f"  AGGREGATE: LONG {label}")
    print(f"{'─'*70}")
    print(f"  Tickers w/ trades:   {len(ticker_results)} / {len(SCAN_UNIVERSE)}")
    print(f"  Total trades:        {total_trades}")
    print(f"  Overall win rate:    {winners/len(all_trades)*100:.1f}%")
    print(f"  Take-profit hits:    {tp_count}/{len(all_trades)} ({tp_count/len(all_trades)*100:.1f}%)")
    print(f"  Max-hold exits:      {len(all_trades) - tp_count}")
    print(f"  Avg days held:       {df['avg_days'].mean():.1f}")
    print(f"  Avg win rate/ticker: {df['win_rate'].mean():.1f}%")
    if failed:
        print(f"  Failed tickers:      {len(failed)}")
    print(f"{'─'*70}\n")

    return {
        "strategy": strategy,
        "ticker_results": df,
        "all_trades": all_trades,
        "total_trades": total_trades,
        "winners": winners,
        "overall_win_rate": winners / len(all_trades) * 100,
        "tp_count": tp_count,
        "tp_rate": tp_count / len(all_trades) * 100,
        "max_hold_exits": len(all_trades) - tp_count,
        "avg_win_rate": float(df["win_rate"].mean()),
        "avg_tp_rate": float(df["tp_rate"].mean()),
        "avg_days_held": float(df["avg_days"].mean()),
        "tickers_with_trades": len(ticker_results),
        "failed": len(failed),
    }


# ═══════════════════════════════════════════════════════════════════
# Main — Straddle vs Strangle Comparison
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  STRADDLE vs STRANGLE — +12% TAKE-PROFIT EXIT")
    print("  Entry: Any weekday with GARCH signal | Max hold: 15 days")
    print("=" * 70)

    straddle_results = run_strategy_backtest("straddle", verbose=True)
    strangle_results = run_strategy_backtest("strangle", verbose=True)

    if straddle_results and strangle_results:
        s1 = straddle_results
        s2 = strangle_results

        print("\n" + "═" * 70)
        print("  HEAD-TO-HEAD: LONG STRADDLE vs LONG STRANGLE")
        print("═" * 70)
        print(f"  {'Metric':<30s} {'Straddle':>15s} {'Strangle':>15s} {'Winner':>10s}")
        print("─" * 70)

        rows = [
            ("Total trades",
             str(s1["total_trades"]), str(s2["total_trades"]), None),
            ("Overall win rate",
             f"{s1['overall_win_rate']:.1f}%", f"{s2['overall_win_rate']:.1f}%",
             "Straddle" if s1["overall_win_rate"] > s2["overall_win_rate"] else "Strangle"),
            ("Avg win rate/ticker",
             f"{s1['avg_win_rate']:.1f}%", f"{s2['avg_win_rate']:.1f}%",
             "Straddle" if s1["avg_win_rate"] > s2["avg_win_rate"] else "Strangle"),
            ("TP hit rate",
             f"{s1['tp_rate']:.1f}%", f"{s2['tp_rate']:.1f}%",
             "Straddle" if s1["tp_rate"] > s2["tp_rate"] else "Strangle"),
            ("Avg TP rate/ticker",
             f"{s1['avg_tp_rate']:.1f}%", f"{s2['avg_tp_rate']:.1f}%",
             "Straddle" if s1["avg_tp_rate"] > s2["avg_tp_rate"] else "Strangle"),
            ("Avg days held",
             f"{s1['avg_days_held']:.1f}", f"{s2['avg_days_held']:.1f}",
             "Straddle" if s1["avg_days_held"] < s2["avg_days_held"] else "Strangle"),
            ("Max-hold exits",
             str(s1["max_hold_exits"]), str(s2["max_hold_exits"]),
             "Straddle" if s1["max_hold_exits"] < s2["max_hold_exits"] else "Strangle"),
            ("Tickers w/ trades",
             str(s1["tickers_with_trades"]), str(s2["tickers_with_trades"]),
             None),
        ]

        for label, v1, v2, winner in rows:
            w = f"← {winner}" if winner else ""
            print(f"  {label:<30s} {v1:>15s} {v2:>15s}   {w}")

        print("─" * 70)

        # ─── Per-Ticker Breakdown ────────────────────────────────
        print(f"\n{'═'*70}")
        print("  PER-TICKER COMPARISON (sorted by straddle TP rate)")
        print(f"{'═'*70}")
        print(f"  {'Ticker':<8s} {'Straddle Win%':>14s} {'Straddle TP%':>14s} "
              f"{'Strangle Win%':>14s} {'Strangle TP%':>14s} {'Better':>10s}")
        print("─" * 70)

        strad_df = s1["ticker_results"].set_index("ticker")
        strang_df = s2["ticker_results"].set_index("ticker")
        all_tickers = sorted(set(strad_df.index) | set(strang_df.index))

        for sym in sorted(all_tickers,
                          key=lambda s: strad_df.loc[s, "tp_rate"] if s in strad_df.index else 0,
                          reverse=True):
            s1_wr = f"{strad_df.loc[sym, 'win_rate']:.0f}%" if sym in strad_df.index else "—"
            s1_tp = f"{strad_df.loc[sym, 'tp_rate']:.0f}%" if sym in strad_df.index else "—"
            s2_wr = f"{strang_df.loc[sym, 'win_rate']:.0f}%" if sym in strang_df.index else "—"
            s2_tp = f"{strang_df.loc[sym, 'tp_rate']:.0f}%" if sym in strang_df.index else "—"

            if sym in strad_df.index and sym in strang_df.index:
                better = "Straddle" if strad_df.loc[sym, "tp_rate"] >= strang_df.loc[sym, "tp_rate"] else "Strangle"
            elif sym in strad_df.index:
                better = "Straddle"
            else:
                better = "Strangle"

            print(f"  {sym:<8s} {s1_wr:>14s} {s1_tp:>14s} {s2_wr:>14s} {s2_tp:>14s} {better:>10s}")

        print("─" * 70)

        # ─── Save Results ────────────────────────────────────────
        comparison = {
            "straddle": {
                "total_trades": s1["total_trades"],
                "overall_win_rate": round(s1["overall_win_rate"], 2),
                "avg_win_rate": round(s1["avg_win_rate"], 2),
                "tp_rate": round(s1["tp_rate"], 2),
                "avg_tp_rate": round(s1["avg_tp_rate"], 2),
                "avg_days_held": round(s1["avg_days_held"], 2),
                "max_hold_exits": s1["max_hold_exits"],
                "tickers_with_trades": s1["tickers_with_trades"],
            },
            "strangle": {
                "total_trades": s2["total_trades"],
                "overall_win_rate": round(s2["overall_win_rate"], 2),
                "avg_win_rate": round(s2["avg_win_rate"], 2),
                "tp_rate": round(s2["tp_rate"], 2),
                "avg_tp_rate": round(s2["avg_tp_rate"], 2),
                "avg_days_held": round(s2["avg_days_held"], 2),
                "max_hold_exits": s2["max_hold_exits"],
                "tickers_with_trades": s2["tickers_with_trades"],
            },
        }
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "results", "straddle_vs_strangle_tp12.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Results saved to {out_path}")

"""
Long Strangle: Thursday vs Friday Entry — +12% Take-Profit Exit.

Runs the strangle backtest twice:
  1. Entry only on Thursdays
  2. Entry only on Fridays

Both close positions immediately when they appreciate by +12% or more.

Usage:
    python backtest/strangle_thu_vs_fri.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INITIAL_CAPITAL
from signals.scanner import SCAN_UNIVERSE
from backtest.straddle_vs_strangle_tp import TPBacktester


def run_strangle_weekday(entry_weekday: int, verbose: bool = True) -> dict:
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][entry_weekday]

    print(f"\n{'═'*70}")
    print(f"  LONG STRANGLE — {day_name.upper()} ENTRY + 12% TAKE-PROFIT EXIT")
    print(f"  Universe: {len(SCAN_UNIVERSE)} tickers | Budget: ${INITIAL_CAPITAL}")
    print(f"  OTM width: 5% | Max hold: 15 days")
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
                strategy="strangle",
                take_profit_pct=12.0,
                max_hold_days=15,
                entry_weekday=entry_weekday,
            )
            results = bt.run(prices, cond_vol, treasury, ticker=sym)

            if results["total_trades"] > 0:
                ticker_results.append({
                    "ticker": sym,
                    "trades": results["total_trades"],
                    "win_rate": results["win_rate"],
                    "tp_rate": results["tp_rate"],
                    "tp_exits": results["tp_exits"],
                    "total_pnl": results["total_pnl"],
                    "avg_days": results["avg_days_held"],
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
        print(f"No trades generated.")
        return {}

    import pandas as pd
    df = pd.DataFrame(ticker_results)
    total_trades = int(df["trades"].sum())
    tp_count = sum(1 for t in all_trades if t.exit_reason == "take_profit")
    winners = sum(1 for t in all_trades if t.net_pnl > 0)

    print(f"\n{'─'*70}")
    print(f"  AGGREGATE: LONG STRANGLE — {day_name.upper()} ENTRY")
    print(f"{'─'*70}")
    print(f"  Tickers w/ trades:   {len(ticker_results)} / {len(SCAN_UNIVERSE)}")
    print(f"  Total trades:        {total_trades}")
    print(f"  Overall win rate:    {winners}/{len(all_trades)} ({winners/len(all_trades)*100:.1f}%)")
    print(f"  Take-profit hits:    {tp_count}/{len(all_trades)} ({tp_count/len(all_trades)*100:.1f}%)")
    print(f"  Max-hold exits:      {len(all_trades) - tp_count}")
    print(f"  Avg days held:       {df['avg_days'].mean():.1f}")
    print(f"  Avg win rate/ticker: {df['win_rate'].mean():.1f}%")
    print(f"  Avg TP rate/ticker:  {df['tp_rate'].mean():.1f}%")
    if failed:
        print(f"  Failed tickers:      {len(failed)}")
    print(f"{'─'*70}\n")

    return {
        "weekday": day_name,
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
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  LONG STRANGLE: THURSDAY vs FRIDAY ENTRY")
    print("  Exit: +12% take-profit (no fixed holding period)")
    print("  Historical data: 2022-01-01 → present")
    print("=" * 70)

    thu = run_strangle_weekday(entry_weekday=3, verbose=True)
    fri = run_strangle_weekday(entry_weekday=4, verbose=True)

    if thu and fri:
        print("\n" + "═" * 70)
        print("  HEAD-TO-HEAD: THURSDAY vs FRIDAY (LONG STRANGLE)")
        print("═" * 70)
        print(f"  {'Metric':<30s} {'Thursday':>15s} {'Friday':>15s} {'Winner':>10s}")
        print("─" * 70)

        rows = [
            ("Total trades",
             str(thu["total_trades"]), str(fri["total_trades"]), None),
            ("Overall win rate",
             f"{thu['overall_win_rate']:.1f}%", f"{fri['overall_win_rate']:.1f}%",
             "Thu" if thu["overall_win_rate"] > fri["overall_win_rate"] else "Fri"),
            ("TP hit rate",
             f"{thu['tp_rate']:.1f}%", f"{fri['tp_rate']:.1f}%",
             "Thu" if thu["tp_rate"] > fri["tp_rate"] else "Fri"),
            ("Avg TP rate/ticker",
             f"{thu['avg_tp_rate']:.1f}%", f"{fri['avg_tp_rate']:.1f}%",
             "Thu" if thu["avg_tp_rate"] > fri["avg_tp_rate"] else "Fri"),
            ("Avg days held",
             f"{thu['avg_days_held']:.1f}", f"{fri['avg_days_held']:.1f}",
             "Thu" if thu["avg_days_held"] < fri["avg_days_held"] else "Fri"),
            ("Max-hold exits",
             str(thu["max_hold_exits"]), str(fri["max_hold_exits"]),
             "Thu" if thu["max_hold_exits"] < fri["max_hold_exits"] else "Fri"),
            ("Tickers w/ trades",
             str(thu["tickers_with_trades"]), str(fri["tickers_with_trades"]),
             None),
        ]

        for label, v1, v2, winner in rows:
            w = f"  ← {winner}" if winner else ""
            print(f"  {label:<30s} {v1:>15s} {v2:>15s}{w}")

        print("─" * 70)

        # Save
        out = {
            "thursday": {k: v for k, v in thu.items() if k != "weekday"},
            "friday": {k: v for k, v in fri.items() if k != "weekday"},
        }
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "results", "strangle_thu_vs_fri_tp12.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n✓ Results saved to {out_path}")

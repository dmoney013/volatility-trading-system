"""
Long Strangle — Take-Profit Threshold Optimization.

Tests different TP thresholds: 10%, 12%, 15%, 17%, 20%
Entry: Any weekday with GARCH signal | Max hold: 15 days
Budget: $150 (compounding)

Usage:
    python backtest/strangle_tp_sweep.py
"""
import sys, os, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INITIAL_CAPITAL
from signals.scanner import SCAN_UNIVERSE
from backtest.straddle_vs_strangle_tp import TPBacktester

TP_THRESHOLDS = [10, 12, 15, 17, 20]


def run_threshold(tp_pct: float) -> dict:
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    print(f"\n{'═'*70}")
    print(f"  LONG STRANGLE — +{tp_pct:.0f}% TAKE-PROFIT EXIT")
    print(f"  Universe: {len(SCAN_UNIVERSE)} tickers | Budget: ${INITIAL_CAPITAL}")
    print(f"{'═'*70}\n")

    treasury = fetch_treasury_yield()
    all_trades = []
    ticker_stats = []
    failed = []

    for sym in SCAN_UNIVERSE:
        try:
            prices = fetch_price_data(sym)
            garch = GARCHVolatilityModel()
            garch.fit(prices, verbose=False)
            cond_vol = garch.get_conditional_volatility()

            bt = TPBacktester(
                strategy="strangle",
                take_profit_pct=tp_pct,
                max_hold_days=15,
            )
            r = bt.run(prices, cond_vol, treasury, ticker=sym)

            if r["total_trades"] > 0:
                ticker_stats.append({
                    "ticker": sym,
                    "trades": r["total_trades"],
                    "win_rate": r["win_rate"],
                    "tp_rate": r["tp_rate"],
                    "tp_exits": r["tp_exits"],
                    "avg_days": r["avg_days_held"],
                    "final_capital": r["final_capital"],
                    "return_pct": r["total_return_pct"],
                })
                all_trades.extend(bt.trades)

                print(f"  {sym:6s} | {r['total_trades']:3d} trades | "
                      f"Win: {r['win_rate']:5.1f}% | "
                      f"TP: {r['tp_exits']:3d}/{r['total_trades']:3d} "
                      f"({r['tp_rate']:.0f}%) | "
                      f"Avg hold: {r['avg_days_held']:.1f}d | "
                      f"Final: ${r['final_capital']:.2f}")
        except Exception as e:
            failed.append(sym)

    if not all_trades:
        return {}

    df = pd.DataFrame(ticker_stats)
    total = len(all_trades)
    winners = sum(1 for t in all_trades if t.net_pnl > 0)
    tp_hits = sum(1 for t in all_trades if t.exit_reason == "take_profit")
    days = [t.days_held for t in all_trades]

    # Median final capital across tickers (more robust than mean with compounding)
    median_final = df["final_capital"].median()
    mean_final = df["final_capital"].mean()
    median_return = df["return_pct"].median()

    print(f"\n{'─'*70}")
    print(f"  AGGREGATE: +{tp_pct:.0f}% TAKE-PROFIT")
    print(f"{'─'*70}")
    print(f"  Tickers w/ trades:     {len(ticker_stats)} / {len(SCAN_UNIVERSE)}")
    print(f"  Total trades:          {total}")
    print(f"  Overall win rate:      {winners/total*100:.1f}%")
    print(f"  TP hit rate:           {tp_hits}/{total} ({tp_hits/total*100:.1f}%)")
    print(f"  Avg days held:         {sum(days)/len(days):.1f}")
    print(f"  Median final capital:  ${median_final:.2f}")
    print(f"  Median return/ticker:  {median_return:.1f}%")
    print(f"{'─'*70}\n")

    return {
        "tp_pct": tp_pct,
        "total_trades": total,
        "winners": winners,
        "overall_win_rate": round(winners / total * 100, 2),
        "tp_hits": tp_hits,
        "tp_rate": round(tp_hits / total * 100, 2),
        "avg_days_held": round(sum(days) / len(days), 2),
        "avg_win_rate": round(float(df["win_rate"].mean()), 2),
        "avg_tp_rate": round(float(df["tp_rate"].mean()), 2),
        "median_final_capital": round(float(median_final), 2),
        "mean_final_capital": round(float(mean_final), 2),
        "median_return_pct": round(float(median_return), 2),
        "mean_return_pct": round(float(df["return_pct"].mean()), 2),
        "tickers_tested": len(ticker_stats),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  LONG STRANGLE — TAKE-PROFIT THRESHOLD SWEEP")
    print(f"  Thresholds: {', '.join(f'+{t}%' for t in TP_THRESHOLDS)}")
    print(f"  Budget: ${INITIAL_CAPITAL} | Max hold: 15 days | Any weekday")
    print("=" * 70)

    results = {}
    for tp in TP_THRESHOLDS:
        results[tp] = run_threshold(tp)

    # ─── Comparison Table ────────────────────────────────────────
    valid = {k: v for k, v in results.items() if v}
    if not valid:
        print("No results to compare.")
        sys.exit(1)

    print("\n" + "═" * 90)
    print("  TAKE-PROFIT THRESHOLD COMPARISON — LONG STRANGLE")
    print("═" * 90)
    print(f"  {'Metric':<28s}", end="")
    for tp in TP_THRESHOLDS:
        print(f" {f'+{tp}%':>12s}", end="")
    print(f" {'Best':>10s}")
    print("─" * 90)

    metrics = [
        ("Overall win rate", "overall_win_rate", "%", True),
        ("Avg win rate/ticker", "avg_win_rate", "%", True),
        ("TP hit rate", "tp_rate", "%", True),
        ("Avg TP rate/ticker", "avg_tp_rate", "%", True),
        ("Avg days held", "avg_days_held", "d", False),
        ("Median final capital", "median_final_capital", "$", True),
        ("Median return/ticker", "median_return_pct", "%", True),
        ("Total trades", "total_trades", "", True),
    ]

    for label, key, unit, higher_is_better in metrics:
        vals = []
        print(f"  {label:<28s}", end="")
        for tp in TP_THRESHOLDS:
            v = valid.get(tp, {}).get(key, 0)
            vals.append((tp, v))
            if unit == "$":
                print(f" {'$'+f'{v:,.0f}':>12s}", end="")
            elif unit == "%":
                print(f" {f'{v:.1f}%':>12s}", end="")
            elif unit == "d":
                print(f" {f'{v:.1f}':>12s}", end="")
            else:
                print(f" {str(int(v)):>12s}", end="")

        if higher_is_better:
            best_tp = max(vals, key=lambda x: x[1])[0]
        else:
            best_tp = min(vals, key=lambda x: x[1])[0]
        print(f" {'← +'+str(best_tp)+'%':>10s}")

    print("─" * 90)

    # ─── Winner Summary ──────────────────────────────────────────
    # Determine winner by win rate and by portfolio growth
    best_wr_tp = max(valid.items(), key=lambda x: x[1]["overall_win_rate"])
    best_cap_tp = max(valid.items(), key=lambda x: x[1]["median_final_capital"])
    best_tp_tp = max(valid.items(), key=lambda x: x[1]["tp_rate"])

    print(f"\n  🏆 HIGHEST WIN RATE:      +{best_wr_tp[0]}% → {best_wr_tp[1]['overall_win_rate']:.1f}%")
    print(f"  🏆 HIGHEST TP HIT RATE:   +{best_tp_tp[0]}% → {best_tp_tp[1]['tp_rate']:.1f}%")
    print(f"  🏆 BEST PORTFOLIO GROWTH: +{best_cap_tp[0]}% → ${best_cap_tp[1]['median_final_capital']:,.2f} "
          f"(median, from $150)")

    # Save
    out = {str(k): v for k, v in valid.items()}
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "results", "strangle_tp_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

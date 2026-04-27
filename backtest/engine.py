"""
Long Straddle Backtester — simulates a GARCH-informed long straddle
options strategy using historical price data.

Strategy:
  - Entry: Buy ATM call + ATM put when GARCH forecasts RV > current IV
    (i.e., the market is underpricing future volatility)
  - Exit: Hold for N trading days, then close at market prices
  - Budget: Constrained to $150 initial capital
  - Pricing: Black-Scholes to synthesize historical option prices
    (no free historical options data available)

The long straddle is defined-risk: max loss = premium paid (no naked exposure).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, HOLDING_PERIOD_DAYS, ENTRY_VOL_THRESHOLD,
    MAX_POSITION_PCT, COMMISSION_PER_CONTRACT, TRADING_DAYS,
    RETURN_SCALE, GARCH_P, GARCH_Q, GARCH_O, GARCH_DIST,
)


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes Option Pricing
# ═══════════════════════════════════════════════════════════════════

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def straddle_price(S, K, T, r, sigma):
    """Price of an ATM straddle (call + put at same strike)."""
    return black_scholes_call(S, K, T, r, sigma) + black_scholes_put(S, K, T, r, sigma)


# ═══════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Record of a single straddle trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    spot_entry: float
    spot_exit: float
    strike: float
    entry_price: float      # Total straddle premium paid (per share)
    exit_price: float        # Total straddle value at exit (per share)
    contracts: int
    entry_vol: float         # GARCH vol used for entry pricing
    exit_vol: float          # Vol at exit
    pnl_per_share: float     # Per-share P&L
    pnl_total: float         # Total P&L including contracts × 100
    commission: float
    net_pnl: float           # P&L after commissions
    return_pct: float        # Return as % of capital deployed
    garch_rv_forecast: float # What GARCH predicted
    signal_strength: float   # How strong the entry signal was


# ═══════════════════════════════════════════════════════════════════
# Backtester
# ═══════════════════════════════════════════════════════════════════

class LongStraddleBacktester:
    """
    Backtests a GARCH-informed long straddle strategy.

    Uses Black-Scholes to synthesize historical option prices since
    free historical options data is unavailable. The GARCH conditional
    volatility serves as the implied volatility input.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        holding_period: int = HOLDING_PERIOD_DAYS,
        entry_threshold: float = ENTRY_VOL_THRESHOLD,
        max_position_pct: float = MAX_POSITION_PCT,
        commission: float = COMMISSION_PER_CONTRACT,
    ):
        self.initial_capital = initial_capital
        self.holding_period = holding_period
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
        verbose: bool = True,
    ) -> dict:
        """
        Run the backtest.

        Args:
            prices: OHLCV DataFrame
            conditional_vol: GARCH annualized conditional volatility series
            treasury: Treasury yield DataFrame for risk-free rate
            ticker: Ticker symbol for labeling

        Returns:
            Dictionary with backtest results and statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"LONG STRADDLE BACKTEST: {ticker}")
            print(f"{'='*60}")
            print(f"Initial capital: ${self.initial_capital:.2f}")
            print(f"Holding period:  {self.holding_period} days")
            print(f"Entry threshold: GARCH RV - IV > {self.entry_threshold*100:.1f}%")

        close = prices["Close"]
        # Align all series
        common_idx = conditional_vol.index.intersection(close.index)
        common_idx = common_idx.intersection(treasury.index) if not treasury.empty else common_idx
        common_idx = common_idx.sort_values()

        capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        in_position = False
        position_entry_idx = 0

        # Get risk-free rate series
        if not treasury.empty and "Close" in treasury.columns:
            rf_rate = treasury["Close"].reindex(common_idx, method="ffill") / 100.0
        else:
            rf_rate = pd.Series(0.04, index=common_idx)

        # Compute a proxy for market IV: use rolling realized vol as a stand-in
        # (In reality, IV is forward-looking; we use lagged RV as a proxy
        #  for what the market would price, then check if GARCH predicts higher)
        log_ret = np.log(close / close.shift(1))
        market_iv_proxy = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
        market_iv_proxy = market_iv_proxy.reindex(common_idx, method="ffill")

        for i in range(len(common_idx)):
            date = common_idx[i]
            spot = close.loc[date]
            garch_vol = conditional_vol.loc[date] if date in conditional_vol.index else None

            if garch_vol is None:
                self.equity_curve.append({"date": date, "equity": capital, "in_trade": False})
                continue

            mkt_iv = market_iv_proxy.loc[date] if date in market_iv_proxy.index and not np.isnan(market_iv_proxy.loc[date]) else garch_vol
            r = rf_rate.loc[date] if date in rf_rate.index else 0.04

            # Record equity
            self.equity_curve.append({"date": date, "equity": capital, "in_trade": in_position})

            if in_position:
                # Check if holding period is up
                days_held = i - position_entry_idx
                if days_held >= self.holding_period:
                    # EXIT: close the straddle
                    trade = self._close_position(
                        entry_trade_data, date, spot, garch_vol, r, capital
                    )
                    capital += trade.net_pnl
                    self.trades.append(trade)
                    in_position = False

                    if verbose and len(self.trades) % 10 == 0:
                        print(f"  Trade #{len(self.trades):3d} | {trade.entry_date.date()} → {trade.exit_date.date()} | "
                              f"PnL: ${trade.net_pnl:+.2f} | Capital: ${capital:.2f}")
            else:
                # Check entry signal: GARCH forecasts higher vol than market prices
                vol_spread = garch_vol - mkt_iv
                signal_strength = vol_spread / max(mkt_iv, 0.01)

                if vol_spread > self.entry_threshold and i + self.holding_period < len(common_idx):
                    # ENTRY: buy long straddle
                    T = self.holding_period / TRADING_DAYS  # Time to expiry in years
                    strike = round(spot, 0)  # ATM rounded to nearest dollar

                    # Price the straddle using GARCH vol (what we believe vol will be)
                    # But we pay the market price (priced at market IV)
                    entry_straddle_price = straddle_price(spot, strike, T, r, mkt_iv)

                    if entry_straddle_price <= 0.01:
                        continue

                    # Cost per contract = price × 100 shares
                    cost_per_contract = entry_straddle_price * 100
                    total_commission = self.commission * 2  # Call + Put

                    # How many contracts can we afford?
                    max_spend = capital * self.max_position_pct
                    contracts = int((max_spend - total_commission) / cost_per_contract)

                    if contracts < 1:
                        continue  # Can't afford even 1 contract

                    # Save entry data
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
                        "T_entry": T,
                        "r": r,
                    }

                    in_position = True
                    position_entry_idx = i

        # ─── Compute Statistics ──────────────────────────────────
        results = self._compute_stats(capital, verbose)
        return results

    def _close_position(self, entry_data, exit_date, spot_exit, exit_vol, r, capital):
        """Close a straddle position and compute P&L."""
        strike = entry_data["strike"]
        T_remaining = 0.001  # Nearly expired

        # Value at exit: intrinsic value only (near expiry)
        call_value = max(spot_exit - strike, 0)
        put_value = max(strike - spot_exit, 0)
        exit_straddle_price = call_value + put_value

        contracts = entry_data["contracts"]
        entry_price = entry_data["entry_price"]

        pnl_per_share = exit_straddle_price - entry_price
        pnl_total = pnl_per_share * contracts * 100
        commission = self.commission * 2 * 2  # 2 legs × (open + close)
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
            exit_price=exit_straddle_price,
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
        )

    def _compute_stats(self, final_capital, verbose=True) -> dict:
        """Compute comprehensive backtest statistics."""
        if not self.trades:
            stats = {
                "total_trades": 0,
                "final_capital": final_capital,
                "total_return_pct": (final_capital - self.initial_capital) / self.initial_capital * 100,
                "message": "No trades executed — entry conditions were never met.",
            }
            if verbose:
                print(f"\n⚠ No trades were executed.")
                print(f"  This may mean the entry threshold is too strict,")
                print(f"  or the ticker's vol was consistently priced fairly.")
            return stats

        trades_df = pd.DataFrame([{
            "entry_date": t.entry_date, "exit_date": t.exit_date,
            "spot_entry": t.spot_entry, "spot_exit": t.spot_exit,
            "strike": t.strike, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "contracts": t.contracts,
            "pnl_total": t.pnl_total, "net_pnl": t.net_pnl,
            "return_pct": t.return_pct, "entry_vol": t.entry_vol,
            "exit_vol": t.exit_vol, "garch_rv_forecast": t.garch_rv_forecast,
            "signal_strength": t.signal_strength,
        } for t in self.trades])

        winners = trades_df[trades_df["net_pnl"] > 0]
        losers = trades_df[trades_df["net_pnl"] <= 0]

        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df["drawdown"] = equity_df["equity"] / equity_df["equity"].cummax() - 1

        total_pnl = trades_df["net_pnl"].sum()
        win_rate = len(winners) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winners["net_pnl"].mean() if len(winners) > 0 else 0
        avg_loss = losers["net_pnl"].mean() if len(losers) > 0 else 0
        profit_factor = abs(winners["net_pnl"].sum() / losers["net_pnl"].sum()) if len(losers) > 0 and losers["net_pnl"].sum() != 0 else float("inf")
        max_dd = equity_df["drawdown"].min() * 100 if not equity_df.empty else 0

        stats = {
            "total_trades": len(trades_df),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_trade": trades_df["net_pnl"].max(),
            "worst_trade": trades_df["net_pnl"].min(),
            "profit_factor": profit_factor,
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return_pct": (final_capital - self.initial_capital) / self.initial_capital * 100,
            "max_drawdown_pct": max_dd,
            "avg_holding_days": (trades_df["exit_date"] - trades_df["entry_date"]).dt.days.mean(),
            "trades_df": trades_df,
            "equity_df": equity_df,
            "avg_signal_strength": trades_df["signal_strength"].mean(),
        }

        if verbose:
            print(f"\n{'─'*50}")
            print(f"BACKTEST RESULTS")
            print(f"{'─'*50}")
            print(f"  Total trades:     {stats['total_trades']}")
            print(f"  Win rate:         {stats['win_rate']:.1f}%")
            print(f"  Profit factor:    {stats['profit_factor']:.2f}")
            print(f"  Total P&L:        ${stats['total_pnl']:+.2f}")
            print(f"  Final capital:    ${stats['final_capital']:.2f}")
            print(f"  Total return:     {stats['total_return_pct']:+.1f}%")
            print(f"  Max drawdown:     {stats['max_drawdown_pct']:.1f}%")
            print(f"  Best trade:       ${stats['best_trade']:+.2f}")
            print(f"  Worst trade:      ${stats['worst_trade']:+.2f}")
            print(f"  Avg win:          ${stats['avg_win']:+.2f}")
            print(f"  Avg loss:         ${stats['avg_loss']:+.2f}")

        return stats

    def get_trades_df(self) -> pd.DataFrame:
        """Return all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Entry": t.entry_date.strftime("%Y-%m-%d"),
            "Exit": t.exit_date.strftime("%Y-%m-%d"),
            "Spot In": f"${t.spot_entry:.2f}",
            "Spot Out": f"${t.spot_exit:.2f}",
            "Strike": f"${t.strike:.0f}",
            "Contracts": t.contracts,
            "Premium": f"${t.entry_price:.2f}",
            "Exit Val": f"${t.exit_price:.2f}",
            "Net P&L": f"${t.net_pnl:+.2f}",
            "Return": f"{t.return_pct:+.1f}%",
            "GARCH Vol": f"{t.garch_rv_forecast*100:.1f}%",
            "Mkt IV": f"{t.entry_vol*100:.1f}%",
        } for t in self.trades])

    def get_equity_curve(self) -> pd.DataFrame:
        """Return the equity curve as a DataFrame."""
        return pd.DataFrame(self.equity_curve)


def run_backtest(
    prices: pd.DataFrame,
    conditional_vol: pd.Series,
    treasury: pd.DataFrame,
    ticker: str = "SIM",
    initial_capital: float = INITIAL_CAPITAL,
    holding_period: int = HOLDING_PERIOD_DAYS,
    entry_threshold: float = ENTRY_VOL_THRESHOLD,
    verbose: bool = True,
) -> dict:
    """
    Convenience function to run a backtest with default parameters.
    """
    bt = LongStraddleBacktester(
        initial_capital=initial_capital,
        holding_period=holding_period,
        entry_threshold=entry_threshold,
    )
    results = bt.run(prices, conditional_vol, treasury, ticker, verbose)
    results["backtester"] = bt
    return results


if __name__ == "__main__":
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    # Test backtest on a budget-friendly ticker
    ticker = "F"
    print(f"Testing backtest on {ticker}...")

    prices = fetch_price_data(ticker)
    treasury = fetch_treasury_yield()

    garch = GARCHVolatilityModel()
    garch.fit(prices, verbose=False)
    cond_vol = garch.get_conditional_volatility()

    results = run_backtest(prices, cond_vol, treasury, ticker)

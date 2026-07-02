"""
Long Strangle Backtester — simulates a GARCH-informed long strangle
options strategy using historical price data.

Strategy:
  - Entry: Buy OTM call + OTM put when GARCH forecasts RV > current IV
    (i.e., the market is underpricing future volatility)
  - Strike selection: call_strike = spot × (1 + width), put_strike = spot × (1 - width)
  - Exit: Hold for N trading days, then close at intrinsic value
  - Budget: Constrained to $150 initial capital
  - Pricing: Black-Scholes to synthesize historical option prices

Difference from straddle:
  - Lower premium (both legs are OTM)
  - Needs a bigger move to profit (spot must exceed one of the OTM strikes)
  - Same defined-risk profile: max loss = premium paid
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, STRANGLE_HOLDING_PERIOD_DAYS, STRANGLE_ENTRY_VOL_THRESHOLD,
    STRANGLE_OTM_WIDTH, MAX_POSITION_PCT, COMMISSION_PER_CONTRACT,
    TRADING_DAYS,
)
from backtest.engine import black_scholes_call, black_scholes_put


# ═══════════════════════════════════════════════════════════════════
# Strangle Pricing
# ═══════════════════════════════════════════════════════════════════

def strangle_price(S, K_call, K_put, T, r, sigma):
    """
    Price of a strangle: OTM call at K_call + OTM put at K_put.

    K_call > S (out-of-the-money call)
    K_put  < S (out-of-the-money put)
    """
    return black_scholes_call(S, K_call, T, r, sigma) + black_scholes_put(S, K_put, T, r, sigma)


def select_strangle_strikes(spot, width=STRANGLE_OTM_WIDTH):
    """
    Select OTM call and put strikes for a strangle.

    Args:
        spot: Current stock price
        width: How far OTM as a fraction (0.05 = 5%)

    Returns:
        (call_strike, put_strike) rounded to nearest dollar
    """
    call_strike = round(spot * (1 + width))
    put_strike = round(spot * (1 - width))
    # Ensure strikes are at least $1 apart from spot
    if call_strike <= round(spot):
        call_strike = round(spot) + 1
    if put_strike >= round(spot):
        put_strike = round(spot) - 1
    return call_strike, put_strike


# ═══════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StrangleTrade:
    """Record of a single strangle trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    spot_entry: float
    spot_exit: float
    call_strike: float
    put_strike: float
    entry_price: float       # Total strangle premium paid (per share)
    exit_price: float        # Total strangle value at exit (per share)
    contracts: int
    entry_vol: float         # Vol used for entry pricing (market IV proxy)
    exit_vol: float          # Vol at exit
    pnl_per_share: float
    pnl_total: float
    commission: float
    net_pnl: float
    return_pct: float
    garch_rv_forecast: float
    signal_strength: float
    otm_width: float         # Width used for this trade


# ═══════════════════════════════════════════════════════════════════
# Backtester
# ═══════════════════════════════════════════════════════════════════

class LongStrangleBacktester:
    """
    Backtests a GARCH-informed long strangle strategy.

    Uses Black-Scholes to synthesize historical option prices since
    free historical options data is unavailable. The GARCH conditional
    volatility serves as the implied volatility input.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        holding_period: int = STRANGLE_HOLDING_PERIOD_DAYS,
        entry_threshold: float = STRANGLE_ENTRY_VOL_THRESHOLD,
        max_position_pct: float = MAX_POSITION_PCT,
        commission: float = COMMISSION_PER_CONTRACT,
        otm_width: float = STRANGLE_OTM_WIDTH,
    ):
        self.initial_capital = initial_capital
        self.holding_period = holding_period
        self.entry_threshold = entry_threshold
        self.max_position_pct = max_position_pct
        self.commission = commission
        self.otm_width = otm_width
        self.trades: List[StrangleTrade] = []
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
        Run the strangle backtest.

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
            print(f"LONG STRANGLE BACKTEST: {ticker}")
            print(f"{'='*60}")
            print(f"Initial capital: ${self.initial_capital:.2f}")
            print(f"Holding period:  {self.holding_period} days")
            print(f"Entry threshold: GARCH RV - IV > {self.entry_threshold*100:.1f}%")
            print(f"OTM width:       {self.otm_width*100:.0f}% from spot")

        close = prices["Close"]
        common_idx = conditional_vol.index.intersection(close.index)
        common_idx = common_idx.intersection(treasury.index) if not treasury.empty else common_idx
        common_idx = common_idx.sort_values()

        capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        in_position = False
        position_entry_idx = 0

        # Risk-free rate series
        if not treasury.empty and "Close" in treasury.columns:
            rf_rate = treasury["Close"].reindex(common_idx, method="ffill") / 100.0
        else:
            rf_rate = pd.Series(0.04, index=common_idx)

        # Market IV proxy: rolling realized vol
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

            self.equity_curve.append({"date": date, "equity": capital, "in_trade": in_position})

            if in_position:
                days_held = i - position_entry_idx
                if days_held >= self.holding_period:
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
                vol_spread = garch_vol - mkt_iv
                signal_strength = vol_spread / max(mkt_iv, 0.01)

                if vol_spread > self.entry_threshold and i + self.holding_period < len(common_idx):
                    T = self.holding_period / TRADING_DAYS
                    call_strike, put_strike = select_strangle_strikes(spot, self.otm_width)

                    # Price the strangle at market IV
                    entry_strangle_price = strangle_price(spot, call_strike, put_strike, T, r, mkt_iv)

                    if entry_strangle_price <= 0.01:
                        continue

                    cost_per_contract = entry_strangle_price * 100
                    total_commission = self.commission * 2  # Call + Put

                    max_spend = capital * self.max_position_pct
                    contracts = int((max_spend - total_commission) / cost_per_contract)

                    if contracts < 1:
                        continue

                    entry_trade_data = {
                        "entry_date": date,
                        "ticker": ticker,
                        "spot_entry": spot,
                        "call_strike": call_strike,
                        "put_strike": put_strike,
                        "entry_price": entry_strangle_price,
                        "contracts": contracts,
                        "entry_vol": mkt_iv,
                        "garch_rv_forecast": garch_vol,
                        "signal_strength": signal_strength,
                        "T_entry": T,
                        "r": r,
                    }

                    in_position = True
                    position_entry_idx = i

        results = self._compute_stats(capital, verbose)
        return results

    def _close_position(self, entry_data, exit_date, spot_exit, exit_vol, r, capital):
        """Close a strangle position and compute P&L."""
        call_strike = entry_data["call_strike"]
        put_strike = entry_data["put_strike"]

        # Value at exit: intrinsic value only (near expiry)
        call_value = max(spot_exit - call_strike, 0)
        put_value = max(put_strike - spot_exit, 0)
        exit_strangle_price = call_value + put_value

        contracts = entry_data["contracts"]
        entry_price = entry_data["entry_price"]

        pnl_per_share = exit_strangle_price - entry_price
        pnl_total = pnl_per_share * contracts * 100
        commission = self.commission * 2 * 2  # 2 legs × (open + close)
        net_pnl = pnl_total - commission

        cost_basis = entry_price * contracts * 100 + commission / 2
        return_pct = net_pnl / max(cost_basis, 0.01) * 100

        return StrangleTrade(
            entry_date=entry_data["entry_date"],
            exit_date=exit_date,
            ticker=entry_data["ticker"],
            spot_entry=entry_data["spot_entry"],
            spot_exit=spot_exit,
            call_strike=call_strike,
            put_strike=put_strike,
            entry_price=entry_price,
            exit_price=exit_strangle_price,
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
            otm_width=self.otm_width,
        )

    def _compute_stats(self, final_capital, verbose=True) -> dict:
        """Compute comprehensive backtest statistics."""
        if not self.trades:
            stats = {
                "total_trades": 0,
                "final_capital": final_capital,
                "total_return_pct": (final_capital - self.initial_capital) / self.initial_capital * 100,
                "message": "No trades executed — entry conditions were never met.",
                "winners": 0, "losers": 0, "win_rate": 0,
                "profit_factor": 0, "max_drawdown_pct": 0,
                "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "best_trade": 0, "worst_trade": 0,
                "initial_capital": self.initial_capital,
                "trades_df": pd.DataFrame(),
                "equity_df": pd.DataFrame(self.equity_curve),
            }
            if verbose:
                print(f"\n⚠ No trades were executed.")
                print(f"  This may mean the entry threshold is too strict,")
                print(f"  or the ticker's vol was consistently priced fairly.")
            return stats

        trades_df = pd.DataFrame([{
            "entry_date": t.entry_date, "exit_date": t.exit_date,
            "spot_entry": t.spot_entry, "spot_exit": t.spot_exit,
            "call_strike": t.call_strike, "put_strike": t.put_strike,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "contracts": t.contracts, "pnl_total": t.pnl_total,
            "net_pnl": t.net_pnl, "return_pct": t.return_pct,
            "entry_vol": t.entry_vol, "exit_vol": t.exit_vol,
            "garch_rv_forecast": t.garch_rv_forecast,
            "signal_strength": t.signal_strength,
            "otm_width": t.otm_width,
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
            print(f"STRANGLE BACKTEST RESULTS")
            print(f"{'─'*50}")
            print(f"  OTM width:        {self.otm_width*100:.0f}%")
            print(f"  Total trades:     {stats['total_trades']}")
            print(f"  Win rate:         {stats['win_rate']:.1f}%")
            print(f"  Profit factor:    {stats['profit_factor']:.2f}" if stats['profit_factor'] < 100 else f"  Profit factor:    ∞")
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
            "Call K": f"${t.call_strike:.0f}",
            "Put K": f"${t.put_strike:.0f}",
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


def run_strangle_backtest(
    prices: pd.DataFrame,
    conditional_vol: pd.Series,
    treasury: pd.DataFrame,
    ticker: str = "SIM",
    initial_capital: float = INITIAL_CAPITAL,
    holding_period: int = STRANGLE_HOLDING_PERIOD_DAYS,
    entry_threshold: float = STRANGLE_ENTRY_VOL_THRESHOLD,
    otm_width: float = STRANGLE_OTM_WIDTH,
    verbose: bool = True,
) -> dict:
    """
    Convenience function to run a strangle backtest with default parameters.
    """
    bt = LongStrangleBacktester(
        initial_capital=initial_capital,
        holding_period=holding_period,
        entry_threshold=entry_threshold,
        otm_width=otm_width,
    )
    results = bt.run(prices, conditional_vol, treasury, ticker, verbose)
    results["backtester"] = bt
    return results


if __name__ == "__main__":
    from data.fetcher import fetch_price_data, fetch_treasury_yield
    from models.garch_model import GARCHVolatilityModel

    ticker = "F"
    print(f"Testing strangle backtest on {ticker}...")

    prices = fetch_price_data(ticker)
    treasury = fetch_treasury_yield()

    garch = GARCHVolatilityModel()
    garch.fit(prices, verbose=False)
    cond_vol = garch.get_conditional_volatility()

    results = run_strangle_backtest(prices, cond_vol, treasury, ticker)

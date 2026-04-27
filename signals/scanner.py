"""
Live Scanner — Scans a broad universe of tickers for the best
GARCH-signaled straddle opportunities within a given budget.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.garch_model import GARCHVolatilityModel
from data.fetcher import fetch_price_data
from config import TRADING_DAYS


SCAN_UNIVERSE = [
    'F', 'SOFI', 'NIO', 'RIVN', 'SNAP', 'MARA', 'PLUG', 'LCID',
    'AMC', 'GME', 'BB', 'CLOV', 'OPEN', 'RIOT', 'SNDL', 'GEVO',
    'ACHR', 'RGTI', 'QUBT', 'FCEL', 'CHPT', 'QS', 'ENVX',
    'RUN', 'XPEV', 'UPST', 'SKLZ', 'FUBO', 'CLSK', 'HIMS',
    'AAL', 'NCLH', 'PATH', 'BLNK', 'T', 'INTC', 'PFE', 'CCL',
    'PYPL', 'DKNG', 'HOOD', 'SIRI',
]


def scan_for_opportunities(budget=150.0, top_n=8):
    """
    Scan the universe for affordable straddles with positive GARCH signals.

    Returns a list of dicts sorted by GARCH spread (strongest signal first).
    """
    target_date = datetime.now() + timedelta(days=14)
    results = []

    for sym in SCAN_UNIVERSE:
        try:
            tk = yf.Ticker(sym)
            exps = tk.options
            if not exps:
                continue
            best_exp = min(exps, key=lambda x: abs(
                datetime.strptime(x, '%Y-%m-%d') - target_date))

            chain = tk.option_chain(best_exp)
            prices = fetch_price_data(sym)
            spot = prices['Close'].iloc[-1]
            atm = round(spot)

            # Try ATM and ±1 strikes
            for strike in [atm - 1, atm, atm + 1]:
                c = chain.calls[chain.calls['strike'] == strike]
                p = chain.puts[chain.puts['strike'] == strike]
                if c.empty or p.empty:
                    continue
                c = c.iloc[0]
                p = p.iloc[0]

                # Use lastPrice (market may be closed)
                c_price = c['lastPrice'] if c['lastPrice'] > 0.01 else (
                    c['bid'] + c['ask']) / 2
                p_price = p['lastPrice'] if p['lastPrice'] > 0.01 else (
                    p['bid'] + p['ask']) / 2
                if c_price < 0.03 or p_price < 0.03:
                    continue

                straddle_cost = (c_price + p_price) * 100 + 1.30
                if straddle_cost > budget or straddle_cost < 10:
                    continue

                contracts = int((budget - 1.30) / straddle_cost)
                if contracts < 1:
                    continue

                c_vol = c['volume'] if not pd.isna(c['volume']) else 0
                p_vol = p['volume'] if not pd.isna(p['volume']) else 0
                avg_iv = (c['impliedVolatility'] + p['impliedVolatility']) / 2

                # GARCH signal
                garch = GARCHVolatilityModel()
                garch.fit(prices, verbose=False)
                cond_vol = garch.get_conditional_volatility()
                garch_rv = cond_vol.iloc[-1]
                log_ret = np.log(
                    prices['Close'] / prices['Close'].shift(1))
                mkt_iv = log_ret.rolling(21).std().iloc[-1] * np.sqrt(
                    TRADING_DAYS)
                spread = garch_rv - mkt_iv

                results.append({
                    'ticker': sym,
                    'spot': round(spot, 2),
                    'strike': int(strike),
                    'expiry': best_exp,
                    'call_price': round(c_price, 2),
                    'put_price': round(p_price, 2),
                    'straddle_cost': round(straddle_cost, 2),
                    'contracts': contracts,
                    'total_cost': round(straddle_cost * contracts + 1.30, 2),
                    'garch_rv': round(garch_rv, 4),
                    'mkt_iv': round(mkt_iv, 4),
                    'spread': round(spread, 4),
                    'signal_strength': round(
                        spread / max(mkt_iv, 0.01), 3),
                    'option_iv': round(avg_iv, 4),
                    'call_volume': int(c_vol),
                    'put_volume': int(p_vol),
                    'liquidity': int(c_vol + p_vol),
                })
                break  # Best strike per ticker

        except Exception:
            pass

    # Sort by spread (strongest signal first), only positive signals
    results.sort(key=lambda x: x['spread'], reverse=True)
    return results[:top_n]


if __name__ == "__main__":
    recs = scan_for_opportunities(budget=150.0, top_n=8)
    for i, r in enumerate(recs):
        print(f"{i+1}. {r['ticker']} ${r['strike']} straddle | "
              f"GARCH {r['garch_rv']:.1%} vs MktIV {r['mkt_iv']:.1%} | "
              f"Spread {r['spread']:+.1%} | ${r['total_cost']:.2f}")

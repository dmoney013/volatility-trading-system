"""
GARCH Futures Model (fv3) — Adapted from the options v3 model for
bracket breakout trading on /MES (Micro E-mini S&P 500).

Key differences from options v3:
  - No implied volatility — signal is vol ratio (current GARCH vol / recent avg GARCH vol)
  - Shorter fit window (90 days) for bracket sizing
  - Tighter persistence dampening threshold (0.90 vs 0.95)
  - Lower calibration scale (1.0–1.15x vs 1.3x for stocks)
  - ATR-based bracket width with a floor at 0.5× ATR(14)
  - Vol ratio gating: deploy brackets when GARCH vol is LOW (< 0.8× avg)
    — vol expansion follows compression (mean-reversion edge)

Safeguards implemented (from conversation):
  1. 90-day fit window for bracket sizing (regime-responsive)
  2. Persistence dampening at α+β > 0.90 (tighter than options 0.95)
  3. Calibration scale tunable via backtest (default 1.0)
  4. RTH-only data fitting (yfinance daily bars are already RTH)
  5. Inverted vol ratio signal (< 0.8×) — trade quiet days, vol expansion follows compression
  6. ATR floor (0.5× ATR14) — never set brackets inside normal noise
"""
import numpy as np
import pandas as pd
from arch import arch_model

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── fv3 Config ─────────────────────────────────────────────────────
FV3_GARCH_P = 1
FV3_GARCH_Q = 1
FV3_GARCH_O = 1              # GJR asymmetric term
FV3_GARCH_DIST = "t"         # Student-t for fat tails
FV3_RETURN_SCALE = 100        # Numerical stability

FV3_FIT_WINDOW = 90           # Safeguard #1: 90-day window for bracket sizing
                              # S&P regime shifts faster than single stocks
FV3_PERSISTENCE_THRESHOLD = 0.90  # Safeguard #2: tighter than options (0.95)
FV3_CALIBRATION_SCALE = 1.0       # Safeguard #3: default 1.0, tune via backtest
FV3_VOL_RATIO_MAX = 0.8           # Safeguard #5: deploy brackets when vol ratio < this
                                  # (inverted signal: quiet days predict breakouts)
FV3_ATR_FLOOR_MULT = 0.5          # Safeguard #6: bracket >= 0.5 × ATR(14)
FV3_ATR_PERIOD = 14               # ATR lookback
FV3_TRADING_DAYS = 252


class GARCHFuturesModel:
    """
    GARCH-based futures bracket model (fv3).

    Produces:
      - Vol ratio signal: should we trade today?
      - Bracket width: how far from spot to place stop orders
      - ATR context: current noise floor
    """

    def __init__(self, calibration_scale=None, fit_window=None,
                 persistence_threshold=None, vol_ratio_max=None,
                 atr_floor_mult=None):
        self.calibration_scale = calibration_scale or FV3_CALIBRATION_SCALE
        self.fit_window = fit_window or FV3_FIT_WINDOW
        self.persistence_threshold = persistence_threshold or FV3_PERSISTENCE_THRESHOLD
        self.vol_ratio_max = vol_ratio_max or FV3_VOL_RATIO_MAX
        self.atr_floor_mult = atr_floor_mult or FV3_ATR_FLOOR_MULT

        self.best_result = None
        self.best_model_name = None
        self.persistence = None
        self.dampened = False
        self.rv_weight = 0.0
        self.predicted_daily_vol = None  # decimal
        self.atr14 = None                # in price points
        self.vol_ratio = None            # today's GARCH vol / recent avg GARCH vol
        self.avg_cond_vol = None         # mean conditional vol over fit window

    def compute_atr(self, prices: pd.DataFrame, period: int = None) -> float:
        """
        Compute ATR(14) from OHLC data.

        True Range = max(H-L, |H-prevC|, |L-prevC|)
        ATR = rolling mean of True Range over 'period' days.

        Returns ATR in absolute price points.
        """
        if period is None:
            period = FV3_ATR_PERIOD

        high = prices['High']
        low = prices['Low']
        close = prices['Close']
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr

    def fit(self, prices: pd.DataFrame, verbose: bool = False) -> dict:
        """
        Fit GARCH model on recent price data and compute bracket signals.

        Args:
            prices: DataFrame with Open, High, Low, Close, Volume columns.
                    Should contain enough history (fit_window + ATR period).
            verbose: Print diagnostics.

        Returns:
            Dict with model info, vol ratio, bracket width, ATR.
        """
        # ─── Compute ATR from full available data ──────────────────
        self.atr14 = self.compute_atr(prices)

        # ATR as implied daily vol (fraction of spot)
        spot = prices['Close'].iloc[-1]
        atr_daily_vol = self.atr14 / spot  # decimal

        # ─── Fit GARCH on truncated window ─────────────────────────
        log_ret = np.log(prices['Close'] / prices['Close'].shift(1)).dropna()

        # Safeguard #1: truncate to fit window
        if len(log_ret) > self.fit_window:
            log_ret = log_ret.iloc[-self.fit_window:]

        returns_scaled = log_ret * FV3_RETURN_SCALE

        # Fit symmetric GARCH(1,1)
        sym_model = arch_model(
            returns_scaled, vol="Garch",
            p=FV3_GARCH_P, q=FV3_GARCH_Q,
            dist=FV3_GARCH_DIST,
        )
        sym_result = sym_model.fit(disp="off", show_warning=False)

        # Fit asymmetric GJR-GARCH(1,1,1)
        asym_model = arch_model(
            returns_scaled, vol="Garch",
            p=FV3_GARCH_P, o=FV3_GARCH_O, q=FV3_GARCH_Q,
            dist=FV3_GARCH_DIST,
        )
        asym_result = asym_model.fit(disp="off", show_warning=False)

        # Model selection via AIC
        if asym_result.aic < sym_result.aic:
            self.best_result = asym_result
            self.best_model_name = f"GJR-GARCH({FV3_GARCH_P},{FV3_GARCH_O},{FV3_GARCH_Q})"
        else:
            self.best_result = sym_result
            self.best_model_name = f"GARCH({FV3_GARCH_P},{FV3_GARCH_Q})"

        # ─── Persistence Dampening (Safeguard #2) ─────────────────
        params = self.best_result.params
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        gamma = params.get('gamma[1]', 0)
        self.persistence = alpha + beta + gamma / 2

        # Extract GARCH conditional vol series (daily, decimal)
        cond_vol_series = self.best_result.conditional_volatility / FV3_RETURN_SCALE
        cond_vol_daily = cond_vol_series.iloc[-1]  # today's prediction
        self.avg_cond_vol = cond_vol_series.mean()  # average over fit window

        if self.persistence > self.persistence_threshold:
            # Blend with recent realized vol
            raw_returns = returns_scaled / FV3_RETURN_SCALE
            recent_rv_daily = raw_returns.iloc[-30:].std()

            # Linear blend: 0% RV at threshold → 60% RV at persistence=1.0
            rv_weight = min(0.60, (self.persistence - self.persistence_threshold)
                           / (1.0 - self.persistence_threshold) * 0.60)
            garch_weight = 1.0 - rv_weight

            self.predicted_daily_vol = garch_weight * cond_vol_daily + rv_weight * recent_rv_daily
            self.dampened = True
            self.rv_weight = rv_weight
        else:
            self.predicted_daily_vol = cond_vol_daily
            self.dampened = False
            self.rv_weight = 0.0
        # ─── Vol Ratio Signal ─────────────────────────────────────
        # Compute vol ratio BEFORE applying calibration scale.
        # Vol ratio = today's GARCH conditional vol / mean GARCH conditional vol.
        # This is a regime detection signal — independent of bracket width scaling.
        # Calibration scale only affects bracket width (how wide to set stops),
        # NOT the trade/no-trade decision.
        if self.avg_cond_vol > 0:
            self.vol_ratio = self.predicted_daily_vol / self.avg_cond_vol
        else:
            self.vol_ratio = 0.0

        # Apply calibration scale AFTER vol ratio (Safeguard #3)
        # This only affects bracket width via predicted_daily_vol → compute_bracket
        self.predicted_daily_vol *= self.calibration_scale

        if verbose:
            print(f"\n{'─'*50}")
            print(f"fv3 Futures GARCH Model")
            print(f"{'─'*50}")
            print(f"Model:       {self.best_model_name}")
            print(f"Fit window:  {min(len(log_ret) + 1, self.fit_window)} days")
            print(f"Persistence: {self.persistence:.4f}"
                  f"{'  [dampened]' if self.dampened else ''}")
            print(f"Spot:        {spot:.2f}")
            print(f"ATR(14):     {self.atr14:.2f} pts ({atr_daily_vol*100:.2f}%)")
            print(f"GARCH daily: {self.predicted_daily_vol*100:.2f}%")
            print(f"GARCH avg:   {self.avg_cond_vol*100:.2f}%")
            print(f"Vol ratio:   {self.vol_ratio:.2f}x (today/avg)")
            print(f"Signal:      {'✅ TRADE (quiet → breakout)' if self.vol_ratio <= self.vol_ratio_max else '❌ SIT (vol already elevated)'}")

        return {
            'model': self.best_model_name,
            'persistence': self.persistence,
            'dampened': self.dampened,
            'atr14': self.atr14,
            'predicted_daily_vol': self.predicted_daily_vol,
            'vol_ratio': self.vol_ratio,
            'has_signal': self.vol_ratio <= self.vol_ratio_max,
        }

    def compute_bracket(self, spot: float) -> dict:
        """
        Compute bracket levels for a given spot price.

        Bracket width = atr_floor_mult × ATR(14).
        GARCH does NOT set the width — it only decides WHETHER to trade.
        ATR sets WHERE the brackets go (the noise floor).

        This implements the core design principle:
          GARCH → trade/no-trade signal (vol expansion detection)
          ATR   → bracket placement (beyond normal noise)

        Args:
            spot: Current price (Open at 9:30 AM).

        Returns:
            Dict with bracket_upper, bracket_lower, bracket_width,
            and the signal decision.
        """
        if self.predicted_daily_vol is None or self.atr14 is None:
            raise RuntimeError("Must call fit() before compute_bracket()")

        # Bracket width = fraction of ATR (always)
        bracket_width = self.atr14 * self.atr_floor_mult

        bracket_upper = spot + bracket_width
        bracket_lower = spot - bracket_width

        # Also compute what GARCH would have suggested (for analysis)
        garch_width = spot * self.predicted_daily_vol

        return {
            'spot': round(spot, 2),
            'bracket_upper': round(bracket_upper, 2),
            'bracket_lower': round(bracket_lower, 2),
            'bracket_width': round(bracket_width, 2),
            'garch_width': round(garch_width, 2),
            'atr_width': round(bracket_width, 2),
            'used_floor': True,  # always ATR-based now
            'vol_ratio': round(self.vol_ratio, 3),
            'has_signal': self.vol_ratio <= self.vol_ratio_max,
        }


if __name__ == "__main__":
    from data.fetcher import fetch_price_data

    prices = fetch_price_data("SPY")
    model = GARCHFuturesModel()
    result = model.fit(prices, verbose=True)
    bracket = model.compute_bracket(prices['Close'].iloc[-1])

    print(f"\nBracket levels:")
    print(f"  Upper: {bracket['bracket_upper']}")
    print(f"  Lower: {bracket['bracket_lower']}")
    print(f"  Width: {bracket['bracket_width']} pts")
    print(f"  GARCH width: {bracket['garch_width']} | ATR floor: {bracket['atr_floor']}")
    print(f"  Used floor: {bracket['used_floor']}")

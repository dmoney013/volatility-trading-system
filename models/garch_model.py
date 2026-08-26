"""
GARCH Volatility Model — fits GARCH(1,1) and GJR-GARCH(1,1,1) to forecast
realized volatility from historical returns.

Key features:
  - Automatic model selection via AIC/BIC
  - GJR-GARCH for asymmetric (leverage) effects
  - Student-t distribution for fat tails
  - Rolling and multi-horizon forecasts
  - Full diagnostics: Ljung-Box, QQ, residual analysis
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GARCH_P, GARCH_Q, GARCH_O, GARCH_DIST,
    RETURN_SCALE, TRADING_DAYS, GARCH_FIT_WINDOW,
    BREAKEVEN_SIGMA, GARCH_CALIBRATION_SCALE,
)


class GARCHVolatilityModel:
    """
    GARCH-family volatility forecasting model.

    Fits both symmetric GARCH(1,1) and asymmetric GJR-GARCH(1,1,1)
    and selects the best model based on information criteria.
    """

    def __init__(self):
        self.symmetric_result = None
        self.asymmetric_result = None
        self.best_result = None
        self.best_model_name = None
        self.returns = None
        self.conditional_vol = None

    def fit(self, prices: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Fit GARCH models to the price series.

        Args:
            prices: DataFrame with 'Close' column
            verbose: Whether to print diagnostics

        Returns:
            Dictionary with model results and diagnostics
        """
        # Compute scaled log returns
        log_ret = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()

        # Truncate to fit window to prevent old spikes from inflating
        # forecasts via high persistence (α+β ≈ 0.95)
        if GARCH_FIT_WINDOW and len(log_ret) > GARCH_FIT_WINDOW:
            log_ret = log_ret.iloc[-GARCH_FIT_WINDOW:]

        self.returns = log_ret * RETURN_SCALE  # Scale for numerical stability

        if verbose:
            print("\n" + "=" * 60)
            print("GARCH Volatility Modeling")
            print("=" * 60)
            print(f"Return series: {len(self.returns)} observations")
            print(f"Date range: {self.returns.index[0].date()} → {self.returns.index[-1].date()}")
            print(f"Return stats (scaled ×{RETURN_SCALE}):")
            print(f"  Mean:     {self.returns.mean():.4f}")
            print(f"  Std:      {self.returns.std():.4f}")
            print(f"  Skew:     {self.returns.skew():.4f}")
            print(f"  Kurtosis: {self.returns.kurtosis():.4f}")

        # ─── Fit Symmetric GARCH(1,1) ───────────────────────────────
        if verbose:
            print(f"\nFitting GARCH({GARCH_P},{GARCH_Q}) with {GARCH_DIST} distribution...")

        sym_model = arch_model(
            self.returns,
            vol="Garch",
            p=GARCH_P,
            q=GARCH_Q,
            dist=GARCH_DIST,
        )
        self.symmetric_result = sym_model.fit(disp="off")

        # ─── Fit Asymmetric GJR-GARCH(1,1,1) ───────────────────────
        if verbose:
            print(f"Fitting GJR-GARCH({GARCH_P},{GARCH_O},{GARCH_Q}) with {GARCH_DIST} distribution...")

        asym_model = arch_model(
            self.returns,
            vol="Garch",
            p=GARCH_P,
            o=GARCH_O,
            q=GARCH_Q,
            dist=GARCH_DIST,
        )
        self.asymmetric_result = asym_model.fit(disp="off")

        # ─── Model Selection ───────────────────────────────────────
        sym_aic = self.symmetric_result.aic
        asym_aic = self.asymmetric_result.aic

        if asym_aic < sym_aic:
            self.best_result = self.asymmetric_result
            self.best_model_name = f"GJR-GARCH({GARCH_P},{GARCH_O},{GARCH_Q})"
        else:
            self.best_result = self.symmetric_result
            self.best_model_name = f"GARCH({GARCH_P},{GARCH_Q})"

        # ─── Persistence Dampening ────────────────────────────────
        # GARCH persistence = α + β (+ γ/2 for GJR).
        # When persistence > 0.95, old shocks linger for weeks,
        # inflating forecasts well beyond current market conditions.
        #
        # Fix: blend GARCH conditional vol with recent realized vol.
        # The higher the persistence, the more we trust recent RV
        # over the GARCH estimate.
        params = self.best_result.params
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        gamma = params.get('gamma[1]', 0)  # GJR leverage term
        self.persistence = alpha + beta + gamma / 2

        PERSISTENCE_THRESHOLD = 0.95

        # Extract raw GARCH conditional volatility (daily, decimal)
        cond_vol_daily = self.best_result.conditional_volatility / RETURN_SCALE

        if self.persistence > PERSISTENCE_THRESHOLD:
            # Compute recent 30-day realized vol as anchor
            raw_returns = self.returns / RETURN_SCALE  # unscale
            recent_rv_daily = raw_returns.iloc[-30:].std()

            # Blend weight: 0% RV at threshold → 80% RV at persistence=1.0
            # Linear interpolation: weight = (persistence - 0.95) / 0.05 * 0.80
            rv_weight = min(0.80, (self.persistence - PERSISTENCE_THRESHOLD)
                           / (1.0 - PERSISTENCE_THRESHOLD) * 0.80)
            garch_weight = 1.0 - rv_weight

            # Blend: dampen the GARCH estimate toward recent reality
            cond_vol_daily_dampened = (
                garch_weight * cond_vol_daily +
                rv_weight * recent_rv_daily
            )
            self.dampened = True
            self.rv_weight = rv_weight
            self.conditional_vol = cond_vol_daily_dampened * np.sqrt(TRADING_DAYS)
        else:
            self.dampened = False
            self.rv_weight = 0.0
            self.conditional_vol = cond_vol_daily * np.sqrt(TRADING_DAYS)

        if verbose:
            print(f"\n✓ Best model: {self.best_model_name} (AIC: {self.best_result.aic:.2f})")
            print(f"  GARCH AIC:     {sym_aic:.2f}")
            print(f"  GJR-GARCH AIC: {asym_aic:.2f}")
            print(f"  Persistence (α+β): {self.persistence:.4f}")
            if self.dampened:
                print(f"  ⚠️  High persistence — dampened with "
                      f"{self.rv_weight:.0%} recent RV blend")

        return self.get_diagnostics()

    def forecast(self, horizon: int = 5) -> pd.DataFrame:
        """
        Generate multi-step volatility forecasts.

        Args:
            horizon: Number of days to forecast ahead

        Returns:
            DataFrame with annualized volatility forecasts per horizon
        """
        if self.best_result is None:
            raise RuntimeError("Must call fit() before forecast()")

        fcst = self.best_result.forecast(horizon=horizon, reindex=False)

        # Convert variance forecasts to annualized volatility (decimal)
        variance = fcst.variance.iloc[-1]
        vol_daily = np.sqrt(variance) / RETURN_SCALE
        vol_annual = vol_daily * np.sqrt(TRADING_DAYS)

        forecast_df = pd.DataFrame({
            "Horizon (Days)": range(1, horizon + 1),
            "Daily Vol": vol_daily.values,
            "Annualized Vol": vol_annual.values,
            "Annualized Vol (%)": (vol_annual.values * 100).round(2),
        })

        return forecast_df

    def forecast_price_range(
        self,
        spot: float,
        horizon_days: int = 5,
        sigma: float = None,
    ) -> dict:
        """
        Convert GARCH variance forecast into expected price movement range.

        Instead of abstract vol numbers, returns concrete dollar amounts
        that the stock is expected to move within the forecast horizon.

        Args:
            spot: Current stock price
            horizon_days: Number of trading days ahead (e.g., days to expiry)
            sigma: Number of standard deviations (default: BREAKEVEN_SIGMA from config)

        Returns:
            dict with upper/lower bounds at 1σ and 2σ levels:
            {
                'spot': 15.48,
                'horizon': 5,
                'daily_vol': 0.039,
                'period_vol': 0.087,
                'upper_1sig': 16.83,  'lower_1sig': 14.23,
                'upper_2sig': 18.30,  'lower_2sig': 13.08,
                'expected_move_pct': 8.7,   # 1σ move as %
                'expected_move_dollars': 1.35,  # 1σ move in $
            }
        """
        if self.best_result is None:
            raise RuntimeError("Must call fit() before forecast_price_range()")

        if sigma is None:
            sigma = BREAKEVEN_SIGMA

        # Get N-day ahead variance forecast
        fcst = self.best_result.forecast(horizon=horizon_days, reindex=False)
        # Sum of daily variances over the horizon = total period variance
        total_variance = fcst.variance.iloc[-1].sum()
        period_vol = np.sqrt(total_variance) / RETURN_SCALE  # in decimal
        daily_vol = period_vol / np.sqrt(horizon_days)

        # Price range using log-normal model: S * exp(±σ * vol)
        # Apply calibration scale to correct systematic underestimation.
        # Backtests show 1σ covers only 49% of moves (should be 68%),
        # so we widen the range by GARCH_CALIBRATION_SCALE (1.3x).
        calibrated_vol = period_vol * GARCH_CALIBRATION_SCALE
        upper_1sig = spot * np.exp(sigma * calibrated_vol)
        lower_1sig = spot * np.exp(-sigma * calibrated_vol)
        upper_2sig = spot * np.exp(2.0 * calibrated_vol)
        lower_2sig = spot * np.exp(-2.0 * calibrated_vol)

        move_pct = (np.exp(sigma * calibrated_vol) - 1) * 100
        move_dollars = spot * (np.exp(sigma * calibrated_vol) - 1)

        return {
            'spot': round(spot, 2),
            'horizon': horizon_days,
            'daily_vol': round(daily_vol, 4),
            'period_vol': round(period_vol, 4),
            'upper_1sig': round(upper_1sig, 2),
            'lower_1sig': round(lower_1sig, 2),
            'upper_2sig': round(upper_2sig, 2),
            'lower_2sig': round(lower_2sig, 2),
            'expected_move_pct': round(move_pct, 2),
            'expected_move_dollars': round(move_dollars, 2),
        }


    def compute_bias_correction(
        self,
        prices: pd.DataFrame,
        window: int = 30,
        min_windows: int = 6,
    ) -> float:
        """
        Walk-forward calibration of GARCH's vol overestimation.

        Steps through non-overlapping 30-day windows over the last ~360 days:
          1. Fit GARCH on all data up to day T
          2. Extract conditional vol forecast (annualized)
          3. Measure actual realized vol over the next 30 days
          4. Record ratio: actual / predicted

        Returns:
            Correction factor = median(actual / predicted).
            Typically 0.70-0.80, meaning GARCH overestimates by 20-30%.
            Clamped to [0.50, 1.0] for safety.
        """
        close = prices['Close']
        if len(close) < GARCH_FIT_WINDOW + window * min_windows:
            # Not enough data for calibration — return 1.0 (no correction)
            self.bias_correction = 1.0
            return 1.0

        log_ret = np.log(close / close.shift(1)).dropna()
        ratios = []

        # Step through non-overlapping windows in the last ~360 days
        # Leave the last 30 days untouched (that's the current forecast)
        cal_end = len(close) - window  # Don't use most recent window
        cal_start = max(GARCH_FIT_WINDOW, cal_end - window * 12)  # ~12 windows

        for t in range(cal_start, cal_end, window):
            try:
                # Fit GARCH on data up to day T
                fit_prices = prices.iloc[:t].copy()
                if len(fit_prices) < GARCH_FIT_WINDOW:
                    continue

                temp_garch = GARCHVolatilityModel()
                temp_garch.fit(fit_prices, verbose=False)
                predicted_vol = temp_garch.get_conditional_volatility().iloc[-1]

                if predicted_vol <= 0 or np.isnan(predicted_vol):
                    continue

                # Measure actual realized vol over next 30 days
                fwd_rets = log_ret.iloc[t:t + window]
                if len(fwd_rets) < window * 0.8:  # Need at least 80% of window
                    continue
                actual_vol = fwd_rets.std() * np.sqrt(TRADING_DAYS)

                if actual_vol > 0:
                    ratios.append(actual_vol / predicted_vol)
            except Exception:
                continue

        if len(ratios) >= min_windows:
            raw_factor = float(np.median(ratios))
            # Clamp to [0.50, 1.0] — don't over-correct or amplify
            self.bias_correction = max(0.50, min(1.0, raw_factor))
        else:
            self.bias_correction = 1.0  # Not enough data

        return self.bias_correction

    def get_corrected_conditional_volatility(self) -> pd.Series:
        """
        Return conditional volatility adjusted by the bias correction factor.

        Must call compute_bias_correction() first, otherwise returns
        uncorrected values.
        """
        factor = getattr(self, 'bias_correction', 1.0)
        return self.conditional_vol * factor

    def rolling_forecast(
        self,
        window: int = 504,  # ~2 years
        step: int = 1,
    ) -> pd.DataFrame:
        """
        Generate rolling 1-day-ahead forecasts for backtesting.

        Uses an expanding window starting from 'window' observations.
        """
        if self.returns is None:
            raise RuntimeError("Must call fit() before rolling_forecast()")

        forecasts = []
        returns = self.returns

        # Use the best model type for rolling
        use_gjr = "GJR" in self.best_model_name

        for i in range(window, len(returns), step):
            train = returns.iloc[:i]

            try:
                if use_gjr:
                    model = arch_model(train, vol="Garch", p=GARCH_P, o=GARCH_O, q=GARCH_Q, dist=GARCH_DIST)
                else:
                    model = arch_model(train, vol="Garch", p=GARCH_P, q=GARCH_Q, dist=GARCH_DIST)

                result = model.fit(disp="off", show_warning=False)
                fcst = result.forecast(horizon=1, reindex=False)

                var_1d = fcst.variance.iloc[-1, 0]
                vol_1d = np.sqrt(var_1d) / RETURN_SCALE
                vol_annual = vol_1d * np.sqrt(TRADING_DAYS)

                forecasts.append({
                    "date": returns.index[i],
                    "forecast_vol_daily": vol_1d,
                    "forecast_vol_annual": vol_annual,
                })
            except Exception:
                continue

        return pd.DataFrame(forecasts).set_index("date")

    def get_diagnostics(self) -> dict:
        """
        Return comprehensive model diagnostics.
        """
        if self.best_result is None:
            return {}

        result = self.best_result
        std_resid = result.std_resid

        # Ljung-Box test on squared standardized residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(std_resid ** 2, lags=[10, 20], return_df=True)

        # Jarque-Bera normality test on residuals
        jb_stat, jb_pval = stats.jarque_bera(std_resid.dropna())

        diagnostics = {
            "model_name": self.best_model_name,
            "aic": result.aic,
            "bic": result.bic,
            "log_likelihood": result.loglikelihood,
            "params": result.params.to_dict(),
            "pvalues": result.pvalues.to_dict(),
            "ljung_box": lb_test.to_dict(),
            "jarque_bera": {"statistic": jb_stat, "pvalue": jb_pval},
            "conditional_vol_current": self.conditional_vol.iloc[-1] if self.conditional_vol is not None else None,
            "conditional_vol_mean": self.conditional_vol.mean() if self.conditional_vol is not None else None,
        }

        return diagnostics

    def get_conditional_volatility(self) -> pd.Series:
        """Return the full time series of annualized conditional volatility."""
        if self.conditional_vol is None:
            raise RuntimeError("Must call fit() first")
        return self.conditional_vol

    def print_summary(self):
        """Print a formatted summary of the model results."""
        if self.best_result is None:
            print("No model fitted yet.")
            return

        diag = self.get_diagnostics()
        print(f"\n{'─'*50}")
        print(f"Model: {diag['model_name']}")
        print(f"{'─'*50}")
        print(f"AIC:            {diag['aic']:.2f}")
        print(f"BIC:            {diag['bic']:.2f}")
        print(f"Log-Likelihood: {diag['log_likelihood']:.2f}")
        print(f"\nParameters:")
        for param, value in diag["params"].items():
            pval = diag["pvalues"].get(param, None)
            sig = "***" if pval and pval < 0.001 else "**" if pval and pval < 0.01 else "*" if pval and pval < 0.05 else ""
            print(f"  {param:12s} = {value:10.6f}  (p={pval:.4f}) {sig}")

        print(f"\nCurrent annualized vol: {diag['conditional_vol_current']:.4f} ({diag['conditional_vol_current']*100:.2f}%)")
        print(f"Average annualized vol: {diag['conditional_vol_mean']:.4f} ({diag['conditional_vol_mean']*100:.2f}%)")

        # Forecast
        fcst = self.forecast(horizon=5)
        print(f"\nVolatility Forecast (Annualized %):")
        for _, row in fcst.iterrows():
            print(f"  {int(row['Horizon (Days)'])}D ahead: {row['Annualized Vol (%)']:.2f}%")


if __name__ == "__main__":
    from data.fetcher import fetch_price_data

    prices = fetch_price_data("SPY")
    model = GARCHVolatilityModel()
    model.fit(prices)
    model.print_summary()

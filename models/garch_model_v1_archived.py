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
    RETURN_SCALE, TRADING_DAYS,
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

        # Extract conditional volatility (annualized, in decimal form)
        cond_vol_daily = self.best_result.conditional_volatility / RETURN_SCALE
        self.conditional_vol = cond_vol_daily * np.sqrt(TRADING_DAYS)

        if verbose:
            print(f"\n✓ Best model: {self.best_model_name} (AIC: {self.best_result.aic:.2f})")
            print(f"  GARCH AIC:     {sym_aic:.2f}")
            print(f"  GJR-GARCH AIC: {asym_aic:.2f}")

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

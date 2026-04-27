# 📊 Volatility Trading System — GARCH + Transformer

A Python options trading analysis system that uses **GARCH time series modeling** to forecast volatility and a **Transformer neural network** to identify which market conditions most strongly influence volatility. Includes a **long straddle backtester** with a $150 budget constraint.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Features

- **GARCH Volatility Modeling** — GJR-GARCH(1,1,1) with Student-t distribution for asymmetric leverage effects
- **Transformer Feature Importance** — 21-feature attention-based analysis identifies top volatility drivers
- **Long Straddle Backtesting** — Historical simulation with Black-Scholes pricing and $150 budget constraint
- **Trading Signals** — IV vs RV spread analysis with strategy recommendations
- **Interactive Dashboard** — Streamlit app with equity curves, feature rankings, and options chain visualization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run dashboard/app.py

# Or run via CLI
python main.py --ticker AAPL --mode full
```

## Architecture

```
├── config.py                 # Global configuration
├── main.py                   # CLI entry point
├── data/
│   ├── fetcher.py            # yfinance data ingestion
│   └── feature_engineer.py   # 21 features across 8 categories
├── models/
│   ├── garch_model.py        # GARCH/GJR-GARCH volatility model
│   └── transformer_model.py  # PyTorch Transformer + attention extraction
├── backtest/
│   └── engine.py             # Long straddle backtester
├── signals/
│   └── generator.py          # IV vs RV trading signals
└── dashboard/
    └── app.py                # Streamlit dashboard
```

## Strategy: GARCH-Informed Long Straddle

- **Entry**: Buy ATM call + ATM put when GARCH forecasts realized vol > market implied vol
- **Exit**: Hold for N trading days, close at intrinsic value
- **Risk**: Defined risk — max loss = premium paid (no naked options)
- **Edge**: GARCH identifies when the market underestimates future volatility

## Tech Stack

- **GARCH**: `arch` package with GJR-GARCH and Student-t distribution
- **Transformer**: PyTorch with multi-head attention for feature importance
- **Data**: `yfinance` for historical prices and options chains
- **Dashboard**: Streamlit with Plotly interactive charts
- **Pricing**: Black-Scholes for historical option price synthesis

## Disclaimer

This is a research and educational tool. It does not constitute financial advice. Backtested results use synthesized option prices and may not reflect actual market conditions. Past performance does not guarantee future results.

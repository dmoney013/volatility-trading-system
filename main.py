"""
Main entry point — CLI orchestrator for the GARCH + Transformer
options trading analysis pipeline.

Usage:
  python main.py --ticker AAPL --mode full      # Full pipeline
  python main.py --ticker SPY --mode garch       # GARCH only
  python main.py --ticker TSLA --mode transformer # Transformer only
  python main.py --mode dashboard                # Launch Streamlit
"""
import argparse
import sys
import os
import json
import pickle
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_TICKERS, DEVICE
from data.fetcher import fetch_all_data
from data.feature_engineer import (
    build_features, normalize_features, build_target,
)
from models.garch_model import GARCHVolatilityModel
from models.transformer_model import TransformerTrainer
from signals.generator import generate_signals, format_signal_report


def run_garch(ticker: str, data: dict, verbose: bool = True) -> dict:
    """Run GARCH volatility analysis."""
    garch = GARCHVolatilityModel()
    diagnostics = garch.fit(data["prices"], verbose=verbose)
    garch.print_summary()

    forecast = garch.forecast(horizon=5)
    cond_vol = garch.get_conditional_volatility()

    return {
        "model": garch,
        "diagnostics": diagnostics,
        "forecast": forecast,
        "conditional_vol": cond_vol,
    }


def run_transformer(
    ticker: str, data: dict, verbose: bool = True
) -> dict:
    """Run Transformer feature importance analysis."""
    # Build features
    print("\nBuilding feature matrix...")
    features = build_features(
        data["prices"], data["vix"], data["treasury"], data["options"]
    )
    print(f"  Raw features: {features.shape}")

    # Normalize
    normalized = normalize_features(features)
    print(f"  Normalized features: {normalized.shape}")

    # Build target
    targets = build_target(data["prices"])

    # Train
    trainer = TransformerTrainer(n_features=normalized.shape[1])
    results = trainer.train(normalized, targets, verbose=verbose)

    return {
        "trainer": trainer,
        "features": features,
        "normalized": normalized,
        "targets": targets,
        "results": results,
        "feature_importance": results["feature_importance"],
        "training_history": trainer.get_training_history(),
    }


def run_full_pipeline(ticker: str, verbose: bool = True) -> dict:
    """Run the complete GARCH + Transformer + Signals pipeline."""
    print(f"\n{'#'*60}")
    print(f"#  FULL PIPELINE: {ticker}")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Device: {DEVICE}")
    print(f"{'#'*60}")

    # 1. Fetch data
    data = fetch_all_data(ticker)

    # 2. GARCH analysis
    garch_results = run_garch(ticker, data, verbose)

    # 3. Transformer analysis
    transformer_results = run_transformer(ticker, data, verbose)

    # 4. Generate signals
    print("\n\nGenerating trading signals...")
    garch_forecast_vol = garch_results["forecast"]["Annualized Vol"].iloc[0]
    garch_current_vol = garch_results["conditional_vol"].iloc[-1]

    signal = generate_signals(
        ticker=ticker,
        garch_forecast_vol=garch_forecast_vol,
        garch_current_vol=garch_current_vol,
        options=data["options"],
        feature_importance=transformer_results["feature_importance"],
        prices=data["prices"],
    )

    report = format_signal_report(signal)
    print(report)

    # 5. Save results
    results = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "garch": garch_results,
        "transformer": transformer_results,
        "signal": signal,
        "data": data,
    }

    _save_results(ticker, results)

    return results


def _save_results(ticker: str, results: dict):
    """Save pipeline results to disk."""
    results_dir = os.path.join("results", ticker)
    os.makedirs(results_dir, exist_ok=True)

    # Save signal as JSON
    signal_data = {k: v for k, v in results["signal"].items()}
    # Convert numpy types for JSON serialization
    for k, v in signal_data.items():
        if hasattr(v, "item"):
            signal_data[k] = v.item()
        elif isinstance(v, list):
            signal_data[k] = [str(x) for x in v]

    signal_path = os.path.join(results_dir, "signal.json")
    with open(signal_path, "w") as f:
        json.dump(signal_data, f, indent=2, default=str)

    # Save feature importance
    fi = results["transformer"]["feature_importance"]
    fi_path = os.path.join(results_dir, "feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    # Save GARCH forecast
    fcst = results["garch"]["forecast"]
    fcst_path = os.path.join(results_dir, "garch_forecast.csv")
    fcst.to_csv(fcst_path, index=False)

    # Save conditional volatility
    cv = results["garch"]["conditional_vol"]
    cv_path = os.path.join(results_dir, "conditional_vol.csv")
    cv.to_csv(cv_path, header=True)

    # Save training history
    th = results["transformer"]["training_history"]
    th_path = os.path.join(results_dir, "training_history.csv")
    th.to_csv(th_path, index=False)

    print(f"\n✓ Results saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Options Trading: GARCH + Transformer Volatility Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker AAPL --mode full
  python main.py --ticker SPY --mode garch
  python main.py --mode dashboard
        """,
    )
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        default="SPY",
        help="Stock ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["full", "garch", "transformer", "dashboard"],
        default="full",
        help="Run mode: full pipeline, garch only, transformer only, or dashboard",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable data caching (re-download all data)",
    )

    args = parser.parse_args()

    if args.mode == "dashboard":
        print("Launching Streamlit dashboard...")
        os.system(f"streamlit run dashboard/app.py")
        return

    # Fetch data
    data = fetch_all_data(args.ticker, use_cache=not args.no_cache)

    if args.mode == "garch":
        run_garch(args.ticker, data)
    elif args.mode == "transformer":
        run_transformer(args.ticker, data)
    elif args.mode == "full":
        run_full_pipeline(args.ticker)


if __name__ == "__main__":
    main()

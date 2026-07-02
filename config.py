"""
Global configuration for the Options Trading GARCH + Transformer system.
"""
import torch
from datetime import datetime, timedelta

# ─── Device Selection ───────────────────────────────────────────────
# Apple M1 → MPS, NVIDIA → CUDA, else CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ─── Tickers ────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["SPY", "AAPL", "TSLA", "QQQ", "MSFT"]
# Budget-friendly tickers where ATM straddles fit within $150
# Backtested winners marked with ★
AFFORDABLE_TICKERS = ["HOOD", "MARA", "DKNG", "PLUG", "LCID", "PLTR", "SOFI", "F", "NIO", "RIVN", "SNAP"]
VIX_TICKER = "^VIX"
TREASURY_TICKER = "^TNX"  # 10-Year Treasury Yield

# ─── Data Parameters ────────────────────────────────────────────────
DATA_START_DATE = "2022-01-01"
DATA_END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR = "cache"

# ─── GARCH Parameters ───────────────────────────────────────────────
GARCH_P = 1
GARCH_Q = 1
GARCH_O = 1          # Asymmetric term for GJR-GARCH
GARCH_DIST = "t"     # Student-t distribution for fat tails
RETURN_SCALE = 100    # Scale returns × 100 for numerical stability
TRADING_DAYS = 252
GARCH_FIT_WINDOW = 252  # Fit on last 252 trading days (~1 year) only.
                        # Prevents old volatility spikes from inflating forecasts
                        # via high persistence (α+β ≈ 0.95).
BREAKEVEN_SIGMA = 1.0   # Use 1-sigma move as the predicted price range.
                        # 1.0 = "does the average-sized move cover the premium?"

# ─── v3 Model Fixes ─────────────────────────────────────────────────
GARCH_CALIBRATION_SCALE = 1.3   # Scale predicted range by 1.3x to correct
                                # systematic underestimation. Backtests show 1σ
                                # coverage is 49% (should be 68%), meaning GARCH
                                # underestimates moves by ~16%. This widens the
                                # predicted range so only higher-conviction trades pass.
MIN_MARGIN_THRESHOLD = 1.00     # Minimum $ past breakeven to accept a trade.
                                # Filters out marginal signals (e.g., $0.23 margin on SKLZ).
IV_RANK_MAX = 50                # Only enter when IV rank < 50 (vol is cheap).
                                # If IV is already at the 90th percentile, the premium
                                # bakes in the expected move — no edge.
MAX_TOP_PICKS = 2               # Only trade the top N highest-conviction picks.

# ─── v3.1 Model Fixes (post-NCLH loss) ──────────────────────────────
REJECT_DAMPENED = True          # Hard-reject tickers where GARCH persistence hit
                                # 1.0 and was clamped. Dampened = model is guessing.
                                # NCLH (dampened, -16.5%) lost. PYPL (not dampened, +15.7%) won.
REALIZED_VS_PREDICTED_MIN = 0.50  # Reject if 5-day realized vol < 50% of GARCH prediction.
                                  # Catches "ghost signals" where GARCH reads old volatility
                                  # from a past event that already resolved.

# ─── Autonomous Trader Strict Filters ────────────────────────────
MIN_PERSISTENCE = 0.70            # Reject if GARCH persistence < 0.70.
                                  # QUBT (0.142) = too low, vol regime won't hold.
                                  # HOOD (0.788) = good, regime is sticky.
SCAN_INTERVAL_MINUTES = 30        # Run v4 scan every 30 minutes during market hours.
MIN_SCAN_BUDGET = 100             # Don't bother scanning if cash < $100.
AUTONOMOUS_TP_PCT = 5.0           # Take profit % for all autonomous positions.

# ─── Transformer Hyperparameters ─────────────────────────────────────
SEQ_LEN = 30          # 30 trading days lookback window
D_MODEL = 64          # Internal dimension of transformer
N_HEAD = 4            # Number of attention heads
NUM_LAYERS = 2        # Transformer encoder layers
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15         # Early stopping patience
BATCH_SIZE = 32
WALK_FORWARD_TEST_DAYS = 60  # Test window for walk-forward validation

# ─── Signal Parameters ──────────────────────────────────────────────
VOL_SPREAD_THRESHOLD = 0.05   # 5% IV-RV spread to generate signal
SIGNAL_STRENGTH_MAX = 100

# ─── Backtest Parameters ────────────────────────────────────────────
INITIAL_CAPITAL = 150.0       # Starting account size (USD)
HOLDING_PERIOD_DAYS = 5       # Hold straddle for N trading days
ENTRY_VOL_THRESHOLD = 0.03    # GARCH RV > IV by 3% → enter long straddle
MAX_POSITION_PCT = 0.90       # Max % of capital to risk per trade
COMMISSION_PER_CONTRACT = 0.65  # Webull commission per options contract

STRANGLE_OTM_WIDTH = 0.03         # 3% OTM from spot
STRANGLE_HOLDING_PERIOD_DAYS = 5  # Hold strangle for N trading days
STRANGLE_ENTRY_VOL_THRESHOLD = 0.03  # Same GARCH signal threshold as straddle
MAX_STRANGLE_SPREAD_PCT = 0.06    # Max spread between call & put strikes as % of spot
MIN_EXPIRY_TRADING_DAYS = 14      # Minimum 14 trading days (~20 calendar days) to expiry
                                  # Prevents theta-heavy short-dated positions

# ─── Auto-Close Rules (ABSOLUTE — applies to ALL positions) ─────────
# These thresholds are unconditional: close IMMEDIATELY when hit,
# regardless of strategy (straddle or strangle), time of day, or
# any other condition. If market is closed, queue a pending close
# for the next market open.
# Validated via backtest: +12% TP yields 95.3% win rate on strangles,
# 91.7% on straddles across 42 tickers over 4.5 years of data.
TAKE_PROFIT_PCT = 8.0             # Close at +8% return — ABSOLUTE RULE (v3: lowered from 12%)
                                  # Lower TP catches profits before they retreat.
STOP_LOSS_PCT = -50.0             # Close at -50% loss — ABSOLUTE RULE (v3: re-enabled)
                                  # Prevents total premium wipeout (-100%).
                                  # EV math: 0.45 × (+8%) + 0.55 × (-50%) = -24%
                                  # vs without SL: 0.45 × (+8%) + 0.55 × (-100%) = -51.4%

# ─── Feature Names (for interpretability) ────────────────────────────
FEATURE_NAMES = [
    # Price / Returns
    "Log Return",
    "Abs Return",
    "Return Sign",
    # Realized Volatility
    "RV 5-Day",
    "RV 10-Day",
    "RV 21-Day",
    # Volume
    "Volume Ratio",
    "Log Volume",
    # Technical Indicators
    "RSI(14)",
    "MACD Signal",
    "Bollinger Width",
    "ATR(14)",
    # Market Regime
    "VIX Level",
    "VIX Change %",
    # Options-Derived
    "Put/Call Ratio",
    "Open Interest Change",
    "Avg Implied Vol",
    # Macro
    "10Y Yield",
    "Yield Change",
    # Momentum
    "Momentum 5D",
    "Momentum 21D",
]

NUM_FEATURES = len(FEATURE_NAMES)

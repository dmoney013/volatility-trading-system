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
# (stock price ~$5-20 → straddle cost ~$50-150 per contract)
AFFORDABLE_TICKERS = ["F", "SOFI", "PLTR", "NIO", "RIVN", "HOOD", "SNAP"]
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

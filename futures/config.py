"""
Futures Configuration — universe mapping, contract specs, and thresholds.

Maps Index ETF symbols (used for LSTM inference) to their corresponding
futures contracts (used for price quotes and trade execution).
"""

# ═══════════════════════════════════════════════════════════════════
#  Futures Universe — ETF → Futures mapping
# ═══════════════════════════════════════════════════════════════════

FUTURES_UNIVERSE = {
    # ETF Symbol → (Futures Symbol, Name, Point Value, Tick Size, Margin Estimate)
    'SPY': {
        'future':       'ES=F',
        'micro':        'MES=F',
        'name':         'E-mini S&P 500',
        'point_value':  50.0,       # $50 per point
        'micro_pv':     5.0,        # $5 per point for micro
        'tick_size':    0.25,
        'margin':       15_000,     # Approx initial margin (ES)
        'micro_margin': 1_500,      # Micro ES
    },
    'QQQ': {
        'future':       'NQ=F',
        'micro':        'MNQ=F',
        'name':         'E-mini Nasdaq 100',
        'point_value':  20.0,
        'micro_pv':     2.0,
        'tick_size':    0.25,
        'margin':       18_000,
        'micro_margin': 1_800,
    },
    'IWM': {
        'future':       'RTY=F',
        'micro':        'M2K=F',
        'name':         'E-mini Russell 2000',
        'point_value':  50.0,
        'micro_pv':     5.0,
        'tick_size':    0.10,
        'margin':       8_000,
        'micro_margin': 800,
    },
    'DIA': {
        'future':       'YM=F',
        'micro':        'MYM=F',
        'name':         'E-mini Dow Jones',
        'point_value':  5.0,
        'micro_pv':     0.50,
        'tick_size':    1.0,
        'margin':       10_000,
        'micro_margin': 1_000,
    },
}

# Crypto — separate because different exchange and hours
CRYPTO_FUTURES = {
    'BITO': {
        'future':       'BTC=F',
        'micro':        'MBT=F',
        'name':         'Bitcoin Futures (CME)',
        'point_value':  5.0,
        'micro_pv':     0.10,
        'tick_size':    5.0,
        'margin':       50_000,
        'micro_margin': 2_500,
    },
}

# All tradeable futures (equity + crypto)
ALL_FUTURES = {**FUTURES_UNIVERSE, **CRYPTO_FUTURES}


# ═══════════════════════════════════════════════════════════════════
#  Signal Thresholds
# ═══════════════════════════════════════════════════════════════════

# LSTM confidence required to fire a signal (same as equity hybrid)
CONFIDENCE_THRESHOLD = 0.70

# Minimum GARCH spread to confirm vol is expanding
MIN_GARCH_SPREAD = 0.0   # > 0% means GARCH RV > 30d HV

# Prediction horizon in trading days (matches LSTM training)
PREDICTION_HORIZON = 5


# ═══════════════════════════════════════════════════════════════════
#  Trading Hours (informational — no auto-execution)
# ═══════════════════════════════════════════════════════════════════

EQUITY_FUTURES_HOURS = "Sun 6:00 PM – Fri 5:00 PM ET (nearly 24h)"
CRYPTO_FUTURES_HOURS = "Sun 5:00 PM – Fri 4:00 PM CT"

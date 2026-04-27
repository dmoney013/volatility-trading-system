"""
Position Tracker — fetches live positions from Webull and computes
straddle-level P&L with historical tracking for the dashboard chart.
"""
import os
import json
import logging
from datetime import datetime, timezone
from broker.webull_client import get_accounts, get_positions, _call_api
import broker.webull_client as wb

log = logging.getLogger(__name__)

# Persistent history file (survives restarts)
_HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "position_history.json")


def _load_history():
    """Load historical P&L snapshots from disk."""
    if os.path.exists(_HISTORY_PATH):
        try:
            with open(_HISTORY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(history):
    """Persist P&L snapshots to disk."""
    os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
    with open(_HISTORY_PATH, "w") as f:
        json.dump(history, f)


def fetch_live_positions():
    """
    Fetch live option positions from Webull and aggregate into straddle view.

    Returns dict:
        {
            "positions": [...],           # raw position list
            "straddles": {                 # grouped by symbol+strike+expiry
                "AMC_2.00_2026-05-15": {
                    "symbol": "AMC",
                    "strike": 2.0,
                    "expiry": "2026-05-15",
                    "call": {...} or None,
                    "put":  {...} or None,
                    "total_cost": float,
                    "total_value": float,
                    "total_pnl": float,
                    "pnl_pct": float,
                    "day_pnl": float,
                }
            },
            "total_cost": float,
            "total_value": float,
            "total_pnl": float,
            "total_pnl_pct": float,
            "timestamp": str,
        }
    """
    # Ensure token is loaded
    if not wb.ACCESS_TOKEN:
        from dotenv import load_dotenv
        load_dotenv(wb._ENV_PATH)
        wb.ACCESS_TOKEN = os.getenv("WEBULL_ACCESS_TOKEN", "")

    accounts = get_accounts()
    if not accounts:
        return None

    # Use the margin account for options
    account_id = None
    for a in accounts:
        if a.get("account_type") == "MARGIN" and a.get("account_class") == "INDIVIDUAL_MARGIN":
            account_id = a["account_id"]
            break
    if not account_id:
        account_id = accounts[0]["account_id"]

    positions = get_positions(account_id)
    if not positions:
        return {"positions": [], "straddles": {}, "total_cost": 0, "total_value": 0,
                "total_pnl": 0, "total_pnl_pct": 0, "timestamp": datetime.now(timezone.utc).isoformat()}

    # Filter to options only
    option_positions = [p for p in positions if p.get("instrument_type") == "OPTION"]

    # Group into straddles by symbol + strike + expiry
    straddles = {}
    for pos in option_positions:
        legs = pos.get("legs", [])
        if not legs:
            continue
        leg = legs[0]
        symbol = leg.get("symbol", pos.get("symbol", ""))
        strike = leg.get("option_exercise_price", "0")
        expiry = leg.get("option_expire_date", "")
        opt_type = leg.get("option_type", "")

        key = f"{symbol}_{strike}_{expiry}"
        if key not in straddles:
            straddles[key] = {
                "symbol": symbol,
                "strike": float(strike),
                "expiry": expiry,
                "call": None,
                "put": None,
                "total_cost": 0,
                "total_value": 0,
                "total_pnl": 0,
                "day_pnl": 0,
            }

        leg_data = {
            "quantity": int(pos.get("quantity", 0)),
            "cost_price": float(pos.get("cost_price", 0)),
            "last_price": float(pos.get("last_price", 0)),
            "cost": float(pos.get("cost", 0)),
            "market_value": float(pos.get("market_value", 0)),
            "unrealized_pnl": float(pos.get("unrealized_profit_loss", 0)),
            "pnl_rate": float(pos.get("unrealized_profit_loss_rate", 0)),
            "day_pnl": float(pos.get("day_profit_loss", 0)),
            "position_id": pos.get("position_id", ""),
        }

        if opt_type == "CALL":
            straddles[key]["call"] = leg_data
        elif opt_type == "PUT":
            straddles[key]["put"] = leg_data

        straddles[key]["total_cost"] += leg_data["cost"]
        straddles[key]["total_value"] += leg_data["market_value"]
        straddles[key]["total_pnl"] += leg_data["unrealized_pnl"]
        straddles[key]["day_pnl"] += leg_data["day_pnl"]

    # Compute P&L percentages
    for key, s in straddles.items():
        s["pnl_pct"] = (s["total_pnl"] / s["total_cost"] * 100) if s["total_cost"] > 0 else 0

    total_cost = sum(s["total_cost"] for s in straddles.values())
    total_value = sum(s["total_value"] for s in straddles.values())
    total_pnl = sum(s["total_pnl"] for s in straddles.values())
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    ts = datetime.now(timezone.utc).isoformat()

    # Save snapshot to history
    history = _load_history()
    history.append({
        "timestamp": ts,
        "total_cost": total_cost,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "straddles": {k: {"pnl": v["total_pnl"], "value": v["total_value"],
                          "pnl_pct": v["pnl_pct"]} for k, v in straddles.items()},
    })
    # Keep last 500 snapshots
    if len(history) > 500:
        history = history[-500:]
    _save_history(history)

    return {
        "positions": option_positions,
        "straddles": straddles,
        "total_cost": total_cost,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "timestamp": ts,
    }


def get_pnl_history():
    """Return the historical P&L snapshots for charting."""
    return _load_history()

"""
Webull OpenAPI Client — handles authentication, account management,
and option order placement via the official HTTP API.

Uses HMAC-SHA1 signature auth (no SDK dependency).
Credentials loaded from .env file.
"""
import hashlib
import hmac
import base64
import json
import uuid
import urllib.parse
import requests
import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(_ENV_PATH)

log = logging.getLogger(__name__)

APP_KEY = os.getenv("WEBULL_APP_KEY", "")
APP_SECRET = os.getenv("WEBULL_APP_SECRET", "")
ACCESS_TOKEN = os.getenv("WEBULL_ACCESS_TOKEN", "")
HOST = "api.webull.com"
BASE_URL = f"https://{HOST}"


# ═══════════════════════════════════════════════════════════════════
# Signature Generation (per official Webull docs)
# ═══════════════════════════════════════════════════════════════════

def _generate_signature(path, query_params, body_string, timestamp, nonce):
    signing_headers = {
        "x-app-key": APP_KEY,
        "x-timestamp": timestamp,
        "x-signature-algorithm": "HMAC-SHA1",
        "x-signature-version": "1.0",
        "x-signature-nonce": nonce,
        "host": HOST,
    }
    all_params = {}
    all_params.update(query_params)
    all_params.update(signing_headers)
    str1 = "&".join(f"{k}={all_params[k]}" for k in sorted(all_params.keys()))

    if body_string:
        str2 = hashlib.md5(body_string.encode("utf-8")).hexdigest().upper()
        str3 = f"{path}&{str1}&{str2}"
    else:
        str3 = f"{path}&{str1}"

    encoded_string = urllib.parse.quote(str3, safe="")
    signing_key = f"{APP_SECRET}&"
    signature = base64.b64encode(
        hmac.new(signing_key.encode("utf-8"),
                 encoded_string.encode("utf-8"),
                 hashlib.sha1).digest()
    ).decode("utf-8")
    return signature


def _call_api(method, path, query_params=None, body=None):
    """Sign and send an API request to Webull."""
    global ACCESS_TOKEN
    query_params = query_params or {}
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nonce = uuid.uuid4().hex
    body_string = json.dumps(body, separators=(",", ":")) if body else None

    signature = _generate_signature(path, query_params, body_string, timestamp, nonce)

    headers = {
        "x-app-key": APP_KEY,
        "x-timestamp": timestamp,
        "x-signature": signature,
        "x-signature-algorithm": "HMAC-SHA1",
        "x-signature-version": "1.0",
        "x-signature-nonce": nonce,
        "x-version": "v2",
    }
    if ACCESS_TOKEN:
        headers["x-access-token"] = ACCESS_TOKEN

    url = f"{BASE_URL}{path}"
    if method.upper() == "GET":
        resp = requests.get(url, headers=headers, params=query_params, timeout=15)
    else:
        headers["Content-Type"] = "application/json"
        resp = requests.post(url, headers=headers, data=body_string, timeout=15)

    return resp


# ═══════════════════════════════════════════════════════════════════
# Token Management (2FA)
# ═══════════════════════════════════════════════════════════════════

def create_token():
    """Request a new 2FA token. You must then verify in the Webull app."""
    resp = _call_api("POST", "/openapi/account/token/create")
    if resp.status_code == 200:
        data = resp.json()
        token = data.get("token", "")
        log.info(f"Token created (PENDING). Verify in Webull app.")
        print("\n📱 CHECK YOUR WEBULL APP:")
        print("   1. Go to Menu → Messages → OpenAPI Notifications")
        print("   2. Tap the latest verification message")
        print("   3. Enter the SMS code and tap Confirm")
        print("   4. You have 5 minutes before the token expires")
        return token
    log.error(f"create_token failed: {resp.status_code} — {resp.text}")
    return None


def check_token():
    """Check if the current access token is valid."""
    resp = _call_api("GET", "/openapi/account/token/check")
    if resp.status_code == 200:
        return resp.json()
    return None


def activate_token():
    """Create a token, wait for verification, and save it."""
    global ACCESS_TOKEN
    import time

    token = create_token()
    if not token:
        print("❌ Failed to create token.")
        return False

    ACCESS_TOKEN = token

    print("\n⏳ Waiting for you to verify in the Webull app...")
    for i in range(60):  # Wait up to 5 minutes
        time.sleep(5)
        status = check_token()
        if status and status.get("status") == "NORMAL":
            # Save token to .env for reuse
            _save_token(token)
            print(f"\n✅ Token verified and saved!")
            return True
        print(f"   Still waiting... ({(i+1)*5}s)", end="\r")

    print("\n❌ Token expired. Try again.")
    return False


def _save_token(token):
    """Persist token to .env file."""
    lines = []
    found = False
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, "r") as f:
            lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith("WEBULL_ACCESS_TOKEN="):
            new_lines.append(f"WEBULL_ACCESS_TOKEN={token}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"WEBULL_ACCESS_TOKEN={token}\n")
    with open(_ENV_PATH, "w") as f:
        f.writelines(new_lines)
    log.info(f"Token saved to {_ENV_PATH}")


# ═══════════════════════════════════════════════════════════════════
# Account Management
# ═══════════════════════════════════════════════════════════════════

def get_accounts():
    """Retrieve all linked accounts."""
    resp = _call_api("GET", "/openapi/account/list")
    if resp.status_code == 200:
        return resp.json()
    log.error(f"get_accounts failed: {resp.status_code} — {resp.text}")
    return None


def get_account_balance(account_id):
    """Get account balance and buying power."""
    resp = _call_api("GET", f"/openapi/account/balance",
                     query_params={"account_id": account_id})
    if resp.status_code == 200:
        return resp.json()
    log.error(f"get_balance failed: {resp.status_code} — {resp.text}")
    return None


def get_positions(account_id):
    """Get current positions."""
    resp = _call_api("GET", f"/openapi/account/positions",
                     query_params={"account_id": account_id})
    if resp.status_code == 200:
        return resp.json()
    log.error(f"get_positions failed: {resp.status_code} — {resp.text}")
    return None


# ═══════════════════════════════════════════════════════════════════
# Order Placement — Options
# ═══════════════════════════════════════════════════════════════════

def place_option_order(account_id, symbol, strike, expiry, option_type,
                       side, quantity, limit_price):
    """
    Place a single-leg option order.

    Args:
        account_id: Webull account ID
        symbol: Underlying ticker (e.g. "AMC")
        strike: Strike price as string (e.g. "2.00")
        expiry: Expiration date "YYYY-MM-DD"
        option_type: "CALL" or "PUT"
        side: "BUY" or "SELL"
        quantity: Number of contracts
        limit_price: Limit price per share as string
    """
    client_order_id = uuid.uuid4().hex
    body = {
        "account_id": account_id,
        "new_orders": [{
            "client_order_id": client_order_id,
            "combo_type": "NORMAL",
            "order_type": "LIMIT",
            "limit_price": str(limit_price),
            "quantity": str(quantity),
            "option_strategy": "SINGLE",
            "side": side,
            "time_in_force": "GTC" if side == "BUY" else "DAY",
            "entrust_type": "QTY",
            "instrument_type": "OPTION",
            "market": "US",
            "symbol": symbol,
            "legs": [{
                "side": side,
                "quantity": str(quantity),
                "symbol": symbol,
                "strike_price": str(strike),
                "option_expire_date": expiry,
                "instrument_type": "OPTION",
                "option_type": option_type,
                "market": "US",
            }]
        }]
    }

    log.info(f"Placing {side} {quantity}x {symbol} {strike} {option_type} "
             f"exp {expiry} @ ${limit_price}")

    resp = _call_api("POST", "/openapi/account/orders/option/place", body=body)

    if resp.status_code == 200:
        result = resp.json()
        log.info(f"Order placed: {result}")
        return {"success": True, "order_id": client_order_id, "response": result}
    else:
        log.error(f"Order failed: {resp.status_code} — {resp.text}")
        return {"success": False, "error": resp.text, "status": resp.status_code}


def place_straddle(account_id, symbol, strike, expiry, quantity, call_price, put_price):
    """
    Place a long straddle = buy call + buy put at same strike.
    Executed as two separate SINGLE leg orders.
    """
    call_result = place_option_order(
        account_id, symbol, strike, expiry, "CALL", "BUY", quantity, call_price)
    put_result = place_option_order(
        account_id, symbol, strike, expiry, "PUT", "BUY", quantity, put_price)

    return {
        "call_order": call_result,
        "put_order": put_result,
        "success": call_result["success"] and put_result["success"],
    }


def get_open_orders(account_id):
    """Get all open orders."""
    resp = _call_api("GET", "/openapi/account/orders/list",
                     query_params={"account_id": account_id})
    if resp.status_code == 200:
        return resp.json()
    log.error(f"get_orders failed: {resp.status_code} — {resp.text}")
    return None


def cancel_order(account_id, client_order_id):
    """Cancel an open order."""
    body = {"account_id": account_id, "client_order_id": client_order_id}
    resp = _call_api("POST", "/openapi/account/orders/cancel", body=body)
    if resp.status_code == 200:
        return resp.json()
    log.error(f"cancel failed: {resp.status_code} — {resp.text}")
    return None


# ═══════════════════════════════════════════════════════════════════
# Autonomous Trading Engine
# ═══════════════════════════════════════════════════════════════════

def execute_top_trade(budget=150.0, dry_run=True):
    """
    Full autonomous pipeline:
    1. Run GARCH scanner to find best opportunity
    2. Fetch account info
    3. Place the straddle order

    Args:
        budget: Maximum spend
        dry_run: If True, just log what would happen (no real orders)
    """
    from signals.scanner import scan_for_opportunities

    print(f"\n{'='*60}")
    print(f"AUTONOMOUS TRADING ENGINE")
    print(f"{'='*60}")
    print(f"Budget: ${budget:.2f} | Mode: {'DRY RUN' if dry_run else '🔴 LIVE'}")

    # Step 1: Scan
    print("\n📡 Scanning for opportunities...")
    recs = scan_for_opportunities(budget=budget, top_n=1)
    if not recs:
        print("❌ No opportunities found.")
        return {"success": False, "reason": "no_signal"}

    pick = recs[0]
    print(f"\n🎯 Best signal: {pick['ticker']} ${pick['strike']} straddle")
    print(f"   GARCH spread: +{pick['spread']*100:.1f}%")
    print(f"   Cost: ${pick['total_cost']:.2f} for {pick['contracts']}x contracts")
    print(f"   Call: ${pick['call_price']:.2f} | Put: ${pick['put_price']:.2f}")
    print(f"   Expiry: {pick['expiry']}")

    if dry_run:
        print(f"\n🧪 DRY RUN — no order placed.")
        print(f"   Would buy {pick['contracts']}x {pick['ticker']} "
              f"${pick['strike']} straddle for ${pick['total_cost']:.2f}")
        return {"success": True, "mode": "dry_run", "pick": pick}

    # Step 2: Get account
    print("\n🔑 Fetching account...")
    accounts = get_accounts()
    if not accounts:
        print("❌ Could not retrieve accounts.")
        return {"success": False, "reason": "auth_failed"}

    account_id = accounts[0]["account_id"]
    print(f"   Account: {account_id}")

    # Step 3: Place straddle
    print(f"\n💰 Placing straddle...")
    result = place_straddle(
        account_id=account_id,
        symbol=pick["ticker"],
        strike=f"{pick['strike']:.2f}",
        expiry=pick["expiry"],
        quantity=pick["contracts"],
        call_price=f"{pick['call_price']:.2f}",
        put_price=f"{pick['put_price']:.2f}",
    )

    if result["success"]:
        print(f"✅ Straddle placed successfully!")
    else:
        print(f"❌ Order failed. Check logs.")

    return {"success": result["success"], "pick": pick, "result": result}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test connectivity
    print("Testing Webull API connection...")
    accounts = get_accounts()
    if accounts:
        print(f"✅ Connected! Found {len(accounts)} account(s):")
        for a in accounts:
            print(f"   {a.get('account_id')} — {a.get('account_type', 'N/A')}")
    else:
        print("❌ Connection failed. Check your app key/secret.")

    # Dry run
    execute_top_trade(budget=150.0, dry_run=True)

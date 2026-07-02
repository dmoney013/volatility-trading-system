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
    resp = _call_api("POST", "/openapi/auth/token/create")
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
    resp = _call_api("POST", "/openapi/auth/token/check")
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
    resp = _call_api("GET", "/openapi/assets/positions",
                     query_params={"account_id": account_id})
    if resp.status_code == 200:
        return resp.json()
    log.error(f"get_positions failed: {resp.status_code} — {resp.text}")
    return None

# ═══════════════════════════════════════════════════════════════════
# Market Data — Real-time Option Quotes
# ═══════════════════════════════════════════════════════════════════

def _build_option_symbol(ticker, expiry, option_type, strike):
    """
    Build OCC option symbol format: TICKER + YYMMDD + C/P + strike*1000 (8 digits).
    Example: CHPT 2026-06-18 CALL $9 → CHPT  260618C00009000
    """
    # Pad ticker to 6 chars (left-aligned, space-padded)
    padded = ticker.ljust(6)
    # Date: YYMMDD
    dt = datetime.strptime(expiry, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    # Type: C or P
    cp = "C" if option_type.upper() == "CALL" else "P"
    # Strike: multiply by 1000, zero-pad to 8 digits
    strike_int = int(float(strike) * 1000)
    strike_str = f"{strike_int:08d}"
    return f"{padded}{date_str}{cp}{strike_str}"


def get_option_quote(ticker, expiry, option_type, strike):
    """
    Fetch real-time option quote from Webull's market data API.

    Returns dict with bid, ask, last, volume, open_interest or None on failure.
    """
    symbol = _build_option_symbol(ticker, expiry, option_type, strike)
    log.info(f"   [webull] Fetching quote for {symbol.strip()}")

    try:
        resp = _call_api("GET", "/openapi/market/options/snapshot",
                        query_params={"symbols": symbol})
        if resp.status_code == 200:
            data = resp.json()
            # Response may be a list or dict depending on API version
            if isinstance(data, list) and len(data) > 0:
                q = data[0]
            elif isinstance(data, dict):
                # Try common response structures
                q = data.get("data", data)
                if isinstance(q, list) and len(q) > 0:
                    q = q[0]
            else:
                log.warning(f"   [webull] Empty response for {symbol.strip()}")
                return None

            return {
                'bid': float(q.get('bid', q.get('bidPrice', 0)) or 0),
                'ask': float(q.get('ask', q.get('askPrice', 0)) or 0),
                'last': float(q.get('last', q.get('lastPrice', q.get('close', 0))) or 0),
                'volume': int(q.get('volume', 0) or 0),
                'open_interest': int(q.get('openInterest', q.get('open_interest', 0)) or 0),
            }
        else:
            log.warning(f"   [webull] Quote failed ({resp.status_code}): {resp.text[:200]}")
            return None
    except Exception as e:
        log.warning(f"   [webull] Quote error for {symbol.strip()}: {e}")
        return None


def get_strangle_quotes(ticker, call_strike, put_strike, expiry):
    """
    Fetch real-time bid/ask for both legs of a strangle from Webull.

    Returns dict compatible with fetch_fresh_strangle_quotes format,
    or None if either leg fails.
    """
    call_q = get_option_quote(ticker, expiry, "CALL", call_strike)
    put_q = get_option_quote(ticker, expiry, "PUT", put_strike)

    if not call_q or not put_q:
        return None

    # Use ask price for buying (most conservative/realistic fill price)
    # Fall back to last price if ask is zero
    c_price = call_q['ask'] if call_q['ask'] > 0 else call_q['last']
    p_price = put_q['ask'] if put_q['ask'] > 0 else put_q['last']

    return {
        'call_strike': float(call_strike),
        'put_strike': float(put_strike),
        'call_price': round(c_price, 2),
        'put_price': round(p_price, 2),
        'call_bid': round(call_q['bid'], 2),
        'put_bid': round(put_q['bid'], 2),
        'call_ask': round(call_q['ask'], 2),
        'put_ask': round(put_q['ask'], 2),
        'call_volume': call_q['volume'],
        'put_volume': put_q['volume'],
        'source': 'webull',
    }


def get_straddle_quotes(ticker, strike, expiry):
    """
    Fetch real-time bid/ask for both legs of a straddle from Webull.
    """
    return get_strangle_quotes(ticker, strike, strike, expiry)


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

    resp = _call_api("POST", "/openapi/trade/order/place", body=body)

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
# Multi-Leg Combo Orders (atomic fill — both legs or neither)
# ═══════════════════════════════════════════════════════════════════

def place_combo_strangle(account_id, symbol, call_strike, put_strike,
                         expiry, side, quantity, call_limit, put_limit):
    """
    Place a long strangle as a single atomic combo order.

    The exchange fills both legs simultaneously or rejects the entire order.
    This prevents partial fills (one leg filled, other not) that create
    naked directional exposure.

    Validated structure (tested against /order/preview):
        - Single new_orders entry with combo_type=NORMAL
        - option_strategy=STRANGLE
        - Two legs in the legs array
        - limit_price = net debit (call + put)

    Args:
        account_id: Webull account ID
        symbol: Underlying ticker (e.g. "XPEV")
        call_strike: Call strike price (e.g. "17.00")
        put_strike: Put strike price (e.g. "15.00")
        expiry: Expiration "YYYY-MM-DD"
        side: "BUY" or "SELL"
        quantity: Number of contracts per leg
        call_limit: Limit price for the call leg
        put_limit: Limit price for the put leg

    Returns:
        dict with success, combo_order_id, response
    """
    client_order_id = uuid.uuid4().hex

    # Net debit = call + put
    net_price = float(call_limit) + float(put_limit)

    body = {
        "account_id": account_id,
        "new_orders": [{
            "client_order_id": client_order_id,
            "combo_type": "NORMAL",
            "order_type": "LIMIT",
            "limit_price": f"{net_price:.2f}",
            "quantity": str(quantity),
            "option_strategy": "STRANGLE",
            "side": side,
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "instrument_type": "OPTION",
            "market": "US",
            "symbol": symbol,
            "legs": [
                {
                    "side": side,
                    "quantity": str(quantity),
                    "symbol": symbol,
                    "strike_price": str(call_strike),
                    "option_expire_date": expiry,
                    "instrument_type": "OPTION",
                    "option_type": "CALL",
                    "market": "US",
                },
                {
                    "side": side,
                    "quantity": str(quantity),
                    "symbol": symbol,
                    "strike_price": str(put_strike),
                    "option_expire_date": expiry,
                    "instrument_type": "OPTION",
                    "option_type": "PUT",
                    "market": "US",
                },
            ],
        }],
    }

    log.info(f"Placing COMBO {side} STRANGLE: {quantity}x {symbol} "
             f"${call_strike}C/${put_strike}P exp {expiry} "
             f"(net debit ${net_price:.2f}/share)")

    resp = _call_api("POST", "/openapi/trade/order/place", body=body)

    if resp.status_code == 200:
        result = resp.json()
        log.info(f"Combo order placed: {result}")
        return {
            "success": True,
            "combo_order_id": client_order_id,
            "order_id": client_order_id,
            "response": result,
        }
    else:
        log.error(f"Combo order failed: {resp.status_code} — {resp.text}")
        return {"success": False, "error": resp.text, "status": resp.status_code}


def place_combo_straddle(account_id, symbol, strike, expiry,
                         side, quantity, call_limit, put_limit):
    """
    Place a long straddle as a single atomic combo order.
    Same as strangle but with identical strikes, option_strategy=STRADDLE.
    """
    client_order_id = uuid.uuid4().hex
    net_price = float(call_limit) + float(put_limit)

    body = {
        "account_id": account_id,
        "new_orders": [{
            "client_order_id": client_order_id,
            "combo_type": "NORMAL",
            "order_type": "LIMIT",
            "limit_price": f"{net_price:.2f}",
            "quantity": str(quantity),
            "option_strategy": "STRADDLE",
            "side": side,
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "instrument_type": "OPTION",
            "market": "US",
            "symbol": symbol,
            "legs": [
                {
                    "side": side,
                    "quantity": str(quantity),
                    "symbol": symbol,
                    "strike_price": str(strike),
                    "option_expire_date": expiry,
                    "instrument_type": "OPTION",
                    "option_type": "CALL",
                    "market": "US",
                },
                {
                    "side": side,
                    "quantity": str(quantity),
                    "symbol": symbol,
                    "strike_price": str(strike),
                    "option_expire_date": expiry,
                    "instrument_type": "OPTION",
                    "option_type": "PUT",
                    "market": "US",
                },
            ],
        }],
    }

    log.info(f"Placing COMBO {side} STRADDLE: {quantity}x {symbol} "
             f"${strike} exp {expiry} (net debit ${net_price:.2f}/share)")

    resp = _call_api("POST", "/openapi/trade/order/place", body=body)

    if resp.status_code == 200:
        result = resp.json()
        log.info(f"Combo order placed: {result}")
        return {
            "success": True,
            "combo_order_id": client_order_id,
            "order_id": client_order_id,
            "response": result,
        }
    else:
        log.error(f"Combo order failed: {resp.status_code} — {resp.text}")
        return {"success": False, "error": resp.text, "status": resp.status_code}


def close_straddle(account_id, straddle_data, check_market_hours=True):
    """
    Close a straddle by selling both call and put legs.

    Args:
        account_id: Webull account ID
        straddle_data: Dict from position_tracker with 'call', 'put', 'symbol',
                       'strike', 'expiry' keys. Each leg should have 'last_price'
                       updated with fresh quotes before calling this.
        check_market_hours: If True, refuse to sell outside market hours
                            (prevents Webull OPTION_ONLY_SUPPORT_MARKET_IN_CORE_TIME error)
    """
    # Market hours guard — Webull rejects option sells outside 9:30-4:00 ET
    if check_market_hours:
        try:
            import pytz
            from datetime import datetime as dt
            et = pytz.timezone("US/Eastern")
            now = dt.now(et)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if now.weekday() >= 5 or now < market_open or now > market_close:
                log.warning(
                    f"close_straddle blocked: market is closed "
                    f"({now.strftime('%A %H:%M ET')}). "
                    f"Save as pending close and retry at next open."
                )
                return {
                    "call_order": None, "put_order": None, "success": False,
                    "error": "market_closed",
                }
        except ImportError:
            pass  # If pytz unavailable, proceed anyway

    results = {"call_order": None, "put_order": None, "success": True}

    for leg_name in ("call", "put"):
        leg = straddle_data.get(leg_name)
        if not leg or leg.get("quantity", 0) <= 0:
            continue

        opt_type = "CALL" if leg_name == "call" else "PUT"
        # Use last_price as limit — callers should update this with fresh
        # bid prices before calling close_straddle
        sell_price = f"{leg['last_price']:.2f}"

        result = place_option_order(
            account_id=account_id,
            symbol=straddle_data["symbol"],
            strike=f"{straddle_data['strike']:.2f}",
            expiry=straddle_data["expiry"],
            option_type=opt_type,
            side="SELL",
            quantity=leg["quantity"],
            limit_price=sell_price,
        )
        results[f"{leg_name}_order"] = result
        if not result.get("success"):
            results["success"] = False
            log.error(f"Failed to close {leg_name} leg: {result}")

    return results


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

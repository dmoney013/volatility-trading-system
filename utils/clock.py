"""
EST/EDT Clock Utility — always returns the current Eastern Time.
Import and call now() anywhere to get accurate ET datetime.

Usage:
    from utils.clock import now, today, day_of_week, market_status
"""
import pytz
from datetime import datetime

ET = pytz.timezone("US/Eastern")


def now():
    """Current datetime in Eastern Time."""
    return datetime.now(ET)


def today():
    """Today's date string in ET (YYYY-MM-DD)."""
    return now().strftime("%Y-%m-%d")


def day_of_week():
    """Current day name in ET (e.g., 'Wednesday')."""
    return now().strftime("%A")


def timestamp():
    """Human-readable timestamp: 'Wednesday, June 24, 2026 at 09:29 PM ET'."""
    return now().strftime("%A, %B %d, %Y at %I:%M %p ET")


def market_status():
    """Returns 'OPEN', 'PRE', 'AFTER', or 'CLOSED'."""
    n = now()
    # Weekends
    if n.weekday() >= 5:
        return "CLOSED"
    h, m = n.hour, n.minute
    t = h * 60 + m
    if t < 4 * 60:       # before 4:00 AM
        return "CLOSED"
    elif t < 9 * 60 + 30: # 4:00 AM - 9:29 AM
        return "PRE"
    elif t < 16 * 60:     # 9:30 AM - 3:59 PM
        return "OPEN"
    elif t < 20 * 60:     # 4:00 PM - 7:59 PM
        return "AFTER"
    else:
        return "CLOSED"


def tomorrow():
    """Tomorrow's date string in ET (YYYY-MM-DD)."""
    from datetime import timedelta
    return (now() + timedelta(days=1)).strftime("%Y-%m-%d")


def tomorrow_day():
    """Tomorrow's day name in ET."""
    from datetime import timedelta
    return (now() + timedelta(days=1)).strftime("%A")


if __name__ == "__main__":
    print(f"  Now:      {timestamp()}")
    print(f"  Today:    {today()} ({day_of_week()})")
    print(f"  Tomorrow: {tomorrow()} ({tomorrow_day()})")
    print(f"  Market:   {market_status()}")

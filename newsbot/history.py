"""Historical market reactions for known event types.

The numbers come from scripts/build_history.py (real price data around real past events), stored in
data/reactions.json. Nothing here predicts anything: it reports what happened after past events of the
same type, and only calls a direction when the record is statistically distinguishable from a coin flip.
"""
import json
import math
import os
from statistics import mean, median

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "reactions.json")

MIN_N_FOR_CALL = 8   # fewer directional observations than this: never call a direction
ALPHA = 0.01         # strict on purpose: ~35 comparisons per build, so 0.05 would hand us ~2 flukes
SMALL_SAMPLE = 10    # flag anything under this in the output


def binom_p(k: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-9)))


def summarize(values) -> dict | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return {
        "n": len(vals),
        "up": sum(v > 0 for v in vals),
        "down": sum(v < 0 for v in vals),
        "median": median(vals),
        "mean": mean(vals),
    }


def verdict(stat: dict) -> str:
    """'up' / 'down' only when the record is consistent enough to say so; otherwise 'none'."""
    n_dir = stat["up"] + stat["down"]
    if n_dir < MIN_N_FOR_CALL or binom_p(stat["up"], n_dir) >= ALPHA:
        return "none"
    return "up" if stat["up"] > stat["down"] else "down"


def _line(name: str, stat: dict, unit: str, noun: str = "days") -> str:
    v = verdict(stat)
    arrow = {"up": "↑", "down": "↓", "none": "≈"}[v]
    tail = "" if v != "none" else " (no consistent direction)"
    move = f"{stat['median']:+.0f}bp" if unit == "bp" else f"{stat['median']:+.2%}"  # yields are stored in bp
    move = move.replace("-0.00%", "+0.00%").replace("-0bp", "+0bp")  # no negative zeros
    return f"{arrow} {name}: up {stat['up']} of {stat['n']} {noun}, median {move}{tail}"


def load(path: str = DATA_PATH) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def describe(event_type: str, ticker: str | None = None, data: dict | None = None) -> str | None:
    """Multi-line historical summary for this event type, or None if we have no data for it."""
    data = data if data is not None else load()
    if not data:
        return None
    ev = data.get("events", {}).get(event_type)
    if not ev:
        return None

    since = data.get("start", "")[:4]
    small = " – small sample" if ev["n"] < SMALL_SAMPLE else ""
    lines = [f"Past {ev['label']} since {since} ({ev['n']}{small}):"]

    t = (data.get("tickers", {}).get((ticker or "").upper(), {}) or {}).get(event_type)
    if t and t["n"] >= 4:
        lines.append(_line(f"{ticker.upper()} itself", t, "pct", "times"))
    for name, stat in ev["instruments"].items():
        unit = "bp" if name.endswith("yield") else "pct"
        noun = "times" if event_type.startswith("earnings") else "days"
        lines.append(_line(name, stat, unit, noun))
    return "\n".join(lines)

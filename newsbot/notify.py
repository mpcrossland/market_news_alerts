"""ntfy publisher. Uses the JSON publish API so emoji / non-latin-1 text in titles survives
(the header-based API chokes on it)."""
import logging

import requests

from .classify import Verdict
from .fetch import Item

log = logging.getLogger("newsbot.notify")

TAGS = {
    "rates": "bank",
    "inflation": "fire",
    "employment": "briefcase",
    "growth_data": "bar_chart",
    "earnings": "moneybag",
    "tech_business": "computer",
    "m_and_a": "handshake",
    "fiscal_trade_reg": "scroll",
    "geopolitics": "globe_with_meridians",
    "commodities": "oil_drum",
    "financial_stress": "rotating_light",
    "trump_post": "mega",
    "other": "newspaper",
}
NO_HISTORY = "No historical data for this type of news."
# ntfy priorities: 1 min, 2 low, 3 default, 4 high, 5 max/urgent
PRIORITY = {5: 5, 4: 4, 3: 3}


def build_message(item: Item, history_text: str | None, context_text: str | None) -> str:
    """What the market was doing (facts), then what happened after past events like this (facts)."""
    parts = [context_text, history_text or NO_HISTORY]
    return "\n\n".join(p for p in parts if p) + f"\n— {item.source.name}"


def _post(topic: str, server: str, token: str | None, **body) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.post(server.rstrip("/") + "/", json={"topic": topic, **body}, headers=headers, timeout=15)
    r.raise_for_status()


def send(item: Item, v: Verdict, message: str, topic: str, server: str = "https://ntfy.sh",
         token: str | None = None) -> None:
    body = {
        "title": (v.headline or item.title)[:250],
        "message": message,
        "priority": PRIORITY.get(v.importance, 3),
        "tags": [TAGS.get(v.category, "newspaper")],
    }
    if item.url:
        body["click"] = item.url
    _post(topic, server, token, **body)


def send_status(title: str, message: str, topic: str, server: str = "https://ntfy.sh",
                token: str | None = None) -> None:
    """A bot-health message (not news). High priority so it isn't missed."""
    _post(topic, server, token, title=title, message=message, priority=4, tags=["warning"])


def build_mover(move, cause, confidence: str) -> tuple[str, str]:
    """(title, message) for a price-triggered alert. The move is fact; the cause is a labelled inference."""
    from .movers import NAMES, NY

    mins = round((move.end - move.start).total_seconds() / 60)
    title = f"{'📈' if move.pct > 0 else '📉'} {NAMES.get(move.symbol, move.symbol)} {move.pct:+.1%} in {mins} min"
    when = f"{move.start.astimezone(NY):%H:%M}–{move.end.astimezone(NY):%H:%M} ET"
    lines = [f"{NAMES.get(move.symbol, move.symbol)} {move.pct:+.2%}, {when}."]
    lines += [f"{NAMES.get(sym, sym)} {pct:+.2%} over the same window." for sym, pct in move.others.items()]
    if cause:
        lines += ["", f"Possible cause (matched by timing, unconfirmed, {confidence} confidence):",
                  f"{cause.title}", f"— {cause.source.name}, {cause.published.astimezone(NY):%H:%M} ET"]
    else:
        lines += ["", "No headline in our sources clearly explains it."]
    return title, "\n".join(lines)


def send_mover(move, cause, confidence: str, topic: str, server: str = "https://ntfy.sh",
               token: str | None = None) -> None:
    title, message = build_mover(move, cause, confidence)
    body = {"title": title, "message": message, "priority": 5,
            "tags": ["chart_with_upwards_trend" if move.pct > 0 else "chart_with_downwards_trend"]}
    if cause and cause.url:
        body["click"] = cause.url
    _post(topic, server, token, **body)

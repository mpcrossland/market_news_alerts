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


def send(item: Item, v: Verdict, message: str, topic: str, server: str = "https://ntfy.sh",
         token: str | None = None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = {
        "topic": topic,
        "title": (v.headline or item.title)[:250],
        "message": message,
        "priority": PRIORITY.get(v.importance, 3),
        "tags": [TAGS.get(v.category, "newspaper")],
    }
    if item.url:
        body["click"] = item.url
    r = requests.post(server.rstrip("/") + "/", json=body, headers=headers, timeout=15)
    r.raise_for_status()


def send_status(title: str, message: str, topic: str, server: str = "https://ntfy.sh",
                token: str | None = None) -> None:
    """A bot-health message (not news). High priority so it isn't missed."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = {"topic": topic, "title": title, "message": message, "priority": 4, "tags": ["warning"]}
    r = requests.post(server.rstrip("/") + "/", json=body, headers=headers, timeout=15)
    r.raise_for_status()

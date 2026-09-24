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
ARROW = {"up": "📈", "down": "📉", "mixed": "↕️", "unclear": ""}
# ntfy priorities: 1 min, 2 low, 3 default, 4 high, 5 max/urgent
PRIORITY = {5: 5, 4: 4, 3: 3}


def send(item: Item, v: Verdict, topic: str, server: str = "https://ntfy.sh", token: str | None = None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = {
        "topic": topic,
        "title": f"{ARROW.get(v.direction, '')} {item.source.name}: {item.title}".strip()[:250],
        "message": v.summary,
        "priority": PRIORITY.get(v.importance, 3),
        "tags": [TAGS.get(v.category, "newspaper")],
    }
    if item.url:
        body["click"] = item.url
    r = requests.post(server.rstrip("/") + "/", json=body, headers=headers, timeout=15)
    r.raise_for_status()

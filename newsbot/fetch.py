"""Fetch feeds and normalise entries into Items."""
import calendar
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser

import feedparser
import requests

from .sources import SOURCES, Source

log = logging.getLogger("newsbot.fetch")

USER_AGENT = "Mozilla/5.0 (compatible; market-news-bot/1.0)"
TIMEOUT = 20
MAX_TEXT = 1500  # chars of body text handed to the classifier


@dataclass
class Item:
    source: Source
    uid: str
    title: str
    text: str
    url: str
    published: datetime | None


class _ParagraphText(HTMLParser):
    """Collect visible text from <p> tags only; skips nav/footer boilerplate."""

    def __init__(self) -> None:
        super().__init__()
        self._in_p = 0
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "p":
            self._in_p += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag == "p":
            self._in_p = max(0, self._in_p - 1)

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self.parts.append(data)


def html_to_text(fragment: str) -> str:
    """Strip tags (all of them) from a feed-embedded HTML fragment."""
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def paragraph_text(page_html: str) -> str:
    p = _ParagraphText()
    p.feed(page_html)
    return re.sub(r"\s+", " ", " ".join(p.parts)).strip()


def _get(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def _published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)
    return None


# Google News returns bare quote pages ("ERG.MU"), monitors and image-only posts; none are news.
JUNK_TITLE = re.compile(r"^\[No Title\]|^[A-Z0-9]{1,6}\.[A-Z]{1,3}\b|Stock Price & Latest News|Misinformation Monitor")


def _clean_title(title: str, source: Source) -> str:
    title = html_to_text(title)
    if source.strip_suffix:
        title = re.sub(r"\s+-\s+[^-]{2,40}$", "", title)  # " - Reuters"
    return title


def fetch_source(source: Source) -> list[Item]:
    try:
        feed = feedparser.parse(_get(source.url))
    except Exception as e:  # one dead feed must not take down the run
        log.warning("fetch failed: %s (%s)", source.name, e)
        return []
    if feed.bozo and not feed.entries:
        log.warning("unparseable feed: %s (%s)", source.name, feed.get("bozo_exception"))
        return []

    items = []
    for e in feed.entries:
        url = e.get("link") or ""
        uid = e.get("id") or e.get("guid") or url
        title = _clean_title(e.get("title", ""), source)
        if not uid or not title or JUNK_TITLE.search(title):
            continue
        body = e.get("summary") or ""
        if e.get("content"):
            body = e["content"][0].get("value", body)
        text = html_to_text(body)
        if source.strip_suffix or text == title:  # Google's body is just the headline again; Fed repeats its title
            text = ""
        items.append(Item(source, uid, title, text[:MAX_TEXT], url, _published(e)))
    return items


def fetch_all(sources: list[Source] = SOURCES) -> list[Item]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(fetch_source, sources)
    items = [i for batch in results for i in batch]
    log.info("fetched %d items from %d sources", len(items), len(sources))
    return items


def enrich(item: Item) -> None:
    """For title-only feeds (Fed, BoC), pull the page's paragraphs so the classifier sees the decision."""
    if not item.source.enrich or item.text or not item.url:
        return
    try:
        item.text = paragraph_text(_get(item.url).decode("utf-8", "replace"))[:MAX_TEXT]
    except Exception as e:
        log.warning("enrich failed: %s (%s)", item.url, e)

"""Dedup state: exact IDs plus fuzzy headline matching, persisted as one JSON file.

Exact IDs stop the same feed entry being reprocessed every run. The fuzzy layer catches
one story arriving from several sources ("Fed holds rates" via Reuters, CNBC, and the Fed).
"""
import json
import os
import re
import tempfile
import time

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or says say said that the to was were will with "
    "after over new his her their this than into amid about up down".split()
)
SEEN_TTL = 7 * 86400
TITLE_TTL = 48 * 3600
MAX_TITLES = 3000
FUZZY_THRESHOLD = 0.6
MIN_TOKENS = 4  # too few tokens and Jaccard is meaningless


def title_tokens(title: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9$%.]+", title.lower())
    return frozenset(w.strip(".") for w in words if w.strip(".") and w not in STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


class State:
    def __init__(self, path: str):
        self.path = path
        self.fresh = not os.path.exists(path)  # first run (or lost cache): seed silently
        self.seen: dict[str, float] = {}
        self.meta: dict[str, float] = {}
        self.titles: list[tuple[float, frozenset[str]]] = []
        # Titles of items queued this run but not yet finished (classifier may still fail).
        # Never persisted: if we saved them, a retried item would match itself and be dropped.
        self._pending: list[frozenset[str]] = []
        if not self.fresh:
            try:
                with open(path) as f:
                    data = json.load(f)
                self.seen = data.get("seen", {})
                self.meta = data.get("meta", {})
                self.titles = [(t, frozenset(toks.split())) for t, toks in data.get("titles", [])]
            except (OSError, ValueError):
                self.fresh = True  # corrupt file: treat like a first run rather than crash-looping

    def is_seen(self, uid: str) -> bool:
        return uid in self.seen

    def is_near_duplicate(self, title: str) -> bool:
        toks = title_tokens(title)
        if len(toks) < MIN_TOKENS:
            return False
        known = [t for _, t in self.titles] + self._pending
        return any(len(t) >= MIN_TOKENS and _jaccard(toks, t) >= FUZZY_THRESHOLD for t in known)

    def reserve_title(self, title: str) -> None:
        """Make later items in this run dedupe against `title` without persisting it."""
        toks = title_tokens(title)
        if len(toks) >= MIN_TOKENS:
            self._pending.append(toks)

    def mark(self, uid: str, title: str = "") -> None:
        now = time.time()
        self.seen[uid] = now
        toks = title_tokens(title)
        if len(toks) >= MIN_TOKENS:
            self.titles.append((now, toks))

    def save(self) -> None:
        now = time.time()
        self.seen = {k: t for k, t in self.seen.items() if now - t < SEEN_TTL}
        self.titles = [(t, toks) for t, toks in self.titles if now - t < TITLE_TTL][-MAX_TITLES:]
        payload = {
            "seen": self.seen,
            "meta": self.meta,
            "titles": [[t, " ".join(sorted(toks))] for t, toks in self.titles],
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)))
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)  # atomic: a killed run can't leave half a file

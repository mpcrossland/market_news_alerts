"""What the market was already doing when the news broke, and what it has done since.

Facts only: prices from 5-minute bars, no forecasting. "Before" is the hour leading up to the story's
timestamp; "since" is the move from then to now, i.e. how much of the reaction is already in the price
by the time the alert reaches you.

Bars are (bar_start_utc, close) tuples so the logic is testable without pandas or a network.
"""
import logging
import re
from datetime import datetime, timedelta, timezone

log = logging.getLogger("newsbot.pricecheck")

BAR = timedelta(minutes=5)
LOOKBACK = timedelta(minutes=60)
MAX_STALE = timedelta(minutes=20)   # a bar older than this at time t means the market was closed then
MIN_SINCE = timedelta(minutes=10)   # younger than this, "since" would just be noise
FLAT = {"pct": 0.0005, "bp": 1.0}   # below this a move is shown as flat (→)

BASE = {"S&P 500 futures": "ES=F", "Nasdaq futures": "NQ=F"}
MACRO_EVENTS = {"fomc_cut", "fomc_hold", "fomc_hike", "cpi_release", "jobs_release"}
TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

Bars = list[tuple[datetime, float]]


def symbols_for(event_type: str, ticker: str) -> dict[str, str]:
    """Display name -> Yahoo symbol for the instruments worth showing on this alert."""
    out = dict(BASE)
    if event_type in MACRO_EVENTS:
        out["10-yr yield"] = "^TNX"
    if ticker and TICKER_RE.match(ticker):
        out[ticker] = ticker
    return out


def _price_at(bars: Bars, t: datetime) -> float | None:
    """Close of the last bar that had finished by time t, unless the data has a gap there."""
    best = None
    for start, close in bars:
        if start + BAR <= t:
            best = (start, close)
        else:
            break
    if best is None or t - (best[0] + BAR) > MAX_STALE:
        return None
    return best[1]


def _move(a: float | None, b: float | None, unit: str) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return (b - a) * 100 if unit == "bp" else b / a - 1  # ^TNX is quoted in %, so *100 = bp


def _fmt(name: str, v: float, unit: str) -> str:
    arrow = "↑" if v > FLAT[unit] else "↓" if v < -FLAT[unit] else "→"
    txt = f"{v:+.0f}bp" if unit == "bp" else f"{v:+.2%}"
    return f"{arrow} {name} {txt.replace('-0.00%', '+0.00%').replace('-0bp', '+0bp')}"


def context(book: dict[str, Bars], instruments: dict[str, str], published: datetime | None,
            now: datetime | None = None) -> str | None:
    """Two short lines ('before' / 'since'), or None if there's no usable data."""
    if published is None:
        return None
    now = now or datetime.now(timezone.utc)
    before, since = [], []
    for name, sym in instruments.items():
        bars = book.get(sym)
        if not bars:
            continue
        unit = "bp" if name.endswith("yield") else "pct"
        p0, p1, p2 = (_price_at(bars, published - LOOKBACK), _price_at(bars, published), _price_at(bars, now))
        if (m := _move(p0, p1, unit)) is not None:
            before.append(_fmt(name, m, unit))
        if now - published >= MIN_SINCE and (m := _move(p1, p2, unit)) is not None:
            since.append(_fmt(name, m, unit))
    lines = []
    if before:
        lines.append("Hour before the news: " + ", ".join(before))
    if since:
        lines.append("Since the news: " + ", ".join(since))
    return "\n".join(lines) or None


def fetch_bars(symbols: set[str]) -> dict[str, Bars]:
    """One Yahoo download for every symbol needed this run. Never raises: alerts just go out without context."""
    if not symbols:
        return {}
    try:
        import yfinance as yf

        syms = sorted(symbols)
        df = yf.download(syms, period="2d", interval="5m", prepost=True, progress=False,
                         group_by="ticker", auto_adjust=True, threads=False)
        out: dict[str, Bars] = {}
        for s in syms:
            try:
                closes = df[s]["Close"].dropna()
            except KeyError:
                continue
            out[s] = [(ts.tz_convert("UTC").to_pydatetime(), float(v)) for ts, v in closes.items()]
        return out
    except Exception as e:
        log.warning("price lookup failed, sending alerts without market context: %s", e)
        return {}

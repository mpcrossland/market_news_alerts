#!/usr/bin/env python3
"""Rebuild newsbot/data/reactions.json from real price history.

Run by hand (needs `pip install yfinance pandas`), e.g. once a month:
    .venv/bin/python scripts/build_history.py

For each event type we find the dates it really happened, then measure what each market did that day
(close vs previous close). Events covered:
  fomc_cut / fomc_hold / fomc_hike  Fed decision days (announced 2pm ET, so day-of close-to-close)
  cpi_release / jobs_release        BLS release days (8:30am ET), all releases (see note below)
  earnings_beat / earnings_miss     ~100 large caps, reaction day, split by EPS above/below estimate

Note: for CPI and jobs we have no free source for *consensus* expectations, and markets react to the
surprise, not the number. So those are reported unconditionally (how markets typically moved on those
days), not as "hot"/"cool" splits.
"""
import io
import json
import os
import re
import sys
import time
import warnings
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from newsbot.history import DATA_PATH, summarize  # noqa: E402

warnings.filterwarnings("ignore")
START = "2015-01-01"
UA = {"User-Agent": "Mozilla/5.0 (compatible; market-news-bot/1.0)"}
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], 1)}
ABBR = {m[:3]: i for m, i in MONTHS.items()}

INSTRUMENTS = {  # display name -> Yahoo symbol. Names ending in "yield" are measured in bp, others in %.
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "10-yr yield": "^TNX",
    "US dollar index": "DX-Y.NYB",
    "Gold": "GC=F",
}

UNIVERSE = """AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V UNH XOM LLY MA JNJ PG HD COST ABBV MRK CVX BAC KO PEP
WMT ADBE CRM NFLX AMD TMO ORCL CSCO ACN MCD ABT DHR WFC TXN INTU QCOM PM IBM GE AMGN CAT VZ DIS NOW ISRG UBER GS
SPGI RTX PFE T MS NKE LOW HON BKNG UNP AXP BLK NEE COP SBUX INTC BA MU LRCX AMAT PLTR C MMM LMT DE GILD MDLZ ADP
CVS CI SCHW ELV TJX PGR MO USB PYPL F GM TGT MRNA UPS FDX BMY ABNB SHOP PANW CRWD SNOW""".split()


def get(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------- event dates
def end_date(month_txt: str, day_txt: str, year: int) -> date | None:
    """'April/May' + '30-1' -> May 1. The decision is announced on the meeting's last day."""
    months = [m.strip() for m in month_txt.split("/")]
    days = re.findall(r"\d+", day_txt)
    if not months or not days or months[-1] not in MONTHS:
        return None
    return date(year, MONTHS[months[-1]], int(days[-1]))


def fomc_meeting_dates() -> list[date]:
    out: list[date] = []
    for y in range(2015, 2021):  # archive pages
        html = get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
        for h in re.findall(r"<h5[^>]*>(.*?)</h5>", html):
            m = re.match(r"^([A-Za-z/]+) ([\d\-]+) Meeting - (\d{4})$", re.sub("<[^>]+>", "", h).strip())
            if m and (d := end_date(m[1], m[2], int(m[3]))):  # unscheduled/cancelled/notation votes don't match
                out.append(d)
    html = get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    parts = re.split(r"(\d{4}) FOMC Meetings", html)
    for year, block in zip(parts[1::2], parts[2::2]):
        months = re.findall(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>', block)
        days = re.findall(r'fomc-meeting__date[^>]*>([^<]+)<', block)
        for mo, dy in zip(months, days):
            if re.search(r"[A-Za-z]", dy.replace("*", "")):  # e.g. "(unscheduled)"
                continue
            if d := end_date(mo, dy, int(year)):
                out.append(d)
    return sorted(set(out))


def bls_release_dates(series: str) -> list[date]:
    """CPI ('cpi') / jobs ('empsit'): dates are encoded in the archive filenames, e.g. cpi_09112026.htm."""
    dates = set()
    for d in re.findall(rf"{series}_(\d{{8}})\.htm", get(f"https://www.bls.gov/bls/news-release/{series}.htm")):
        dates.add(date(int(d[4:]), int(d[:2]), int(d[2:4])))
    sched = re.sub(r"<[^>]+>", " ", get(f"https://www.bls.gov/schedule/news_release/{series}.htm"))
    for mo, dy, yr in re.findall(r"\b([A-Z][a-z]{2})[a-z]*\.? (\d{1,2}), (20\d\d)\b", sched):
        if mo in ABBR:
            dates.add(date(int(yr), ABBR[mo], int(dy)))
    return sorted(d for d in dates if d >= date.fromisoformat(START))


def fed_funds_target() -> pd.Series:
    df = pd.read_csv(io.StringIO(get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU")))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def classify_decisions(meetings: list[date], target: pd.Series) -> dict[str, list[date]]:
    out = {"fomc_cut": [], "fomc_hold": [], "fomc_hike": []}
    for d in meetings:
        before = target.asof(pd.Timestamp(d) - pd.Timedelta(days=1))
        after = target.asof(pd.Timestamp(d) + pd.Timedelta(days=1))
        if pd.isna(before) or pd.isna(after):
            continue
        out["fomc_hike" if after > before + 1e-9 else "fomc_cut" if after < before - 1e-9 else "fomc_hold"].append(d)
    return out


# ---------------------------------------------------------------- price maths
def day_move(closes: pd.Series, day: date | pd.Timestamp, unit: str) -> float | None:
    """Close-to-close move on `day`. pct -> fraction (0.01 = 1%); bp -> basis points (yields quoted in %)."""
    ts = pd.Timestamp(day)
    pos = closes.index.searchsorted(ts)
    if pos == 0 or pos >= len(closes) or closes.index[pos] != ts:
        return None  # not a trading day for this instrument
    cur, prev = float(closes.iloc[pos]), float(closes.iloc[pos - 1])
    return (cur - prev) * 100 if unit == "bp" else cur / prev - 1


def reaction_day(ts: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Which session first reflects an earnings release stamped `ts` (Yahoo gives US/Eastern time)."""
    day = pd.Timestamp(ts.date())
    if 4 <= ts.hour < 12:        # before the open: same session
        target = day
    elif ts.hour >= 15:          # after the close: next session
        target = day + pd.Timedelta(days=1)
    else:
        return None              # midnight / midday stamps are unreliable; skip rather than guess
    pos = trading_days.searchsorted(target)
    return trading_days[pos] if pos < len(trading_days) else None


def macro_stats(dates: list[date], closes: dict[str, pd.Series]) -> dict:
    inst = {}
    for name, sym in INSTRUMENTS.items():
        unit = "bp" if name.endswith("yield") else "pct"
        s = summarize([day_move(closes[sym], d, unit) for d in dates])
        if s:
            inst[name] = s
    return {"n": inst["S&P 500"]["n"], "instruments": inst}


# ---------------------------------------------------------------- earnings
def earnings_reactions(prices: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """{ticker: [('beat'|'miss', reaction_day_return), ...]}"""
    out: dict[str, list[tuple[str, float]]] = {}
    skipped = 0
    for i, sym in enumerate(UNIVERSE, 1):
        if sym not in prices.columns:
            continue
        closes = prices[sym].dropna()
        ed = None
        for attempt in range(3):
            try:
                ed = yf.Ticker(sym).get_earnings_dates(limit=80)
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if ed is None or ed.empty:
            print(f"  [{i}/{len(UNIVERSE)}] {sym}: no earnings data")
            continue
        rows = []
        for ts, r in ed.iterrows():
            sur = r.get("Surprise(%)")
            if pd.isna(sur) or pd.isna(r.get("Reported EPS")) or ts.date() < date.fromisoformat(START) or sur == 0:
                continue
            rd = reaction_day(ts, closes.index)
            if rd is None:
                skipped += 1
                continue
            pos = closes.index.get_loc(rd)
            if pos == 0:
                continue
            rows.append(("beat" if sur > 0 else "miss", float(closes.iloc[pos] / closes.iloc[pos - 1] - 1)))
        out[sym] = rows
        print(f"  [{i}/{len(UNIVERSE)}] {sym}: {len(rows)} events")
        time.sleep(0.3)
    print(f"  ({skipped} events skipped for ambiguous release times)")
    return out


def main() -> None:
    print("fetching prices...")
    syms = list(INSTRUMENTS.values())
    raw = yf.download(syms + UNIVERSE, start=START, auto_adjust=True, progress=False)["Close"]
    closes = {s: raw[s].dropna() for s in syms}

    print("finding event dates...")
    target = fed_funds_target()
    fomc = classify_decisions(fomc_meeting_dates(), target)
    cpi, jobs = bls_release_dates("cpi"), bls_release_dates("empsit")
    print({k: len(v) for k, v in fomc.items()}, "cpi", len(cpi), "jobs", len(jobs))

    events = {}
    for key, label in [("fomc_cut", "Fed rate cuts"), ("fomc_hold", "Fed holds"), ("fomc_hike", "Fed rate hikes")]:
        events[key] = {"label": label, **macro_stats(fomc[key], closes)}
    events["cpi_release"] = {"label": "US CPI releases", **macro_stats(cpi, closes)}
    events["jobs_release"] = {"label": "US jobs reports", **macro_stats(jobs, closes)}

    print("earnings (slow: ~100 tickers)...")
    per_ticker = earnings_reactions(raw[UNIVERSE])
    tickers = {}
    pooled = {"beat": [], "miss": []}
    for sym, rows in per_ticker.items():
        entry = {}
        for kind in ("beat", "miss"):
            vals = [v for k, v in rows if k == kind]
            pooled[kind] += vals
            if s := summarize(vals):
                entry[f"earnings_{kind}"] = s
        tickers[sym] = entry
    for kind, label in [("beat", "large-cap earnings beats (EPS above estimate)"),
                        ("miss", "large-cap earnings misses (EPS below estimate)")]:
        s = summarize(pooled[kind])
        events[f"earnings_{kind}"] = {"label": label, "n": s["n"], "instruments": {"The stock, reaction day": s}}

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump({"built": date.today().isoformat(), "start": START, "events": events, "tickers": tickers}, f, indent=1)
    print(f"wrote {DATA_PATH}")


if __name__ == "__main__":
    main()

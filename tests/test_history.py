import importlib.util
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from newsbot import history, pricecheck

# scripts/ isn't a package; load the builder by path to test its pure helpers.
_spec = importlib.util.spec_from_file_location(
    "build_history", os.path.join(os.path.dirname(__file__), "..", "scripts", "build_history.py"))
bh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bh)


# ---- statistics: the "no guessing" rule ------------------------------------------------

def test_binomial_p_values():
    assert history.binom_p(10, 20) == pytest.approx(1.0)
    assert history.binom_p(15, 20) == pytest.approx(0.0414, abs=1e-3)
    assert history.binom_p(20, 20) == pytest.approx(2 / 2**20)


def test_direction_only_called_when_statistically_distinguishable_from_a_coin_flip():
    assert history.verdict({"n": 20, "up": 15, "down": 5, "median": 0.003}) == "none"   # p=0.04, not enough
    assert history.verdict({"n": 3804, "up": 2085, "down": 1719, "median": 0.006}) == "up"
    assert history.verdict({"n": 698, "up": 225, "down": 473, "median": -0.02}) == "down"
    assert history.verdict({"n": 7, "up": 7, "down": 0, "median": 0.01}) == "none"       # too few, however lopsided
    assert history.verdict({"n": 0, "up": 0, "down": 0, "median": 0.0}) == "none"


def test_summarize_ignores_missing_values():
    s = history.summarize([0.01, -0.02, None, float("nan"), 0.03, 0.0])
    assert (s["n"], s["up"], s["down"]) == (4, 2, 1)
    assert history.summarize([None]) is None


DATA = {
    "start": "2015-01-01",
    "events": {
        "fomc_cut": {"label": "Fed rate cuts", "n": 9, "instruments": {
            "S&P 500": {"n": 9, "up": 4, "down": 5, "median": -0.0003, "mean": 0.0},
            "10-yr yield": {"n": 9, "up": 4, "down": 5, "median": -2.0, "mean": 0.0}}},
        "earnings_beat": {"label": "large-cap earnings beats", "n": 3804, "instruments": {
            "The stock, reaction day": {"n": 3804, "up": 2085, "down": 1719, "median": 0.0064, "mean": 0.01}}},
    },
    "tickers": {"NVDA": {"earnings_beat": {"n": 43, "up": 27, "down": 16, "median": 0.029, "mean": 0.03}},
                "XYZ": {"earnings_beat": {"n": 2, "up": 2, "down": 0, "median": 0.05, "mean": 0.05}}},
}


def test_describe_formats_small_samples_and_units():
    out = history.describe("fomc_cut", data=DATA)
    assert out.splitlines()[0] == "Past Fed rate cuts since 2015 (9 – small sample):"
    assert "≈ S&P 500: up 4 of 9 days, median -0.03% (no consistent direction)" in out
    assert "≈ 10-yr yield: up 4 of 9 days, median -2bp (no consistent direction)" in out


def test_describe_earnings_adds_ticker_line_only_with_enough_history():
    out = history.describe("earnings_beat", "NVDA", DATA)
    assert "NVDA itself: up 27 of 43 times, median +2.90% (no consistent direction)" in out
    assert "↑ The stock, reaction day: up 2085 of 3804 times" in out
    assert "XYZ itself" not in history.describe("earnings_beat", "XYZ", DATA)   # only 2 past events


def test_describe_returns_none_for_unknown_events_or_missing_data():
    assert history.describe("none", data=DATA) is None
    assert history.describe("fomc_hike", data=DATA) is None
    assert history.describe("fomc_cut", data={}) is None


def test_shipped_data_file_loads_and_covers_every_classifier_event_type():
    from newsbot.classify import EVENT_TYPES

    data = history.load()
    assert data is not None, "newsbot/data/reactions.json missing: run scripts/build_history.py"
    for et in EVENT_TYPES:
        if et != "none":
            assert history.describe(et, data=data), et


# ---- price context: what the market was doing before the news ---------------------------

T0 = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)   # story timestamp


def bars(start, closes, step=5):
    return [(start + timedelta(minutes=step * i), c) for i, c in enumerate(closes)]


def test_context_reports_before_and_since_moves():
    # 13 bars from 12:55 to 13:55 rise 100 -> 101 (an hour before), then keep rising to 102 by 14:30
    b = bars(T0 - timedelta(minutes=65), [100 + i * (1 / 12) for i in range(13)] + [101 + i * 0.2 for i in range(1, 6)])
    ctx = pricecheck.context({"ES=F": b}, {"S&P 500 futures": "ES=F"}, T0, now=T0 + timedelta(minutes=30))
    before, since = ctx.splitlines()
    assert before.startswith("Hour before the news: ↑ S&P 500 futures +1.00%")
    assert since.startswith("Since the news: ↑ S&P 500 futures +")


def test_context_skips_since_when_alert_is_fresh_and_flags_flat_moves():
    b = bars(T0 - timedelta(minutes=65), [100.0] * 14)
    ctx = pricecheck.context({"ES=F": b}, {"S&P 500 futures": "ES=F"}, T0, now=T0 + timedelta(minutes=3))
    assert ctx == "Hour before the news: → S&P 500 futures +0.00%"


def test_context_omits_instruments_when_market_was_closed_or_data_missing():
    stale = bars(T0 - timedelta(hours=6), [100.0] * 10)            # last bar long before the story
    assert pricecheck.context({"NVDA": stale}, {"NVDA": "NVDA"}, T0, now=T0 + timedelta(minutes=30)) is None
    assert pricecheck.context({}, {"NVDA": "NVDA"}, T0) is None
    assert pricecheck.context({"NVDA": stale}, {"NVDA": "NVDA"}, None) is None   # no story timestamp


def test_context_yields_are_shown_in_basis_points():
    b = bars(T0 - timedelta(minutes=65), [4.00] * 7 + [4.06] * 7)   # yields quoted in %: +0.06 = +6bp
    ctx = pricecheck.context({"^TNX": b}, {"10-yr yield": "^TNX"}, T0, now=T0 + timedelta(minutes=30))
    assert "↑ 10-yr yield +6bp" in ctx


def test_symbols_for_event_types():
    assert pricecheck.symbols_for("none", "") == {"S&P 500 futures": "ES=F", "Nasdaq futures": "NQ=F"}
    assert pricecheck.symbols_for("fomc_cut", "")["10-yr yield"] == "^TNX"
    assert pricecheck.symbols_for("earnings_beat", "NVDA")["NVDA"] == "NVDA"
    assert "$(rm -rf /)" not in pricecheck.symbols_for("none", "$(rm -rf /)").values()   # bad ticker ignored


# ---- builder helpers ----------------------------------------------------------------------

def test_meeting_end_date_handles_cross_month_meetings():
    assert bh.end_date("April/May", "30-1", 2019) == date(2019, 5, 1)
    assert bh.end_date("January", "28-29*", 2025) == date(2025, 1, 29)
    assert bh.end_date("Smarch", "1-2", 2025) is None


def test_decisions_classified_from_the_target_rate_path():
    idx = pd.date_range("2024-01-01", "2024-12-31")
    target = pd.Series(5.5, index=idx)
    target[target.index >= "2024-09-19"] = 5.0     # a cut announced on 9/18 takes effect the next day
    target[target.index >= "2024-12-19"] = 4.75    # and another
    got = bh.classify_decisions([date(2024, 7, 31), date(2024, 9, 18), date(2024, 12, 18)], target)
    assert got["fomc_hold"] == [date(2024, 7, 31)]
    assert got["fomc_cut"] == [date(2024, 9, 18), date(2024, 12, 18)]
    assert got["fomc_hike"] == []


def test_day_move_uses_previous_close_and_ignores_non_trading_days():
    s = pd.Series([100.0, 102.0, 101.0], index=pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-24"]))
    assert bh.day_move(s, date(2026, 9, 22), "pct") == pytest.approx(0.02)
    assert bh.day_move(s, date(2026, 9, 23), "pct") is None          # no bar that day
    assert bh.day_move(s, date(2026, 9, 21), "pct") is None          # no previous close
    y = pd.Series([4.00, 4.06], index=pd.to_datetime(["2026-09-21", "2026-09-22"]))
    assert bh.day_move(y, date(2026, 9, 22), "bp") == pytest.approx(6.0)


def test_earnings_reaction_day_before_open_after_close_and_ambiguous():
    days = pd.bdate_range("2026-09-14", "2026-09-30")
    ts = lambda h: pd.Timestamp("2026-09-23 %02d:00" % h)   # a Wednesday
    assert bh.reaction_day(ts(7), days) == pd.Timestamp("2026-09-23")    # before the open: same day
    assert bh.reaction_day(ts(16), days) == pd.Timestamp("2026-09-24")   # after the close: next day
    assert bh.reaction_day(pd.Timestamp("2026-09-25 16:00"), days) == pd.Timestamp("2026-09-28")  # Fri -> Mon
    assert bh.reaction_day(ts(0), days) is None                          # unreliable stamp: skipped, not guessed
    assert bh.reaction_day(ts(13), days) is None

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from newsbot import classify as clf
from newsbot import main, notify
from newsbot.fetch import Item, JUNK_TITLE, paragraph_text
from newsbot.sources import BROAD_KEYWORDS, Source
from newsbot.state import State, title_tokens

NOW = datetime.now(timezone.utc)
OFFICIAL = Source("Fed", "u", priority=0)
WIRE = Source("PRN", "u", prefilter=True, priority=3)
TRUMP = Source("Trump", "u", fuzzy_dedupe=False, priority=0)


def item(uid, title, source=OFFICIAL, age_min=5, text=""):
    return Item(source, uid, title, text, f"https://x/{uid}", NOW - timedelta(minutes=age_min))


# ---- state / dedupe ----------------------------------------------------------

def test_near_duplicate_headlines_match_but_distinct_ones_dont(tmp_path):
    s = State(str(tmp_path / "s.json"))
    s.mark("a", "Fed holds interest rates steady, signals cuts later this year")
    assert s.is_near_duplicate("Fed holds rates steady and signals cuts later this year")
    assert not s.is_near_duplicate("Bank of Canada cuts rates by 25 basis points")
    assert not s.is_near_duplicate("Oil jumps")  # too short to compare meaningfully


def test_state_roundtrip_and_fresh_flag(tmp_path):
    path = str(tmp_path / "sub" / "s.json")
    s = State(path)
    assert s.fresh
    s.mark("a", "Fed holds interest rates steady today")
    s.save()
    s2 = State(path)
    assert not s2.fresh and s2.is_seen("a") and s2.is_near_duplicate("Fed holds interest rates steady today")


def test_corrupt_state_is_treated_as_first_run(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert State(str(p)).fresh


def test_reserved_titles_dedupe_within_run_but_are_never_persisted(tmp_path):
    path = str(tmp_path / "s.json")
    s = State(path)
    s.reserve_title("Fed holds interest rates steady, signals cuts")
    assert s.is_near_duplicate("Fed holds interest rates steady, signals cuts")
    s.save()
    assert not State(path).is_near_duplicate("Fed holds interest rates steady, signals cuts")


def test_title_tokens_ignore_case_and_stopwords():
    assert title_tokens("The Fed HOLDS rates") == title_tokens("fed holds rates")


# ---- candidate selection --------------------------------------------------------

def test_select_candidates(tmp_path):
    s = State(str(tmp_path / "s.json"))
    s.fresh = False
    items = [
        item("old", "Ancient news about the economy today", age_min=60 * 24),
        item("prn-noise", "Acme announces new webinar series for customers", WIRE),
        item("prn-deal", "Acme to be acquired by Bigco in $2 billion deal", WIRE),
        item("wire-dup", "Fed holds interest rates steady, signals cuts", WIRE.__class__("CNBC", "u", priority=3)),
        item("fed", "Fed holds interest rates steady, signals cuts"),
    ]
    got = [i.uid for i in main.select_candidates(items, s, timedelta(hours=6))]
    # Official source is processed first, so the CNBC copy is the dropped duplicate.
    assert got == ["fed", "prn-deal"]
    assert s.is_seen("old") and s.is_seen("prn-noise") and s.is_seen("wire-dup")
    assert not s.is_seen("fed")  # candidates stay unseen until classified


def test_fuzzy_dedupe_can_be_disabled_per_source(tmp_path):
    s = State(str(tmp_path / "s.json"))
    s.fresh = False
    a = item("t1", "Thank you for your attention to this matter President Trump", TRUMP)
    b = item("t2", "Thank you for your attention to this matter President Trump", TRUMP)
    assert len(main.select_candidates([a, b], s, timedelta(hours=6))) == 2


def test_junk_titles_and_prefilter():
    for junk in ["ERG.MU", "7US.TG - | Stock Price & Latest News", "[No Title] - Post from September 23, 2026"]:
        assert JUNK_TITLE.search(junk)
    assert not JUNK_TITLE.search("US.-China trade truce extended")
    assert BROAD_KEYWORDS.search("Company X reports second quarter results")
    assert not BROAD_KEYWORDS.search("Company X launches loyalty app")


def test_paragraph_text_skips_nav_and_scripts():
    html = "<nav>Menu</nav><script>var x=1</script><p>The Committee decided to <b>lower</b> the rate.</p>"
    assert paragraph_text(html) == "The Committee decided to lower the rate."


# ---- classifier ------------------------------------------------------------------

class FakeClient:
    def __init__(self, rows=None, exc=None):
        self.rows, self.exc, self.calls = rows, exc, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={"items": self.rows})])


def row(i, imp, summary="Something happened.", cat="rates", direction="up"):
    return {"id": i, "importance": imp, "category": cat, "direction": direction, "summary": summary}


def test_classify_batch_parses_clamps_and_skips_bad_rows():
    batch = [item("a", "one"), item("b", "two"), item("c", "three")]
    client = FakeClient([row(0, 9), row(1, 3, cat="bogus"), {"id": 7, "importance": 5}, {"id": 2}])
    out = clf.classify_batch(client, batch)
    assert set(out) == {0, 1}  # id 7 out of range, id 2 malformed, item 2 unanswered
    assert out[0].importance == 5 and out[1].category == "other"
    kw = client.calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "report"} and kw["model"] == clf.MODEL


def test_classify_returns_none_for_failed_batches():
    res = clf.classify(FakeClient(exc=RuntimeError("529 overloaded")), [item("a", "one")])
    assert res[0][1] is None


def test_classify_batches_large_inputs():
    items = [item(str(i), f"headline {i}") for i in range(clf.BATCH_SIZE + 1)]
    client = FakeClient([row(0, 1, "")])
    clf.classify(client, items)
    assert len(client.calls) == 2


# ---- ntfy -------------------------------------------------------------------------

def test_ntfy_payload(monkeypatch):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, json=json, headers=headers)
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(notify.requests, "post", fake_post)
    v = clf.Verdict(5, "rates", "down", "Fed hikes 25bp; typically pressures equities and lifts USD.")
    notify.send(item("a", "Fed raises rates — “unexpected”"), v, "my-topic", "https://ntfy.example/", "tk_1")
    assert sent["url"] == "https://ntfy.example/"
    assert sent["json"]["topic"] == "my-topic" and sent["json"]["priority"] == 5
    assert sent["json"]["message"].startswith("Fed hikes") and sent["json"]["click"] == "https://x/a"
    assert sent["json"]["title"].startswith("📉 Fed:") and sent["headers"] == {"Authorization": "Bearer tk_1"}


# ---- full run ---------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # so load_dotenv can't pick up a real .env
    monkeypatch.setenv("NTFY_TOPIC", "t")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(main.clf, "MODEL", "m")
    return tmp_path


def wire(monkeypatch, items, client):
    monkeypatch.setattr(main, "fetch_all", lambda: items)
    monkeypatch.setattr(main, "enrich", lambda it: None)
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: client)
    sent = []
    monkeypatch.setattr(main.notify, "send", lambda it, v, *a: sent.append((it.uid, v.importance)))
    return sent


def test_first_run_seeds_silently_then_alerts_on_new_items(env, monkeypatch):
    old = [item("old", "Old story about interest rates and inflation")]
    sent = wire(monkeypatch, old, FakeClient([row(0, 5)]))
    assert main.run() == 0 and sent == []  # seeded, no alerts for the backlog

    new = old + [item("new", "Fed cuts interest rates by half a point in surprise move"),
                 item("meh", "Analyst previews next week's economic calendar events")]
    sent = wire(monkeypatch, new, FakeClient([row(0, 5), row(1, 2, "")]))
    assert main.run() == 0
    assert sent == [("new", 5)]

    # Third pass: everything is now known, nothing is re-alerted or re-classified.
    client = FakeClient([])
    sent = wire(monkeypatch, new, client)
    main.run()
    assert sent == [] and client.calls == []


def test_classifier_outage_leaves_items_unseen_so_they_retry(env, monkeypatch):
    wire(monkeypatch, [item("old", "Old story about interest rates today")], FakeClient([]))
    main.run()  # seed
    story = [item("s", "Central bank surprises markets with emergency rate hike")]
    sent = wire(monkeypatch, story, FakeClient(exc=RuntimeError("boom")))
    main.run()
    assert sent == []
    sent = wire(monkeypatch, story, FakeClient([row(0, 4)]))
    main.run()
    assert sent == [("s", 4)]


def test_ntfy_failure_leaves_item_unseen_so_it_retries(env, monkeypatch):
    wire(monkeypatch, [item("old", "Old story about interest rates today")], FakeClient([]))
    main.run()
    story = [item("s", "Central bank surprises markets with emergency rate hike")]
    wire(monkeypatch, story, FakeClient([row(0, 4)]))

    def boom(*a):
        raise RuntimeError("ntfy down")

    monkeypatch.setattr(main.notify, "send", boom)
    assert main.run() == 0
    sent = wire(monkeypatch, story, FakeClient([row(0, 4)]))
    main.run()
    assert sent == [("s", 4)]


def test_notification_cap_sends_highest_importance_first(env, monkeypatch):
    monkeypatch.setenv("MAX_NOTIFICATIONS_PER_RUN", "2")
    wire(monkeypatch, [item("old", "Old story about interest rates today")], FakeClient([]))
    main.run()
    stories = [item(f"s{i}", f"Distinct market headline number {i} alpha{i} beta{i} gamma{i}") for i in range(3)]
    sent = wire(monkeypatch, stories, FakeClient([row(0, 3), row(1, 5), row(2, 4)]))
    main.run()
    assert [u for u, _ in sent] == ["s1", "s2"]


def test_missing_config_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main.run() == 2

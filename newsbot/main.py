"""One polling pass: fetch -> dedupe -> classify -> notify. Run it from cron / Actions every 1-5 min."""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from . import classify as clf
from . import notify
from .fetch import Item, enrich, fetch_all
from .sources import BROAD_KEYWORDS
from .state import State

log = logging.getLogger("newsbot")


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def select_candidates(items: list[Item], state: State, max_age: timedelta) -> list[Item]:
    """Everything that's new, fresh, on-topic and not a duplicate. Everything else is marked seen
    so it is never looked at again."""
    now = datetime.now(timezone.utc)
    candidates: list[Item] = []
    # Official sources first, so if Reuters and the Fed both carry a story, the Fed's copy wins.
    ordered = sorted(items, key=lambda i: (i.source.priority, -(i.published or now).timestamp()))
    for it in ordered:
        if state.is_seen(it.uid):
            continue
        if it.published and now - it.published > max_age:
            state.mark(it.uid, it.title)
        elif it.source.prefilter and not BROAD_KEYWORDS.search(f"{it.title} {it.text}"):
            state.mark(it.uid)
        elif it.source.fuzzy_dedupe and state.is_near_duplicate(it.title):
            state.mark(it.uid)
        else:
            candidates.append(it)
            state.reserve_title(it.title)
    return candidates


def run(dry_run: bool = False) -> int:
    load_dotenv()
    topic = os.environ.get("NTFY_TOPIC")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not topic or not api_key:
        log.error("NTFY_TOPIC and ANTHROPIC_API_KEY must be set")
        return 2
    min_importance = int(os.environ.get("MIN_IMPORTANCE", "3"))
    max_age = timedelta(hours=float(os.environ.get("MAX_AGE_HOURS", "6")))
    max_notifs = int(os.environ.get("MAX_NOTIFICATIONS_PER_RUN", "10"))
    model = os.environ.get("CLASSIFIER_MODEL", clf.MODEL)
    server = os.environ.get("NTFY_SERVER") or "https://ntfy.sh"  # `or`: Actions exports unset secrets as ""
    token = os.environ.get("NTFY_TOKEN") or None

    state = State(os.environ.get("STATE_PATH", "state/seen.json"))
    items = fetch_all()
    if not items:
        log.error("no items fetched from any source; network or feed outage?")
        return 1

    if state.fresh and not dry_run:
        # First run (or the state cache was lost): record the backlog without alerting on it.
        # (A dry run skips this so it shows what the bot would alert on right now.)
        for it in items:
            state.mark(it.uid, it.title)
        state.save()
        log.info("seeded state with %d items; alerts start next run", len(items))
        return 0

    candidates = select_candidates(items, state, max_age)
    log.info("%d new candidates", len(candidates))
    for it in candidates:
        enrich(it)

    alerts: list[tuple[Item, clf.Verdict]] = []
    if candidates:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        for it, verdict in clf.classify(client, candidates, model):
            if verdict is None:
                continue  # left unseen; retried next run (bounded by MAX_AGE_HOURS)
            log.info("[%d] %s: %s", verdict.importance, it.source.name, it.title[:100])
            if verdict.importance >= min_importance:
                alerts.append((it, verdict))
            else:
                state.mark(it.uid, it.title)

    alerts.sort(key=lambda a: -a[1].importance)
    for n, (it, v) in enumerate(alerts):
        if n >= max_notifs:
            log.warning("over cap: dropping alert %r", it.title[:80])
            state.mark(it.uid, it.title)
            continue
        if dry_run:
            print(f"\n[{v.importance}/{v.category}/{v.direction}] {it.source.name}: {it.title}\n  -> {v.summary}")
            continue
        try:
            notify.send(it, v, topic, server, token)
            state.mark(it.uid, it.title)
        except Exception as e:
            log.error("ntfy send failed (will retry next run): %s", e)

    if not dry_run:
        state.save()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print alerts instead of sending; don't save state")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run(args.dry_run))


if __name__ == "__main__":
    main()

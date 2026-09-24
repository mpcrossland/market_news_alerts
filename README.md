# market-news-bot

Polls free news sources every few minutes, drops duplicates, has Claude Haiku decide whether each new
item is market-moving, and pushes an alert to your phone via [ntfy](https://ntfy.sh). **The bot never
predicts.** Each alert is a short headline plus facts:

```
📱 Fed raises rates by a quarter point
Hour before the news: → S&P 500 futures +0.02%, ↓ 10-yr yield -1bp        <- what markets were doing
Since the news: ↑ S&P 500 futures +0.30%                                  <- how much has already moved

Past Fed rate hikes since 2015 (20):                                      <- what past events like it did
≈ S&P 500: up 6 of 20 days, median -0.26% (no consistent direction)
— Fed
```

Anything without history says "No historical data for this type of news." instead of guessing.

```
feeds -> dedupe -> Haiku (importance, headline, event type) -> history lookup + live price check -> ntfy
```

## Where the numbers come from

- **Historical reactions** (`newsbot/data/reactions.json`, built by `scripts/build_history.py` from Yahoo
  prices, FRED, the Fed and BLS date lists): Fed cut/hold/hike days, US CPI and jobs-report days, and
  earnings beats/misses for ~100 large caps (EPS vs analyst estimate), all since 2015. A direction (↑/↓) is only
  stated when it's statistically distinguishable from a coin flip (p < 0.01, at least 8 cases); otherwise
  it says "no consistent direction". Rebuild monthly: `.venv/bin/python scripts/build_history.py`.
- **Live price check** (`newsbot/pricecheck.py`): 5-minute Yahoo bars for S&P/Nasdaq futures (plus the 10-yr
  yield for macro news and the company's stock when known). Prices are unofficial Yahoo data, so this
  can fail (e.g. Yahoo throttling GitHub's servers); the alert then goes out without that line.

Known limits: CPI and jobs are reported unconditionally, because markets react to the surprise vs
expectations and there is no free source for expectations. The earnings universe is today's large caps
(survivorship bias) and "beat" means EPS only, not guidance. Not yet covered: M&A, Trump posts, oil/
geopolitics, Bank of Canada / StatCan.

## Sources

| Tier | Sources | Notes |
|---|---|---|
| Official | Fed press releases, BLS (CPI / Jobs / PPI / JOLTS), StatCan *The Daily*, Bank of Canada, Trump Truth Social | Near real-time. Fed/BoC items are title-only in RSS, so the bot fetches the page text. Truth Social is read via the trumpstruth.org RSS mirror (truthsocial.com itself blocks bots with a 403). |
| Wires | CNBC (top/economy/world/energy/politics), PR Newswire (keyword pre-filtered) | PR Newswire is a firehose; only earnings/M&A/legal/etc. keywords reach the LLM. |
| Reuters | 9 category queries via Google News RSS | Reuters has no free RSS. Google News lags Reuters by up to ~1h, so lean on the official and CNBC feeds for speed. |

Edit [newsbot/sources.py](newsbot/sources.py) to add or remove feeds. These are release *feeds*, not
calendars: you get alerted when the Fed/BLS/StatCan publish, not ahead of time.

## Setup (same pattern as tv-price-bot)

1. **ntfy**: subscribe in the ntfy phone app to a new hard-to-guess topic (topics on ntfy.sh are
   public, so don't pick something obvious). Use a different topic from tv-price-bot so you can
   mute the two independently.
2. **Repo**: create a *private* GitHub repo and push this folder.
3. **Secrets** (Settings → Secrets and variables → Actions):
   - `ANTHROPIC_API_KEY`
   - `NTFY_TOPIC` (tv-price-bot hard-codes its topic in the workflow; that's fine here too if you'd rather, but the API key must stay a secret)
4. **Test**: Actions tab → "News check" → Run workflow. The first run only seeds state (no
   alerts); the next run alerts on anything new.

Optional env: `NTFY_SERVER`, `NTFY_TOKEN` (self-hosted/protected ntfy), `MIN_IMPORTANCE`,
`MAX_AGE_HOURS`, `MAX_NOTIFICATIONS_PER_RUN`, `CLASSIFIER_MODEL` (see `.env.example`).

**State** (dedup memory) is committed back to the repo like tv-price-bot's `prices.json`, but to a
separate `state` branch as one force-pushed commit, so 5-minute polling doesn't add ~288 commits/day
to `main`. Don't delete that branch; if you do, the next run re-seeds and you'll miss at most one
interval of news.

**Cron caveats**: 5 min is GitHub's minimum, scheduled runs are frequently delayed several minutes,
and GitHub disables scheduled workflows after 60 days without repo activity (re-enable under the
Actions tab). **Want 1-2 min polling?** Run it from any always-on machine instead:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# crontab -e   (needs a .env with ANTHROPIC_API_KEY and NTFY_TOPIC in that directory)
*/2 * * * * cd /path/to/market-news-bot && .venv/bin/python -m newsbot.main >> bot.log 2>&1
```

## Behaviour worth knowing

- **First run seeds silently.** It records the current backlog without alerting, so you don't get
  ~100 notifications on day one.
- **Failures retry.** If the Haiku call or ntfy fails, the item stays unseen and is retried next run
  (until it is older than `MAX_AGE_HOURS`).
- **Caps.** At most `MAX_NOTIFICATIONS_PER_RUN` alerts per run, highest importance first.
- **Priority.** ntfy priority follows importance (5 = urgent, 4 = high, 3 = default).
- **Cost.** Items are classified in batches of 15; importance 1-2 items return an empty headline to
  save output tokens. Expect on the order of a dollar or two a day on Haiku; check your usage after
  the first day and tune `MIN_IMPORTANCE` / the queries in `sources.py` if it's higher.
- News text is treated as untrusted: the classifier only returns structured fields through a forced tool call.

## Development

```bash
.venv/bin/python -m pytest -q                       # unit + end-to-end (stubbed LLM/ntfy)
.venv/bin/python -m newsbot.main --dry-run          # real feeds + real Haiku, prints instead of pushing, saves nothing
```

`--dry-run` still needs `NTFY_TOPIC` set (any value) and a real `ANTHROPIC_API_KEY`. It skips the
first-run seeding, so it classifies everything from the last `MAX_AGE_HOURS` and shows what would alert.

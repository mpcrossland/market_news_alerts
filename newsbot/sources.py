"""Feed definitions.

Three tiers, which decide how much filtering an item gets before it costs an LLM call:

* official  - Fed / BLS / StatCan / BoC / Trump. Low volume, always worth classifying.
* targeted  - Google News queries scoped to reuters.com. The query is the filter.
* broad     - PR Newswire firehose. Keyword pre-filter first, or we'd classify hundreds
              of routine releases an hour.
"""
import re
from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    prefilter: bool = False   # apply BROAD_KEYWORDS before classifying
    enrich: bool = False      # feed text is just a title; fetch the page body
    fuzzy_dedupe: bool = True  # collapse near-identical headlines across sources
    priority: int = 1         # lower sorts first, so the primary source wins a dedupe tie
    strip_suffix: bool = False  # Google News appends " - Publisher" to titles


def _gnews(name: str, query: str) -> Source:
    # when:6h, not shorter: at 2-3h Google pads the results with unrelated "latest" stories.
    # Reuters has no free RSS, so this goes through Google News (which lags Reuters by up to ~1h).
    q = quote_plus(f"site:reuters.com ({query}) when:6h")
    return Source(
        name=f"Reuters/{name}",
        url=f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
        priority=2,
        strip_suffix=True,
    )


SOURCES: list[Source] = [
    # ---- official releases ------------------------------------------------
    Source("Fed", "https://www.federalreserve.gov/feeds/press_all.xml", enrich=True, priority=0),
    Source("BLS CPI", "https://www.bls.gov/feed/cpi.rss", priority=0),
    Source("BLS Jobs", "https://www.bls.gov/feed/empsit.rss", priority=0),
    Source("BLS PPI", "https://www.bls.gov/feed/ppi.rss", priority=0),
    Source("BLS JOLTS", "https://www.bls.gov/feed/jolts.rss", priority=0),
    Source("StatCan", "https://www150.statcan.gc.ca/n1/rss/dai-quo/0-eng.rss", priority=0),
    Source("BoC", "https://www.bankofcanada.ca/content_type/press-releases/feed/", enrich=True, priority=0),
    # trumpstruth.org mirrors @realDonaldTrump; truthsocial.com itself 403s bots (Cloudflare).
    Source("Trump/Truth Social", "https://www.trumpstruth.org/feed", fuzzy_dedupe=False, priority=0),
    # ---- wires --------------------------------------------------------------
    *[
        Source(f"CNBC/{name}", f"https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id={fid}", priority=3)
        for name, fid in [("Top", 100003114), ("Economy", 20910258), ("World", 100727362),
                          ("Energy", 19836768), ("Politics", 10000113)]
    ],
    Source("PRNewswire", "https://www.prnewswire.com/rss/news-releases-list.rss", prefilter=True, priority=3),
    Source(
        "PRNewswire/Financial",
        "https://www.prnewswire.com/rss/financial-services-latest-news/financial-services-latest-news-list.rss",
        prefilter=True,
        priority=3,
    ),
    # ---- Reuters, by category ----------------------------------------------
    _gnews("Rates", 'Fed OR FOMC OR ECB OR "Bank of Canada" OR "Bank of England" OR BOJ OR "rate cut" OR "rate hike"'),
    _gnews("Macro", 'CPI OR inflation OR payrolls OR "jobs report" OR unemployment OR GDP OR "retail sales" OR PMI'),
    _gnews("Earnings", 'earnings OR guidance OR "profit warning" OR "raises forecast" OR "cuts forecast"'),
    _gnews("Deals", 'acquire OR acquisition OR merger OR takeover OR buyout OR "to buy" OR "bid for"'),
    _gnews("Tech", '(Apple OR Microsoft OR Nvidia OR Alphabet OR Amazon OR Meta OR Tesla OR OpenAI OR TSMC) AND (probe OR ban OR recall OR outage OR lawsuit OR launches OR unveils OR chips OR layoffs)'),
    _gnews("Policy", 'tariff OR tariffs OR "trade deal" OR "export controls" OR "debt ceiling" OR shutdown OR antitrust OR regulator OR "executive order"'),
    _gnews("Geopolitics", 'war OR ceasefire OR missile OR sanctions OR invasion OR Taiwan OR Iran OR Russia OR "Strait of Hormuz"'),
    _gnews("Commodities", 'oil OR OPEC OR "natural gas" OR gold OR copper OR wheat OR LNG OR pipeline'),
    _gnews("Stress", '"bank failure" OR "bank run" OR "credit crunch" OR default OR downgrade OR "liquidity" OR bailout OR "circuit breaker" OR selloff'),
]

# Cheap pre-filter for the PR Newswire firehose. Deliberately loose: the LLM does the
# real judging, this only exists to avoid paying to classify "Acme Announces Webinar".
BROAD_KEYWORDS = re.compile(
    r"\b(earnings|quarter(ly)? results|financial results|guidance|outlook|forecast|"
    r"acqui(re|res|sition|red)|merger|buyout|takeover|tender offer|definitive agreement|"
    r"to be acquired|strategic alternatives|going private|"
    r"bankruptcy|chapter 11|default|restructuring|delist|"
    r"share repurchase|buyback|special dividend|stock split|"
    r"FDA (approv|clear)|recall|investigation|subpoena|SEC |DOJ|antitrust|"
    r"tariff|sanction|central bank|interest rate)",
    re.IGNORECASE,
)

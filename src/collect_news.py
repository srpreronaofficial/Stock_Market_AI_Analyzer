
"""
Final Google News RSS collector for Stock Market AI Analyzer.

Collects recent news for:
- TCS
- Tata Steel
- Tata Power
- Tata Consumer Products
- Infosys
- Reliance Industries
- Jio Financial Services
- Reliance Industrial Infrastructure
- NIFTY 50

Google News RSS links are resolved to the original publisher URL
whenever possible.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote_plus, urlparse
from urllib.error import HTTPError, URLError
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET
import shutil
import time

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
NEWS_CSV = DATA_DIR / "news_data.csv"
BACKUP_CSV = DATA_DIR / "news_data_previous.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

DAYS_BACK = 30
MAX_ARTICLES_PER_SYMBOL = 30
MAX_RETRIES = 3

SLEEP_BETWEEN_COMPANIES = 2
SLEEP_BETWEEN_URLS = 0.25


# ============================================================
# NEWS SEARCH QUERIES
# ============================================================

NEWS_QUERIES = {

    "TCS.NS": {
        "name": "Tata Consultancy Services",
        "query": '"Tata Consultancy Services" when:30d',
        "keywords": [
            "tata consultancy services",
            "tcs"
        ],
    },

    "TATASTEEL.NS": {
        "name": "Tata Steel",
        "query": '"Tata Steel" when:30d',
        "keywords": [
            "tata steel"
        ],
    },

    "TATAPOWER.NS": {
        "name": "Tata Power",
        "query": '"Tata Power" when:30d',
        "keywords": [
            "tata power"
        ],
    },

    "TATACONSUM.NS": {
        "name": "Tata Consumer Products",
        "query": '"Tata Consumer Products" OR "Tata Consumer" when:30d',
        "keywords": [
            "tata consumer products",
            "tata consumer"
        ],
    },

    "INFY.NS": {
        "name": "Infosys",
        "query": '"Infosys Limited" OR "Infosys" when:30d',
        "keywords": [
            "infosys"
        ],
    },

    "RELIANCE.NS": {
        "name": "Reliance Industries",
        "query": '"Reliance Industries" when:30d',
        "keywords": [
            "reliance industries"
        ],
    },

    "JIOFIN.NS": {
        "name": "Jio Financial Services",
        "query": '"Jio Financial Services" OR "JioFinance" when:30d',
        "keywords": [
            "jio financial services",
            "jiofinance",
            "jio financial"
        ],
    },

    "RIIL.NS": {
        "name": "Reliance Industrial Infrastructure",
        "query": '"Reliance Industrial Infrastructure" OR "RIIL" when:30d',
        "keywords": [
            "reliance industrial infrastructure",
            "riil"
        ],
    },

    "^NSEI": {
        "name": "NIFTY 50",
        "query": '"Nifty 50" OR "Nifty" when:30d',
        "keywords": [
            "nifty 50",
            "nifty"
        ],
    },
}


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    """Normalize text for relevance checking."""
    if not text:
        return ""

    return " ".join(str(text).lower().split())


def is_relevant(title, keywords):
    """Check whether a headline is relevant to the instrument."""
    text = normalize_text(title)

    for keyword in keywords:
        if keyword.lower() in text:
            return True

    return False


def parse_pubdate(pubdate):
    """Convert RSS publication date to UTC datetime."""

    if not pubdate:
        return None

    try:
        dt = parsedate_to_datetime(pubdate)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except Exception:
        return None


def clean_headline(title, source_name):
    """Remove duplicate publisher suffix from headline."""

    if not title:
        return ""

    title = " ".join(title.split()).strip()

    if source_name:

        suffix = f" - {source_name}"

        if title.lower().endswith(
            suffix.lower()
        ):
            title = title[:-len(suffix)].strip()

    return title


def get_domain(source_url, source_name):
    """Extract publisher domain."""

    if source_url:

        try:
            domain = urlparse(
                source_url
            ).netloc

            if domain:
                return domain.replace(
                    "www.",
                    ""
                )

        except Exception:
            pass

    return source_name or ""


# ============================================================
# RESOLVE GOOGLE NEWS URL
# ============================================================

def resolve_article_url(google_url):
    """
    Try to follow the Google News RSS redirect and obtain
    the original publisher URL.

    If resolution fails, the original Google News URL is kept.
    """

    if not google_url:
        return ""

    # Already a publisher URL.
    if "news.google.com" not in google_url:
        return google_url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/154.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,"
                  "application/xml;q=0.9,*/*;q=0.8",
    }

    try:

        request = Request(
            google_url,
            headers=headers,
            method="GET"
        )

        with urlopen(
            request,
            timeout=15
        ) as response:

            final_url = response.geturl()

            if final_url:
                return final_url

    except Exception:
        pass

    # If Google News resolution fails,
    # retain the working RSS URL.
    return google_url


# ============================================================
# FETCH GOOGLE NEWS RSS
# ============================================================

def fetch_google_news(symbol, config):

    query = config["query"]

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/154.0 Safari/537.36"
        )
    }

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(days=DAYS_BACK)
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            request = Request(
                rss_url,
                headers=headers
            )

            with urlopen(
                request,
                timeout=30
            ) as response:

                xml_data = response.read()

            root = ET.fromstring(
                xml_data
            )

            articles = []

            for item in root.findall(
                ".//item"
            ):

                title = item.findtext(
                    "title",
                    ""
                ).strip()

                google_url = item.findtext(
                    "link",
                    ""
                ).strip()

                pubdate = item.findtext(
                    "pubDate",
                    ""
                ).strip()

                source_element = item.find(
                    "source"
                )

                source_name = ""
                source_url = ""

                if source_element is not None:

                    source_name = (
                        source_element.text or ""
                    ).strip()

                    source_url = (
                        source_element.attrib.get(
                            "url",
                            ""
                        )
                    )

                dt = parse_pubdate(
                    pubdate
                )

                if dt is None:
                    continue

                # Keep only recent articles.
                if dt < cutoff:
                    continue

                # Relevance check.
                if not is_relevant(
                    title,
                    config["keywords"]
                ):
                    continue

                title = clean_headline(
                    title,
                    source_name
                )

                if not title or not google_url:
                    continue

                # Resolve Google News URL.
                article_url = resolve_article_url(
                    google_url
                )

                # Prefer publisher domain from RSS source.
                domain = get_domain(
                    source_url,
                    source_name
                )

                articles.append({
                    "Symbol": symbol,
                    "Date": dt.date(),
                    "Headline": title,
                    "Domain": domain,
                    "SourceCountry": "",
                    "URL": article_url,
                    "CollectedAt": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

                if len(articles) >= (
                    MAX_ARTICLES_PER_SYMBOL
                ):
                    break

                time.sleep(
                    SLEEP_BETWEEN_URLS
                )

            return articles

        except HTTPError as exc:

            print(
                f"  Attempt {attempt}/{MAX_RETRIES}: "
                f"HTTP error {exc.code}"
            )

        except URLError as exc:

            print(
                f"  Attempt {attempt}/{MAX_RETRIES}: "
                f"URL error {exc.reason}"
            )

        except ET.ParseError:

            print(
                f"  Attempt {attempt}/{MAX_RETRIES}: "
                "RSS XML parsing failed."
            )

        except Exception as exc:

            print(
                f"  Attempt {attempt}/{MAX_RETRIES}: "
                f"{type(exc).__name__}: {exc}"
            )

        if attempt < MAX_RETRIES:

            wait_time = 5 * attempt

            print(
                f"  Waiting {wait_time} seconds..."
            )

            time.sleep(
                wait_time
            )

    return []


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL GOOGLE NEWS RSS COLLECTION")
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"News file:    {NEWS_CSV}"
    )

    print(
        f"News window:  Last {DAYS_BACK} days"
    )

    print()

    # --------------------------------------------------------
    # BACKUP CURRENT DATA
    # --------------------------------------------------------

    if NEWS_CSV.exists():

        try:

            shutil.copy2(
                NEWS_CSV,
                BACKUP_CSV
            )

            print(
                f"Previous news file backed up to:"
            )

            print(
                BACKUP_CSV
            )

        except Exception as exc:

            print(
                f"Warning: backup failed: {exc}"
            )

    print()

    # --------------------------------------------------------
    # COLLECT
    # --------------------------------------------------------

    all_articles = []

    for index, (
        symbol,
        config
    ) in enumerate(
        NEWS_QUERIES.items(),
        start=1
    ):

        print(
            f"[{index}/{len(NEWS_QUERIES)}] "
            f"{config['name']} ({symbol})"
        )

        print(
            f"  Query: {config['query']}"
        )

        articles = fetch_google_news(
            symbol,
            config
        )

        all_articles.extend(
            articles
        )

        print(
            f"  Articles collected: "
            f"{len(articles)}"
        )

        if index < len(
            NEWS_QUERIES
        ):

            time.sleep(
                SLEEP_BETWEEN_COMPANIES
            )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    columns = [
        "Symbol",
        "Date",
        "Headline",
        "Domain",
        "SourceCountry",
        "URL",
        "CollectedAt",
    ]

    news_df = pd.DataFrame(
        all_articles,
        columns=columns
    )

    if news_df.empty:

        print()
        print(
            "ERROR: No news was collected."
        )

        print(
            "The existing news_data.csv "
            "was NOT replaced."
        )

        return

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    news_df["Date"] = pd.to_datetime(
        news_df["Date"],
        errors="coerce"
    )

    news_df = news_df.dropna(
        subset=[
            "Date",
            "Headline",
            "URL"
        ]
    )

    news_df["Headline"] = (
        news_df["Headline"]
        .astype(str)
        .str.strip()
    )

    news_df["URL"] = (
        news_df["URL"]
        .astype(str)
        .str.strip()
    )

    # Remove duplicate Symbol + URL.
    news_df = news_df.drop_duplicates(
        subset=[
            "Symbol",
            "URL"
        ]
    )

    # Remove duplicate Symbol + headline.
    news_df = news_df.drop_duplicates(
        subset=[
            "Symbol",
            "Headline"
        ]
    )

    # Sort newest first within each symbol.
    news_df = news_df.sort_values(
        [
            "Symbol",
            "Date"
        ],
        ascending=[
            True,
            False
        ]
    )

    news_df = news_df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    news_df.to_csv(
        NEWS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NEWS COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Total articles: {len(news_df)}"
    )

    print()
    print(
        "Articles by company/instrument:"
    )

    counts = (
        news_df["Symbol"]
        .value_counts()
        .reindex(
            NEWS_QUERIES.keys(),
            fill_value=0
        )
    )

    for symbol, count in counts.items():

        print(
            f"{symbol:<18} {count}"
        )

    print()

    print(
        f"Date range: "
        f"{news_df['Date'].min().date()} "
        f"to "
        f"{news_df['Date'].max().date()}"
    )

    print()

    print(
        f"Saved to:\n{NEWS_CSV}"
    )

    print()

    # Check how many URLs are still Google News URLs.
    google_urls = news_df[
        news_df["URL"].str.contains(
            "news.google.com",
            na=False
        )
    ]

    print(
        f"Publisher URLs resolved: "
        f"{len(news_df) - len(google_urls)}"
    )

    print(
        f"Google News fallback URLs: "
        f"{len(google_urls)}"
    )

    print()

    print("Sample headlines and URLs:")

    print(
        news_df[
            [
                "Symbol",
                "Headline",
                "Domain",
                "URL"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
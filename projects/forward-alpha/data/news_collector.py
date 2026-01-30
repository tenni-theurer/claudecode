"""
News Collector - Gather recent news for injection candidates

Collects headlines from multiple sources and formats them as
potential events to inject into the collision engine.
"""
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import requests

from config import CACHE_DIR

# NewsAPI (free tier: 100 requests/day)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Cache settings
NEWS_CACHE_HOURS = 4  # Cache news for 4 hours


def get_cache_path(ticker: str) -> str:
    """Get cache file path for ticker"""
    return CACHE_DIR / f"news_{ticker.lower()}.json"


def is_cache_valid(cache_path: str) -> bool:
    """Check if cache is still valid"""
    if not os.path.exists(cache_path):
        return False

    mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - mtime < timedelta(hours=NEWS_CACHE_HOURS)


def collect_news(ticker: str, company_name: str = None, days: int = 7) -> list[dict]:
    """
    Collect recent news for a ticker

    Returns list of news items:
    [
        {
            "headline": "Apple iPhone sales drop 10% in China",
            "source": "Reuters",
            "date": "2026-01-29",
            "url": "https://...",
            "sentiment": "negative",  # positive/negative/neutral
            "category": "product_sales"
        }
    ]
    """
    cache_path = get_cache_path(ticker)

    # Check cache
    if is_cache_valid(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    news_items = []

    # Try NewsAPI
    if NEWS_API_KEY:
        news_items.extend(_fetch_newsapi(ticker, company_name, days))

    # Try Google News RSS (free, no key needed)
    news_items.extend(_fetch_google_news_rss(ticker, company_name, days))

    # Deduplicate by headline similarity
    news_items = _deduplicate_news(news_items)

    # Sort by date (newest first)
    news_items.sort(key=lambda x: x.get("date", ""), reverse=True)

    # Limit to top 20
    news_items = news_items[:20]

    # Cache results
    with open(cache_path, "w") as f:
        json.dump(news_items, f, indent=2)

    return news_items


def _fetch_newsapi(ticker: str, company_name: str, days: int) -> list[dict]:
    """Fetch from NewsAPI"""
    items = []

    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Search by ticker and company name
        queries = [ticker]
        if company_name:
            queries.append(company_name.split()[0])  # First word of company name

        for query in queries:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": 10,
                "apiKey": NEWS_API_KEY,
            }

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for article in data.get("articles", []):
                    items.append({
                        "headline": article.get("title", ""),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "date": article.get("publishedAt", "")[:10],
                        "url": article.get("url", ""),
                        "description": article.get("description", ""),
                    })
    except Exception as e:
        print(f"NewsAPI error: {e}")

    return items


def _fetch_google_news_rss(ticker: str, company_name: str, days: int) -> list[dict]:
    """Fetch from Google News RSS (free, no key needed)"""
    import xml.etree.ElementTree as ET

    items = []

    try:
        # Search query
        query = f"{ticker} stock"
        if company_name:
            query = f"{company_name} {ticker}"

        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"

        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)

            for item in root.findall(".//item")[:15]:
                title = item.find("title")
                pub_date = item.find("pubDate")
                link = item.find("link")
                source = item.find("source")

                if title is not None:
                    # Parse date
                    date_str = ""
                    if pub_date is not None and pub_date.text:
                        try:
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(pub_date.text)
                            date_str = dt.strftime("%Y-%m-%d")
                        except:
                            date_str = datetime.now().strftime("%Y-%m-%d")

                    items.append({
                        "headline": title.text or "",
                        "source": source.text if source is not None else "Google News",
                        "date": date_str,
                        "url": link.text if link is not None else "",
                    })
    except Exception as e:
        print(f"Google News RSS error: {e}")

    return items


def _deduplicate_news(items: list[dict]) -> list[dict]:
    """Remove duplicate headlines"""
    seen = set()
    unique = []

    for item in items:
        # Create simple hash of headline
        headline = item.get("headline", "").lower()
        # Normalize: remove common prefixes and punctuation
        normalized = headline[:50]  # First 50 chars

        if normalized not in seen:
            seen.add(normalized)
            unique.append(item)

    return unique


def format_as_injection_candidates(news_items: list[dict]) -> list[str]:
    """
    Format news items as injection candidate strings

    Returns list of strings suitable for injection:
    [
        "Apple iPhone sales reportedly down 10% in China (Reuters, Jan 29)",
        "Analyst upgrades Apple to Buy citing AI potential (Bloomberg, Jan 28)",
    ]
    """
    candidates = []

    for item in news_items:
        headline = item.get("headline", "").strip()
        source = item.get("source", "")
        date = item.get("date", "")

        # Clean up headline (remove source suffix if present)
        if " - " in headline:
            headline = headline.split(" - ")[0]

        # Format date nicely
        date_str = ""
        if date:
            try:
                dt = datetime.strptime(date, "%Y-%m-%d")
                date_str = dt.strftime("%b %d")
            except:
                date_str = date

        # Build candidate string
        if source and date_str:
            candidate = f"{headline} ({source}, {date_str})"
        elif source:
            candidate = f"{headline} ({source})"
        else:
            candidate = headline

        if candidate and len(candidate) > 10:
            candidates.append(candidate)

    return candidates


def collect_injection_candidates(ticker: str, company_name: str = None) -> list[str]:
    """
    Main function: collect news and return as injection candidates

    Usage:
        candidates = collect_injection_candidates("AAPL", "Apple Inc.")
        for i, c in enumerate(candidates):
            print(f"{i+1}. {c}")
    """
    news = collect_news(ticker, company_name)
    return format_as_injection_candidates(news)


if __name__ == "__main__":
    # Test
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"Collecting news for {ticker}...")
    candidates = collect_injection_candidates(ticker, "Apple Inc.")

    print(f"\nFound {len(candidates)} injection candidates:\n")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c}")

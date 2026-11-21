"""Best-effort collection of recent Hindi and Bengali news articles from public RSS feeds.

This gives a small "natural" set (no reference summaries; evaluated reference-free on
factuality). Feeds change often: edit FEEDS if one stops working. Respect each site's terms;
we only fetch a few dozen articles at a polite rate.
"""
from __future__ import annotations

import time

FEEDS = {
    "hi": [
        "https://feeds.bbci.co.uk/hindi/rss.xml",
        "https://navbharattimes.indiatimes.com/rssfeedsdefault.cms",
        "https://www.amarujala.com/rss/breaking-news.xml",
        "https://www.aajtak.in/rssfeeds/?id=home",
        "https://feeds.feedburner.com/ndtvkhabar-latest",
    ],
    "bn": [
        "https://feeds.bbci.co.uk/bengali/rss.xml",
        "https://eisamay.com/rssfeedsdefault.cms",
        "https://www.prothomalo.com/feed/",
        "https://bengali.news18.com/rss/khabar/west-bengal.xml",
        "https://www.sangbadpratidin.in/feed/",
    ],
}
UA = {"User-Agent": "Mozilla/5.0 (FactIndic student research project; contact via GitHub)"}


def scrape(langs=("hi", "bn"), per_lang: int = 100, per_feed: int = 40, pause: float = 1.0, log=print):
    import pandas as pd

    try:
        import feedparser
        import requests
        import trafilatura
    except ImportError as e:
        log(f"  scraping skipped (missing package: {e.name}); pip install feedparser trafilatura")
        return pd.DataFrame(columns=["lang", "text", "url", "title", "source"])
    rows = []
    for lang in langs:
        got = 0
        for feed in FEEDS.get(lang, []):
            if got >= per_lang:
                break
            try:
                r = requests.get(feed, headers=UA, timeout=20)
                parsed = feedparser.parse(r.content)
            except Exception as e:
                log(f"  feed failed: {feed} ({type(e).__name__})")
                continue
            n_feed = 0
            for entry in parsed.entries[:per_feed]:
                if got >= per_lang:
                    break
                url = entry.get("link")
                if not url:
                    continue
                try:
                    html = requests.get(url, headers=UA, timeout=20).text
                    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
                except Exception:
                    continue
                if len(text.split()) < 80:
                    continue
                rows.append({"lang": lang, "text": text, "url": url, "title": entry.get("title", ""),
                             "source": "rss:" + feed.split("/")[2]})
                got += 1
                n_feed += 1
                time.sleep(pause)
            log(f"  {lang}: {n_feed} articles from {feed}")
    return pd.DataFrame(rows)

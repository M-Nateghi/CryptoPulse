from datetime import UTC, datetime

import httpx
import pytest

from cryptopulse.ingestion.google_news import (
    GoogleNewsRssClient,
    clean_rss_summary,
    parse_rss_articles,
)

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>BNB Chain activity rises - Example News</title>
      <link>https://news.google.com/rss/articles/example</link>
      <guid isPermaLink="false">example-guid</guid>
      <pubDate>Mon, 14 Sep 2026 18:25:00 GMT</pubDate>
      <source url="https://example.com">Example News</source>
      <description><![CDATA[
        <a href="https://example.com">BNB Chain activity rises</a>
        Network usage expanded during the latest reporting period. Example News
      ]]></description>
    </item>
  </channel>
</rss>
"""

SECOND_ITEM = b"""
    <item>
      <title>BNB reaches a new milestone - Second Source</title>
      <link>https://news.google.com/rss/articles/second</link>
      <guid isPermaLink="false">second-guid</guid>
      <pubDate>Mon, 14 Sep 2026 19:25:00 GMT</pubDate>
      <source url="https://second.example">Second Source</source>
    </item>
"""


def test_parse_rss_articles_creates_canonical_article():
    retrieved_at = datetime(2026, 9, 15, 12, tzinfo=UTC)

    articles = parse_rss_articles(RSS, "BNB query", retrieved_at)

    assert len(articles) == 1
    assert articles[0].source == "google_news"
    assert articles[0].external_id == "example-guid"
    assert articles[0].title == "BNB Chain activity rises"
    assert (
        articles[0].summary
        == "Network usage expanded during the latest reporting period."
    )
    assert articles[0].published_at == datetime(2026, 9, 14, 18, 25, tzinfo=UTC)
    assert articles[0].retrieved_at == retrieved_at


def test_google_news_client_builds_query_and_honors_limit():
    def handle_request(request):
        assert request.url.path == "/rss/search"
        assert '"BNB Chain"' in request.url.params["q"]
        assert "when:7d" in request.url.params["q"]
        assert request.url.params["ceid"] == "GB:en"
        content = RSS.replace(b"  </channel>", SECOND_ITEM + b"  </channel>")
        return httpx.Response(200, content=content, request=request)

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(base_url="https://example.com", transport=transport) as client:
        articles = GoogleNewsRssClient(client).fetch_recent_articles_for_assets(
            ("BNB",), max_records=1, timespan="7d"
        )

    assert len(articles) == 1


def test_parse_rss_articles_rejects_invalid_xml():
    with pytest.raises(RuntimeError, match="invalid RSS XML"):
        parse_rss_articles(b"not xml", "BNB", datetime.now(UTC))


def test_clean_rss_summary_discards_headline_and_publisher_only_description():
    summary = clean_rss_summary(
        "<a>Bitcoin market update</a> - Example News",
        title="Bitcoin market update",
        publisher="Example News",
    )

    assert summary is None


def test_parse_rss_articles_allows_missing_description():
    content = RSS.replace(
        b"      <description><![CDATA[\n"
        b"        <a href=\"https://example.com\">BNB Chain activity rises</a>\n"
        b"        Network usage expanded during the latest reporting period. Example News\n"
        b"      ]]></description>\n",
        b"",
    )

    articles = parse_rss_articles(content, "BNB query", datetime.now(UTC))

    assert articles[0].summary is None

from datetime import UTC, datetime

import httpx
import pytest

from cryptopulse.ingestion.coindesk import (
    CoinDeskRssClient,
    parse_coindesk_articles,
)

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Bitcoin ETF demand increases</title>
      <link>https://www.coindesk.com/markets/bitcoin-etf</link>
      <guid isPermaLink="false">bitcoin-etf-guid</guid>
      <pubDate>Wed, 16 Sep 2026 10:00:00 +0000</pubDate>
      <description><![CDATA[
        <p>Investment funds reported stronger demand for bitcoin.</p>
      ]]></description>
    </item>
    <item>
      <title>Payments company reports quarterly earnings</title>
      <link>https://www.coindesk.com/business/company-earnings</link>
      <guid isPermaLink="false">earnings-guid</guid>
      <pubDate>Wed, 16 Sep 2026 11:00:00 +0000</pubDate>
      <description>No supported crypto asset is named.</description>
    </item>
  </channel>
</rss>
"""


def test_parse_coindesk_articles_keeps_clean_summary():
    retrieved_at = datetime(2026, 9, 16, 12, tzinfo=UTC)

    articles = parse_coindesk_articles(
        RSS,
        retrieved_at=retrieved_at,
        raw_query="BTC, ETH",
    )

    assert len(articles) == 2
    assert articles[0].source == "coindesk"
    assert articles[0].external_id == "bitcoin-etf-guid"
    assert articles[0].title == "Bitcoin ETF demand increases"
    assert articles[0].summary == (
        "Investment funds reported stronger demand for bitcoin."
    )
    assert articles[0].retrieved_at == retrieved_at


def test_coindesk_client_filters_by_assets_and_timespan():
    def handle_request(request):
        assert request.url.path == "/arc/outboundfeeds/rss"
        return httpx.Response(200, content=RSS, request=request)

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(base_url="https://example.com", transport=transport) as client:
        articles = CoinDeskRssClient(client).fetch_recent_articles_for_assets(
            ("BTC",),
            max_records=10,
            timespan="1d",
            retrieved_at=datetime(2026, 9, 16, 12, tzinfo=UTC),
        )

    assert [article.external_id for article in articles] == ["bitcoin-etf-guid"]


def test_parse_coindesk_articles_rejects_invalid_xml():
    with pytest.raises(RuntimeError, match="invalid RSS XML"):
        parse_coindesk_articles(
            b"not xml",
            retrieved_at=datetime.now(UTC),
            raw_query="BTC",
        )

from datetime import UTC, datetime

import httpx
import pytest

from cryptopulse.ingestion.gdelt import GdeltClient, parse_articles


def test_gdelt_client_fetches_and_parses_articles():
    retrieved_at = datetime(2026, 9, 8, 12, tzinfo=UTC)

    def handle_request(request):
        assert request.url.path == "/api/v2/doc/doc"
        assert request.url.params["mode"] == "artlist"
        assert request.url.params["format"] == "json"
        assert request.url.params["maxrecords"] == "25"
        assert "Ethereum" in request.url.params["query"]
        assert '"ETH"' not in request.url.params["query"]
        assert "sourcelang:english" in request.url.params["query"]
        return httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "url": "https://example.com/ethereum",
                        "title": "Ethereum headline",
                        "seendate": "20260908T113000Z",
                        "language": "English",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(
        base_url="https://api.gdeltproject.org",
        transport=transport,
    ) as http_client:
        articles = GdeltClient(http_client).fetch_recent_articles(
            "ETH",
            max_records=25,
            retrieved_at=retrieved_at,
        )

    assert len(articles) == 1
    assert articles[0].source == "gdelt"
    assert articles[0].title == "Ethereum headline"
    assert articles[0].published_at == datetime(2026, 9, 8, 11, 30, tzinfo=UTC)
    assert articles[0].retrieved_at == retrieved_at


def test_parse_articles_rejects_invalid_dates():
    payload = {
        "articles": [
            {
                "url": "https://example.com/story",
                "title": "Headline",
                "seendate": "not-a-date",
            }
        ]
    }

    with pytest.raises(ValueError, match="Invalid GDELT seendate"):
        parse_articles(
            payload,
            query="Bitcoin",
            retrieved_at=datetime.now(UTC),
        )


def test_gdelt_client_raises_for_http_errors():
    def handle_request(request):
        return httpx.Response(503, request=request)

    transport = httpx.MockTransport(handle_request)
    with (
        httpx.Client(base_url="https://example.com", transport=transport) as http_client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        GdeltClient(http_client, max_retries=0).fetch_recent_articles("SOL")


def test_gdelt_client_retries_rate_limit_responses():
    request_count = 0
    sleep_calls = []

    def handle_request(request):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(429, text="Please limit requests", request=request)
        return httpx.Response(200, json={"articles": []}, request=request)

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(base_url="https://example.com", transport=transport) as http_client:
        articles = GdeltClient(
            http_client,
            max_retries=1,
            retry_delay_seconds=5,
            min_request_interval_seconds=0,
            sleep_func=sleep_calls.append,
        ).fetch_recent_articles("BTC")

    assert articles == []
    assert request_count == 2
    assert sleep_calls == [5]


def test_gdelt_client_reports_non_json_responses_clearly():
    def handle_request(request):
        return httpx.Response(200, text="temporary upstream page", request=request)

    transport = httpx.MockTransport(handle_request)
    with (
        httpx.Client(base_url="https://example.com", transport=transport) as http_client,
        pytest.raises(RuntimeError, match="GDELT returned non-JSON data"),
    ):
        GdeltClient(http_client, max_retries=0).fetch_recent_articles("ETH")


def test_gdelt_client_combines_asset_terms_in_one_request():
    def handle_request(request):
        query = request.url.params["query"]
        assert "Bitcoin" in query
        assert "Ethereum" in query
        assert "Solana" in query
        assert query.count("sourcelang:english") == 1
        return httpx.Response(200, json={"articles": []}, request=request)

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(base_url="https://example.com", transport=transport) as http_client:
        articles = GdeltClient(http_client).fetch_recent_articles_for_assets(
            ("BTC", "ETH", "SOL")
        )

    assert articles == []

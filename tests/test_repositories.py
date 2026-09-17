from datetime import UTC, datetime, timedelta

import pytest

from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article, InsertSummary, MarketCandle
from cryptopulse.db.repositories import (
    deduplicate_cross_source_articles,
    finish_ingestion_run,
    insert_articles,
    insert_market_candles,
    list_articles_for_classification,
    start_ingestion_run,
)
from cryptopulse.db.schema import create_schema
from cryptopulse.sentiment import ClassifierIdentity


def test_classification_query_allows_200_but_rejects_larger_batches(tmp_path):
    identity = ClassifierIdentity(
        provider="openai",
        model="test-model",
        prompt_version="test-prompt",
    )
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        assert list_articles_for_classification(connection, identity, 200) == []
        with pytest.raises(ValueError, match="between 1 and 200"):
            list_articles_for_classification(connection, identity, 201)


def test_insert_articles_reports_and_skips_duplicate_urls(tmp_path):
    database_path = tmp_path / "test.db"
    published_at = datetime(2026, 9, 8, 10, tzinfo=UTC)
    retrieved_at = datetime(2026, 9, 8, 11, tzinfo=UTC)
    articles = [
        Article(
            source="gdelt",
            external_id="first",
            title="First title",
            url="https://example.com/story",
            published_at=published_at,
            retrieved_at=retrieved_at,
            raw_query="Bitcoin OR BTC",
        ),
        Article(
            source="gdelt",
            external_id="second",
            title="Duplicate URL",
            url="https://example.com/story",
            published_at=published_at,
            retrieved_at=retrieved_at,
            raw_query="BTC",
        ),
    ]

    with open_database(database_path) as connection:
        create_schema(connection)
        summary = insert_articles(connection, articles)
        row_count = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    assert summary == InsertSummary(received=2, inserted=1, skipped=1)
    assert row_count == 1


def test_insert_market_candles_is_safe_to_repeat(tmp_path):
    database_path = tmp_path / "test.db"
    candle = MarketCandle(
        asset="BTC",
        symbol="BTCUSDT",
        candle_timestamp=datetime(2026, 9, 8, 10, tzinfo=UTC),
        open=110_000.0,
        high=111_000.0,
        low=109_500.0,
        close=110_500.0,
        volume=120.0,
        quote_volume=13_200_000.0,
        trade_count=1_500,
    )

    with open_database(database_path) as connection:
        create_schema(connection)
        first_summary = insert_market_candles(connection, [candle])
        second_summary = insert_market_candles(connection, [candle])

    assert first_summary == InsertSummary(received=1, inserted=1, skipped=0)
    assert second_summary == InsertSummary(received=1, inserted=0, skipped=1)


def test_duplicate_article_can_be_enriched_with_missing_summary(tmp_path):
    published_at = datetime(2026, 9, 8, 10, tzinfo=UTC)
    base = {
        "source": "google_news",
        "external_id": "story-guid",
        "title": "Bitcoin market update",
        "url": "https://example.com/story",
        "published_at": published_at,
        "retrieved_at": published_at,
        "raw_query": "Bitcoin",
    }
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        first = insert_articles(connection, [Article(**base)])
        second = insert_articles(
            connection,
            [Article(**base, summary="Bitcoin trading volume increased.")],
        )
        stored_summary = connection.execute(
            "SELECT summary FROM articles"
        ).fetchone()["summary"]

    assert first == InsertSummary(received=1, inserted=1, skipped=0)
    assert second == InsertSummary(received=1, inserted=0, skipped=1)
    assert stored_summary == "Bitcoin trading volume increased."


def test_cross_source_near_duplicate_is_skipped_and_enriches_summary(tmp_path):
    published_at = datetime(2026, 9, 8, 10, tzinfo=UTC)
    google_article = Article(
        source="google_news",
        external_id="google-guid",
        title="Bitcoin ETF demand jumps as institutions return",
        url="https://news.google.com/articles/google-guid",
        published_at=published_at,
        retrieved_at=published_at,
        raw_query="Bitcoin",
    )
    coindesk_article = Article(
        source="coindesk",
        external_id="coindesk-guid",
        title="Bitcoin ETF demand jumps as institutions return today",
        summary="ETF issuers reported stronger institutional demand.",
        url="https://www.coindesk.com/markets/bitcoin-etf-demand",
        published_at=published_at + timedelta(hours=1),
        retrieved_at=published_at + timedelta(hours=1),
        raw_query="BTC",
    )

    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        first = insert_articles(connection, [google_article])
        second = insert_articles(connection, [coindesk_article])
        stored = connection.execute(
            "SELECT source, summary FROM articles"
        ).fetchall()

    assert first == InsertSummary(received=1, inserted=1, skipped=0)
    assert second == InsertSummary(received=1, inserted=0, skipped=1)
    assert len(stored) == 1
    assert stored[0]["source"] == "google_news"
    assert stored[0]["summary"] == coindesk_article.summary


def test_cross_source_canonical_url_duplicate_is_skipped(tmp_path):
    published_at = datetime(2026, 9, 8, 10, tzinfo=UTC)
    shared = {
        "external_id": None,
        "title": "Different syndication headlines",
        "published_at": published_at,
        "retrieved_at": published_at,
        "raw_query": "Bitcoin",
    }
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        insert_articles(
            connection,
            [
                Article(
                    source="gdelt",
                    url="https://www.example.com/story/?utm_source=feed",
                    **shared,
                )
            ],
        )
        summary = insert_articles(
            connection,
            [
                Article(
                    source="google_news",
                    url="https://example.com/story?fbclid=tracking",
                    **shared,
                )
            ],
        )

    assert summary == InsertSummary(received=1, inserted=0, skipped=1)


def test_similar_cross_source_title_outside_window_is_kept(tmp_path):
    published_at = datetime(2026, 9, 1, 10, tzinfo=UTC)
    base = {
        "external_id": None,
        "title": "Bitcoin ETF demand jumps as institutions return",
        "retrieved_at": published_at,
        "raw_query": "Bitcoin",
    }
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        insert_articles(
            connection,
            [
                Article(
                    source="google_news",
                    url="https://example.com/first",
                    published_at=published_at,
                    **base,
                )
            ],
        )
        summary = insert_articles(
            connection,
            [
                Article(
                    source="coindesk",
                    url="https://example.com/later",
                    published_at=published_at + timedelta(days=4),
                    **base,
                )
            ],
        )

    assert summary == InsertSummary(received=1, inserted=1, skipped=0)


def test_existing_cross_source_duplicates_are_cleaned_safely(tmp_path):
    published_at = datetime(2026, 9, 8, 10, tzinfo=UTC)
    articles = [
        Article(
            source="google_news",
            external_id="google-guid",
            title="Ethereum upgrade receives final developer approval",
            url="https://news.google.com/articles/ethereum-upgrade",
            published_at=published_at,
            retrieved_at=published_at,
            raw_query="Ethereum",
        ),
        Article(
            source="coindesk",
            external_id="coindesk-guid",
            title="Ethereum upgrade receives final developer approval today",
            summary="Developers approved the network upgrade for deployment.",
            url="https://www.coindesk.com/tech/ethereum-upgrade",
            published_at=published_at + timedelta(hours=1),
            retrieved_at=published_at + timedelta(hours=1),
            raw_query="ETH",
        ),
    ]
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        insert_articles(connection, articles)
        removed = deduplicate_cross_source_articles(connection)
        stored = connection.execute(
            "SELECT source, summary FROM articles"
        ).fetchall()

    assert removed == 1
    assert len(stored) == 1
    assert stored[0]["source"] == "coindesk"
    assert stored[0]["summary"] == articles[1].summary


def test_ingestion_run_can_be_started_and_finished(tmp_path):
    database_path = tmp_path / "test.db"
    started_at = datetime(2026, 9, 8, 10, tzinfo=UTC)
    completed_at = datetime(2026, 9, 8, 10, 1, tzinfo=UTC)

    with open_database(database_path) as connection:
        create_schema(connection)
        run_id = start_ingestion_run(connection, "binance", started_at)
        finish_ingestion_run(
            connection=connection,
            run_id=run_id,
            completed_at=completed_at,
            status="succeeded",
            summary=InsertSummary(received=3, inserted=2, skipped=1),
        )
        run = connection.execute(
            "SELECT * FROM ingestion_runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    assert run["source"] == "binance"
    assert run["status"] == "succeeded"
    assert run["records_received"] == 3
    assert run["records_inserted"] == 2
    assert run["records_skipped"] == 1

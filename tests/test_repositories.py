from datetime import UTC, datetime

from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article, InsertSummary, MarketCandle
from cryptopulse.db.repositories import (
    finish_ingestion_run,
    insert_articles,
    insert_market_candles,
    start_ingestion_run,
)
from cryptopulse.db.schema import create_schema


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

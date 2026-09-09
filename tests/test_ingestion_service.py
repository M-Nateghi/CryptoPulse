from datetime import UTC, datetime

import pytest

from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article, InsertSummary, MarketCandle
from cryptopulse.db.schema import create_schema
from cryptopulse.ingestion.service import ingest_market, ingest_news


class StubBinanceClient:
    def fetch_hourly_candles(self, symbol):
        return [
            MarketCandle(
                asset="BTC",
                symbol=symbol,
                candle_timestamp=datetime(2026, 9, 8, 10, tzinfo=UTC),
                open=110_000.0,
                high=111_000.0,
                low=109_500.0,
                close=110_500.0,
                volume=120.0,
                quote_volume=13_200_000.0,
                trade_count=1_500,
            )
        ]


class StubGdeltClient:
    def fetch_recent_articles_for_assets(self, assets, max_records):
        return [
            Article(
                source="gdelt",
                external_id=None,
                title=f"{asset} headline",
                url=f"https://example.com/{asset.lower()}",
                published_at=datetime(2026, 9, 8, 10, tzinfo=UTC),
                retrieved_at=datetime(2026, 9, 8, 11, tzinfo=UTC),
                raw_query=asset,
            )
            for asset in assets
        ]


class FailingGdeltClient:
    def fetch_recent_articles_for_assets(self, assets, max_records):
        raise RuntimeError(f"GDELT unavailable for {', '.join(assets)}")


def test_market_ingestion_stores_candles_and_skips_them_on_repeat(tmp_path):
    database_path = tmp_path / "test.db"

    with open_database(database_path) as connection:
        create_schema(connection)
        first = ingest_market(connection, StubBinanceClient(), symbols=("BTCUSDT",))
        second = ingest_market(connection, StubBinanceClient(), symbols=("BTCUSDT",))
        run_statuses = connection.execute(
            "SELECT status FROM ingestion_runs ORDER BY id"
        ).fetchall()

    assert first == InsertSummary(received=1, inserted=1, skipped=0)
    assert second == InsertSummary(received=1, inserted=0, skipped=1)
    assert [row["status"] for row in run_statuses] == ["succeeded", "succeeded"]


def test_news_ingestion_stores_articles(tmp_path):
    database_path = tmp_path / "test.db"

    with open_database(database_path) as connection:
        create_schema(connection)
        summary = ingest_news(connection, StubGdeltClient(), assets=("BTC", "ETH"))
        article_count = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    assert summary == InsertSummary(received=2, inserted=2, skipped=0)
    assert article_count == 2


def test_failed_ingestion_is_recorded(tmp_path):
    database_path = tmp_path / "test.db"

    with open_database(database_path) as connection:
        create_schema(connection)
        with pytest.raises(RuntimeError, match="GDELT unavailable"):
            ingest_news(connection, FailingGdeltClient(), assets=("SOL",))
        run = connection.execute(
            "SELECT status, error_message FROM ingestion_runs"
        ).fetchone()

    assert run["status"] == "failed"
    assert "GDELT unavailable for SOL" in run["error_message"]

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from cryptopulse.db.database import open_database, open_readonly_database
from cryptopulse.db.models import Article, MarketCandle
from cryptopulse.db.repositories import (
    insert_articles,
    insert_market_candles,
    save_classification_success,
)
from cryptopulse.db.schema import create_schema
from cryptopulse.demo import export_demo_database
from cryptopulse.sentiment import (
    ArticleClassification,
    AssetSentiment,
    ClassifierIdentity,
)


def _populate_source(database_path):
    start = datetime(2026, 9, 1, tzinfo=UTC)
    with open_database(database_path) as connection:
        create_schema(connection)
        insert_market_candles(
            connection,
            [
                MarketCandle(
                    asset="BTC",
                    symbol="BTCUSDT",
                    candle_timestamp=start + timedelta(hours=hour),
                    open=100 + hour,
                    high=101 + hour,
                    low=99 + hour,
                    close=100.5 + hour,
                    volume=1_000 + hour,
                    quote_volume=100_000 + hour,
                    trade_count=100 + hour,
                )
                for hour in range(30)
            ],
        )
        insert_articles(
            connection,
            [
                Article(
                    source="gdelt",
                    external_id=None,
                    title="Bitcoin ETF demand grows",
                    summary="Fund inflows increased during the latest session.",
                    url="https://example.com/btc-etf",
                    published_at=start,
                    retrieved_at=start,
                    raw_query="Bitcoin",
                )
            ],
        )
        article_id = connection.execute("SELECT id FROM articles").fetchone()["id"]
        for version, score, offset in (("v1", -0.2, 0), ("v2", 0.8, 1)):
            save_classification_success(
                connection,
                article_id,
                ClassifierIdentity(
                    provider="openai",
                    model="test-model",
                    prompt_version=version,
                ),
                "Bitcoin ETF demand grows",
                ArticleClassification(
                    is_relevant=True,
                    asset_sentiments=[
                        AssetSentiment(
                            asset="BTC",
                            sentiment="bullish" if score > 0 else "bearish",
                            sentiment_score=score,
                            confidence=0.9,
                            category="etf_flows",
                            reason="Test evidence.",
                        )
                    ],
                ),
                start + timedelta(hours=offset),
            )


def test_export_demo_keeps_recent_market_and_latest_production_result(tmp_path):
    source_path = tmp_path / "source.db"
    output_path = tmp_path / "demo.db"
    _populate_source(source_path)

    with open_database(source_path) as source:
        summary = export_demo_database(
            source,
            output_path,
            market_hours=24,
            article_days=30,
            generated_at=datetime(2026, 9, 15, tzinfo=UTC),
        )

    with open_readonly_database(output_path) as demo:
        prompt_version = demo.execute(
            "SELECT prompt_version FROM article_classifications"
        ).fetchone()["prompt_version"]
        oldest_market = demo.execute(
            "SELECT MIN(candle_timestamp) FROM market_data"
        ).fetchone()[0]
        metadata = dict(demo.execute("SELECT key, value FROM demo_metadata"))
        exported_summary = demo.execute(
            "SELECT summary FROM articles"
        ).fetchone()["summary"]

    assert summary.market_rows == 24
    assert summary.article_rows == 1
    assert summary.classification_rows == 1
    assert summary.sentiment_rows == 1
    assert prompt_version == "v2"
    assert oldest_market == "2026-09-01T06:00:00Z"
    assert metadata["generated_at"] == "2026-09-15T00:00:00Z"
    assert exported_summary == "Fund inflows increased during the latest session."


def test_export_demo_keeps_recent_irrelevant_classification_state(tmp_path):
    source_path = tmp_path / "source.db"
    output_path = tmp_path / "demo.db"
    published_at = datetime(2026, 9, 14, tzinfo=UTC)
    with open_database(source_path) as source:
        create_schema(source)
        insert_articles(
            source,
            [
                Article(
                    source="gdelt",
                    external_id=None,
                    title="Unrelated company announcement",
                    url="https://example.com/unrelated",
                    published_at=published_at,
                    retrieved_at=published_at,
                    raw_query="Bitcoin",
                )
            ],
        )
        article_id = source.execute("SELECT id FROM articles").fetchone()["id"]
        save_classification_success(
            source,
            article_id,
            ClassifierIdentity(
                provider="openai",
                model="test-model",
                prompt_version="v1",
            ),
            "Unrelated company announcement",
            ArticleClassification(is_relevant=False, asset_sentiments=[]),
            published_at,
        )
        source.commit()
        summary = export_demo_database(
            source,
            output_path,
            generated_at=datetime(2026, 9, 15, tzinfo=UTC),
        )

    with open_readonly_database(output_path) as demo:
        classification = demo.execute(
            "SELECT is_relevant FROM article_classifications"
        ).fetchone()

    assert summary.article_rows == 1
    assert summary.classification_rows == 1
    assert summary.sentiment_rows == 0
    assert classification["is_relevant"] == 0


def test_export_demo_refuses_to_overwrite_by_default(tmp_path):
    source_path = tmp_path / "source.db"
    output_path = tmp_path / "demo.db"
    _populate_source(source_path)
    output_path.touch()

    with (
        open_database(source_path) as source,
        pytest.raises(FileExistsError, match="already exists"),
    ):
        export_demo_database(source, output_path)


def test_readonly_database_rejects_writes(tmp_path):
    database_path = tmp_path / "readonly.db"
    with open_database(database_path) as connection:
        connection.execute("CREATE TABLE example (value TEXT NOT NULL)")

    with (
        open_readonly_database(database_path) as connection,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        connection.execute("INSERT INTO example VALUES ('changed')")

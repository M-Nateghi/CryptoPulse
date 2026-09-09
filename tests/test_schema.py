import sqlite3

import pytest

from cryptopulse.db.database import open_database
from cryptopulse.db.schema import SCHEMA_SQL, create_schema


def test_create_schema_creates_project_tables_and_is_repeatable(tmp_path):
    database_path = tmp_path / "test.db"

    with open_database(database_path) as connection:
        create_schema(connection)
        create_schema(connection)
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

    table_names = {row["name"] for row in table_rows}
    assert table_names == {
        "article_classifications",
        "articles",
        "ingestion_runs",
        "market_data",
        "sentiment_results",
    }


def test_create_schema_migrates_existing_asset_tables_for_bnb(tmp_path):
    database_path = tmp_path / "legacy.db"
    legacy_schema = SCHEMA_SQL.replace(", 'BNB'", "")
    assert "'BNB'" not in legacy_schema

    with open_database(database_path) as connection:
        connection.executescript(legacy_schema)
        connection.execute(
            """
            INSERT INTO market_data (
                asset, symbol, candle_timestamp, open, high, low, close,
                volume, quote_volume, trade_count
            ) VALUES ('BTC', 'BTCUSDT', '2026-09-08T10:00:00Z', 100, 110,
                      90, 105, 12, 1260, 20)
            """
        )
        article_id = connection.execute(
            """
            INSERT INTO articles (
                source, external_id, title, url, published_at, retrieved_at,
                raw_query
            ) VALUES ('gdelt', NULL, 'Bitcoin update',
                      'https://example.com/bitcoin', '2026-09-08T10:00:00Z',
                      '2026-09-08T11:00:00Z', 'Bitcoin')
            """
        ).lastrowid
        classification_id = connection.execute(
            """
            INSERT INTO article_classifications (
                article_id, provider, model, prompt_version, input_text,
                status, is_relevant, processed_at, error_message
            ) VALUES (?, 'fake', 'legacy', 'v1', 'Bitcoin update',
                      'succeeded', 1, '2026-09-08T11:00:00Z', NULL)
            """,
            (article_id,),
        ).lastrowid
        connection.execute(
            """
            INSERT INTO sentiment_results (
                classification_id, asset, sentiment, sentiment_score,
                confidence, category, reason
            ) VALUES (?, 'BTC', 'neutral', 0, 1, 'other', 'Legacy result')
            """,
            (classification_id,),
        )

        create_schema(connection)

        connection.execute(
            """
            INSERT INTO market_data (
                asset, symbol, candle_timestamp, open, high, low, close,
                volume, quote_volume, trade_count
            ) VALUES ('BNB', 'BNBUSDT', '2026-09-08T10:00:00Z', 800, 810,
                      790, 805, 15, 12075, 30)
            """
        )
        connection.execute(
            """
            INSERT INTO sentiment_results (
                classification_id, asset, sentiment, sentiment_score,
                confidence, category, reason
            ) VALUES (?, 'BNB', 'bullish', 0.5, 0.8, 'market_movement',
                      'BNB result')
            """,
            (classification_id,),
        )
        market_assets = connection.execute(
            "SELECT asset FROM market_data ORDER BY id"
        ).fetchall()
        sentiment_assets = connection.execute(
            "SELECT asset FROM sentiment_results ORDER BY id"
        ).fetchall()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert [row["asset"] for row in market_assets] == ["BTC", "BNB"]
    assert [row["asset"] for row in sentiment_assets] == ["BTC", "BNB"]
    assert foreign_key_errors == []


def test_articles_reject_duplicate_source_url(tmp_path):
    database_path = tmp_path / "test.db"
    article = (
        "gdelt",
        None,
        "Bitcoin headline",
        "https://example.com/bitcoin",
        "2026-09-08T10:00:00Z",
        "2026-09-08T10:05:00Z",
        "Bitcoin OR BTC",
    )

    with open_database(database_path) as connection:
        create_schema(connection)
        connection.execute(
            """
            INSERT INTO articles (
                source, external_id, title, url, published_at, retrieved_at, raw_query
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            article,
        )

    with (
        pytest.raises(sqlite3.IntegrityError),
        open_database(database_path) as connection,
    ):
        connection.execute(
            """
            INSERT INTO articles (
                source, external_id, title, url, published_at, retrieved_at, raw_query
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            article,
        )


def test_market_data_rejects_duplicate_symbol_timestamp(tmp_path):
    database_path = tmp_path / "test.db"
    candle = (
        "BTC",
        "BTCUSDT",
        "2026-09-08T10:00:00Z",
        110_000.0,
        111_000.0,
        109_500.0,
        110_500.0,
        120.0,
        13_200_000.0,
        1_500,
    )

    with open_database(database_path) as connection:
        create_schema(connection)
        connection.execute(
            """
            INSERT INTO market_data (
                asset, symbol, candle_timestamp, open, high, low, close,
                volume, quote_volume, trade_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            candle,
        )

    with (
        pytest.raises(sqlite3.IntegrityError),
        open_database(database_path) as connection,
    ):
        connection.execute(
            """
            INSERT INTO market_data (
                asset, symbol, candle_timestamp, open, high, low, close,
                volume, quote_volume, trade_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            candle,
        )

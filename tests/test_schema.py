import sqlite3

import pytest

from cryptopulse.db.database import open_database
from cryptopulse.db.schema import create_schema


def test_create_schema_creates_stage_one_tables_and_is_repeatable(tmp_path):
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
    assert table_names == {"articles", "ingestion_runs", "market_data"}


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

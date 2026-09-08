import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

from cryptopulse.db.models import Article, InsertSummary, MarketCandle


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Database timestamps must include timezone information")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def insert_articles(
    connection: sqlite3.Connection,
    articles: Iterable[Article],
) -> InsertSummary:
    article_list = list(articles)
    rows = [
        (
            article.source,
            article.external_id,
            article.title,
            article.url,
            _utc_text(article.published_at),
            _utc_text(article.retrieved_at),
            article.raw_query,
        )
        for article in article_list
    ]

    changes_before = connection.total_changes
    connection.executemany(
        """
        INSERT INTO articles (
            source, external_id, title, url, published_at, retrieved_at, raw_query
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    inserted = connection.total_changes - changes_before
    return InsertSummary(
        received=len(article_list),
        inserted=inserted,
        skipped=len(article_list) - inserted,
    )


def insert_market_candles(
    connection: sqlite3.Connection,
    candles: Iterable[MarketCandle],
) -> InsertSummary:
    candle_list = list(candles)
    rows = [
        (
            candle.asset,
            candle.symbol,
            _utc_text(candle.candle_timestamp),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.quote_volume,
            candle.trade_count,
        )
        for candle in candle_list
    ]

    changes_before = connection.total_changes
    connection.executemany(
        """
        INSERT INTO market_data (
            asset, symbol, candle_timestamp, open, high, low, close,
            volume, quote_volume, trade_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    inserted = connection.total_changes - changes_before
    return InsertSummary(
        received=len(candle_list),
        inserted=inserted,
        skipped=len(candle_list) - inserted,
    )


def start_ingestion_run(
    connection: sqlite3.Connection,
    source: str,
    started_at: datetime,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO ingestion_runs (source, started_at, status)
        VALUES (?, ?, 'running')
        """,
        (source, _utc_text(started_at)),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an ingestion run ID")
    return cursor.lastrowid


def finish_ingestion_run(
    connection: sqlite3.Connection,
    run_id: int,
    completed_at: datetime,
    status: Literal["succeeded", "failed"],
    summary: InsertSummary,
    error_message: str | None = None,
) -> None:
    cursor = connection.execute(
        """
        UPDATE ingestion_runs
        SET completed_at = ?,
            status = ?,
            records_received = ?,
            records_inserted = ?,
            records_skipped = ?,
            error_message = ?
        WHERE id = ?
        """,
        (
            _utc_text(completed_at),
            status,
            summary.received,
            summary.inserted,
            summary.skipped,
            error_message,
            run_id,
        ),
    )
    if cursor.rowcount != 1:
        raise ValueError(f"Ingestion run {run_id} does not exist")

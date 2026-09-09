import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

from cryptopulse.db.models import (
    Article,
    ArticleForClassification,
    InsertSummary,
    MarketCandle,
)
from cryptopulse.sentiment.models import ArticleClassification
from cryptopulse.sentiment.provider import ClassifierIdentity


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


def _find_classification(
    connection: sqlite3.Connection,
    article_id: int,
    identity: ClassifierIdentity,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT id, status
        FROM article_classifications
        WHERE article_id = ?
          AND provider = ?
          AND model = ?
          AND prompt_version = ?
        """,
        (
            article_id,
            identity.provider,
            identity.model,
            identity.prompt_version,
        ),
    ).fetchone()


def save_classification_success(
    connection: sqlite3.Connection,
    article_id: int,
    identity: ClassifierIdentity,
    input_text: str,
    classification: ArticleClassification,
    processed_at: datetime,
) -> int:
    """Store one validated classification without replacing an earlier success."""
    if not input_text.strip():
        raise ValueError("Classification input text cannot be blank")

    existing = _find_classification(connection, article_id, identity)
    if existing is not None and existing["status"] == "succeeded":
        return existing["id"]

    values = (
        input_text,
        int(classification.is_relevant),
        _utc_text(processed_at),
    )
    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO article_classifications (
                article_id, provider, model, prompt_version, input_text,
                status, is_relevant, processed_at, error_message
            ) VALUES (?, ?, ?, ?, ?, 'succeeded', ?, ?, NULL)
            """,
            (
                article_id,
                identity.provider,
                identity.model,
                identity.prompt_version,
                *values,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a classification ID")
        classification_id = cursor.lastrowid
    else:
        classification_id = existing["id"]
        connection.execute(
            """
            UPDATE article_classifications
            SET input_text = ?,
                status = 'succeeded',
                is_relevant = ?,
                processed_at = ?,
                error_message = NULL
            WHERE id = ?
            """,
            (*values, classification_id),
        )
        connection.execute(
            "DELETE FROM sentiment_results WHERE classification_id = ?",
            (classification_id,),
        )

    connection.executemany(
        """
        INSERT INTO sentiment_results (
            classification_id, asset, sentiment, sentiment_score,
            confidence, category, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                classification_id,
                result.asset.value,
                result.sentiment.value,
                result.sentiment_score,
                result.confidence,
                result.category.value,
                result.reason,
            )
            for result in classification.asset_sentiments
        ],
    )
    return classification_id


def save_classification_failure(
    connection: sqlite3.Connection,
    article_id: int,
    identity: ClassifierIdentity,
    input_text: str,
    error_message: str,
    processed_at: datetime,
) -> int:
    """Store a failed attempt unless a successful result already exists."""
    if not input_text.strip():
        raise ValueError("Classification input text cannot be blank")
    normalized_error = error_message.strip()[:1_000]
    if not normalized_error:
        raise ValueError("Classification failure message cannot be blank")

    existing = _find_classification(connection, article_id, identity)
    if existing is not None and existing["status"] == "succeeded":
        return existing["id"]

    values = (
        input_text,
        _utc_text(processed_at),
        normalized_error,
    )
    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO article_classifications (
                article_id, provider, model, prompt_version, input_text,
                status, is_relevant, processed_at, error_message
            ) VALUES (?, ?, ?, ?, ?, 'failed', NULL, ?, ?)
            """,
            (
                article_id,
                identity.provider,
                identity.model,
                identity.prompt_version,
                *values,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a classification ID")
        return cursor.lastrowid

    connection.execute(
        """
        UPDATE article_classifications
        SET input_text = ?,
            status = 'failed',
            is_relevant = NULL,
            processed_at = ?,
            error_message = ?
        WHERE id = ?
        """,
        (*values, existing["id"]),
    )
    return existing["id"]


def list_articles_for_classification(
    connection: sqlite3.Connection,
    identity: ClassifierIdentity,
    limit: int,
) -> list[ArticleForClassification]:
    """Return newest articles not yet successfully processed by this version."""
    if not 1 <= limit <= 100:
        raise ValueError("Classification limit must be between 1 and 100")

    rows = connection.execute(
        """
        SELECT a.id, a.title
        FROM articles AS a
        LEFT JOIN article_classifications AS c
          ON c.article_id = a.id
         AND c.provider = ?
         AND c.model = ?
         AND c.prompt_version = ?
        WHERE c.id IS NULL OR c.status = 'failed'
        ORDER BY a.published_at DESC, a.id DESC
        LIMIT ?
        """,
        (
            identity.provider,
            identity.model,
            identity.prompt_version,
            limit,
        ),
    ).fetchall()
    return [
        ArticleForClassification(id=row["id"], title=row["title"])
        for row in rows
    ]

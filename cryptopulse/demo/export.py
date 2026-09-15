import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptopulse.db.database import open_database
from cryptopulse.db.schema import create_schema


@dataclass(frozen=True)
class DemoExportSummary:
    market_rows: int
    article_rows: int
    classification_rows: int
    sentiment_rows: int
    data_through: str | None
    output_path: Path


def _copy_market_rows(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    market_hours: int,
) -> list[sqlite3.Row]:
    rows = source.execute(
        """
        WITH ranked_market AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY asset
                    ORDER BY candle_timestamp DESC, id DESC
                ) AS observation_rank
            FROM market_data
        )
        SELECT
            id, asset, symbol, candle_timestamp, open, high, low, close,
            volume, quote_volume, trade_count
        FROM ranked_market
        WHERE observation_rank <= ?
        ORDER BY asset, candle_timestamp, id
        """,
        (market_hours,),
    ).fetchall()
    destination.executemany(
        """
        INSERT INTO market_data (
            id, asset, symbol, candle_timestamp, open, high, low, close,
            volume, quote_volume, trade_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(row) for row in rows],
    )
    return rows


def _latest_production_classifications(
    source: sqlite3.Connection,
    published_after: datetime,
) -> list[sqlite3.Row]:
    return source.execute(
        """
        WITH ranked_classifications AS (
            SELECT
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.article_id, c.provider
                    ORDER BY c.processed_at DESC, c.id DESC
                ) AS version_rank
            FROM article_classifications AS c
            WHERE c.provider = 'openai'
              AND c.status = 'succeeded'
        )
        SELECT
            c.id, c.article_id, c.provider, c.model, c.prompt_version,
            c.input_text, c.status, c.is_relevant, c.processed_at,
            c.error_message
        FROM ranked_classifications AS c
        JOIN articles AS a ON a.id = c.article_id
        WHERE c.version_rank = 1
          AND a.published_at >= ?
        ORDER BY c.id
        """,
        (
            published_after.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        ),
    ).fetchall()


def _copy_classification_rows(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    published_after: datetime,
) -> tuple[int, int, int]:
    classifications = _latest_production_classifications(source, published_after)
    if not classifications:
        return 0, 0, 0

    article_ids = [row["article_id"] for row in classifications]
    article_placeholders = ", ".join("?" for _ in article_ids)
    articles = source.execute(
        f"""
        SELECT id, source, external_id, title, url, published_at, retrieved_at,
               raw_query
        FROM articles
        WHERE id IN ({article_placeholders})
        ORDER BY id
        """,
        article_ids,
    ).fetchall()
    destination.executemany(
        """
        INSERT INTO articles (
            id, source, external_id, title, url, published_at, retrieved_at,
            raw_query
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(row) for row in articles],
    )
    destination.executemany(
        """
        INSERT INTO article_classifications (
            id, article_id, provider, model, prompt_version, input_text,
            status, is_relevant, processed_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(row) for row in classifications],
    )

    classification_ids = [row["id"] for row in classifications]
    classification_placeholders = ", ".join("?" for _ in classification_ids)
    sentiment_rows = source.execute(
        f"""
        SELECT
            id, classification_id, asset, sentiment, sentiment_score,
            confidence, category, reason
        FROM sentiment_results
        WHERE classification_id IN ({classification_placeholders})
        ORDER BY id
        """,
        classification_ids,
    ).fetchall()
    destination.executemany(
        """
        INSERT INTO sentiment_results (
            id, classification_id, asset, sentiment, sentiment_score,
            confidence, category, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(row) for row in sentiment_rows],
    )
    return len(articles), len(classifications), len(sentiment_rows)


def _write_metadata(
    connection: sqlite3.Connection,
    generated_at: datetime,
    market_rows: int,
    article_rows: int,
    classification_rows: int,
    sentiment_rows: int,
    data_through: str | None,
) -> None:
    connection.execute(
        """
        CREATE TABLE demo_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    values = {
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "data_through": data_through or "",
        "market_rows": str(market_rows),
        "article_rows": str(article_rows),
        "classification_rows": str(classification_rows),
        "sentiment_rows": str(sentiment_rows),
    }
    connection.executemany(
        "INSERT INTO demo_metadata (key, value) VALUES (?, ?)",
        values.items(),
    )


def export_demo_database(
    source: sqlite3.Connection,
    output_path: Path,
    market_hours: int = 168,
    article_days: int = 7,
    *,
    force: bool = False,
    generated_at: datetime | None = None,
) -> DemoExportSummary:
    """Create an atomic portfolio snapshot without ingestion audits or secrets."""
    if not 24 <= market_hours <= 1_000:
        raise ValueError("Demo market hours must be between 24 and 1000")
    if not 1 <= article_days <= 30:
        raise ValueError("Demo article days must be between 1 and 30")
    source_file = source.execute("PRAGMA database_list").fetchone()["file"]
    if source_file and Path(source_file).resolve() == output_path.resolve():
        raise ValueError("Demo output must be different from the source database")
    if output_path.exists() and not force:
        raise FileExistsError(f"Demo database already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    exported_at = generated_at or datetime.now(UTC)
    if exported_at.tzinfo is None or exported_at.utcoffset() is None:
        raise ValueError("Demo generation time must include timezone information")
    article_cutoff = exported_at - timedelta(days=article_days)

    try:
        with open_database(temporary_path) as destination:
            create_schema(destination)
            market_rows = _copy_market_rows(source, destination, market_hours)
            article_count, classification_count, sentiment_count = (
                _copy_classification_rows(
                    source,
                    destination,
                    article_cutoff,
                )
            )
            data_through = (
                max(row["candle_timestamp"] for row in market_rows)
                if market_rows
                else None
            )
            _write_metadata(
                destination,
                exported_at,
                len(market_rows),
                article_count,
                classification_count,
                sentiment_count,
                data_through,
            )
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return DemoExportSummary(
        market_rows=len(market_rows),
        article_rows=article_count,
        classification_rows=classification_count,
        sentiment_rows=sentiment_count,
        data_through=data_through,
        output_path=output_path,
    )

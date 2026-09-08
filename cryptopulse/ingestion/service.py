import logging
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime

from cryptopulse.db.models import InsertSummary
from cryptopulse.db.repositories import (
    finish_ingestion_run,
    insert_articles,
    insert_market_candles,
    start_ingestion_run,
)
from cryptopulse.ingestion.binance import SYMBOL_TO_ASSET, BinanceClient
from cryptopulse.ingestion.gdelt import ASSET_QUERIES, GdeltClient

LOGGER = logging.getLogger(__name__)


def _combine_summaries(first: InsertSummary, second: InsertSummary) -> InsertSummary:
    return InsertSummary(
        received=first.received + second.received,
        inserted=first.inserted + second.inserted,
        skipped=first.skipped + second.skipped,
    )


def _record_failed_run(
    connection: sqlite3.Connection,
    run_id: int,
    summary: InsertSummary,
    error: Exception,
) -> None:
    connection.rollback()
    finish_ingestion_run(
        connection=connection,
        run_id=run_id,
        completed_at=datetime.now(UTC),
        status="failed",
        summary=summary,
        error_message=f"{type(error).__name__}: {error}"[:1_000],
    )
    connection.commit()


def ingest_market(
    connection: sqlite3.Connection,
    client: BinanceClient,
    symbols: Iterable[str] = tuple(SYMBOL_TO_ASSET),
) -> InsertSummary:
    """Fetch and store hourly market candles for supported symbols."""
    run_id = start_ingestion_run(connection, "binance", datetime.now(UTC))
    connection.commit()
    total = InsertSummary(received=0, inserted=0, skipped=0)

    try:
        for symbol in symbols:
            LOGGER.info("Starting Binance ingestion for %s", symbol)
            candles = client.fetch_hourly_candles(symbol)
            summary = insert_market_candles(connection, candles)
            connection.commit()
            total = _combine_summaries(total, summary)
            LOGGER.info(
                "Binance %s: received=%d inserted=%d skipped=%d",
                symbol,
                summary.received,
                summary.inserted,
                summary.skipped,
            )

        finish_ingestion_run(
            connection=connection,
            run_id=run_id,
            completed_at=datetime.now(UTC),
            status="succeeded",
            summary=total,
        )
        connection.commit()
        return total
    except Exception as error:
        _record_failed_run(connection, run_id, total, error)
        LOGGER.exception("Binance ingestion failed")
        raise


def ingest_news(
    connection: sqlite3.Connection,
    client: GdeltClient,
    assets: Iterable[str] = tuple(ASSET_QUERIES),
) -> InsertSummary:
    """Fetch and store recent news for supported assets."""
    run_id = start_ingestion_run(connection, "gdelt", datetime.now(UTC))
    connection.commit()
    total = InsertSummary(received=0, inserted=0, skipped=0)

    try:
        for asset in assets:
            LOGGER.info("Starting GDELT ingestion for %s", asset)
            articles = client.fetch_recent_articles(asset)
            summary = insert_articles(connection, articles)
            connection.commit()
            total = _combine_summaries(total, summary)
            LOGGER.info(
                "GDELT %s: received=%d inserted=%d skipped=%d",
                asset,
                summary.received,
                summary.inserted,
                summary.skipped,
            )

        finish_ingestion_run(
            connection=connection,
            run_id=run_id,
            completed_at=datetime.now(UTC),
            status="succeeded",
            summary=total,
        )
        connection.commit()
        return total
    except Exception as error:
        _record_failed_run(connection, run_id, total, error)
        LOGGER.exception("GDELT ingestion failed")
        raise

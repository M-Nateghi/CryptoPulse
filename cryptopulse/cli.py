import argparse
import logging
import sqlite3
from collections.abc import Sequence

import httpx

from cryptopulse.config import Settings
from cryptopulse.db.database import open_database
from cryptopulse.db.schema import create_schema
from cryptopulse.ingestion.binance import BinanceClient
from cryptopulse.ingestion.gdelt import GdeltClient
from cryptopulse.ingestion.service import ingest_market, ingest_news
from cryptopulse.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)
USER_AGENT = "CryptoPulse/0.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptopulse",
        description="Ingest cryptocurrency news and market data.",
    )
    parser.add_argument(
        "command",
        choices=("ingest-market", "ingest-news", "ingest-all"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)

    try:
        with open_database(settings.database_path) as connection:
            create_schema(connection)

            if args.command in {"ingest-market", "ingest-all"}:
                with httpx.Client(
                    base_url=settings.binance_base_url,
                    timeout=settings.request_timeout_seconds,
                    headers={"User-Agent": USER_AGENT},
                ) as http_client:
                    ingest_market(connection, BinanceClient(http_client))

            if args.command in {"ingest-news", "ingest-all"}:
                with httpx.Client(
                    base_url=settings.gdelt_base_url,
                    timeout=settings.request_timeout_seconds,
                    headers={"User-Agent": USER_AGENT},
                ) as http_client:
                    ingest_news(connection, GdeltClient(http_client))
    except (httpx.HTTPError, sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.error("Command %s failed: %s", args.command, error)
        return 1

    LOGGER.info("Command %s completed successfully", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import logging
import sqlite3
from collections.abc import Sequence

import httpx
from openai import OpenAI

from cryptopulse.config import Settings
from cryptopulse.db.database import open_database
from cryptopulse.db.schema import create_schema
from cryptopulse.ingestion.binance import BinanceClient
from cryptopulse.ingestion.gdelt import GdeltClient
from cryptopulse.ingestion.service import ingest_market, ingest_news
from cryptopulse.logging_config import configure_logging
from cryptopulse.sentiment.fake_provider import FakeSentimentClassifier
from cryptopulse.sentiment.openai_provider import OpenAISentimentClassifier
from cryptopulse.sentiment.service import classify_articles

LOGGER = logging.getLogger(__name__)
USER_AGENT = "CryptoPulse/0.1"


def _classification_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be an integer") from error
    if not 1 <= limit <= 100:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100")
    return limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptopulse",
        description="Ingest cryptocurrency news and market data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest-market", help="Ingest Binance market candles.")
    commands.add_parser("ingest-news", help="Ingest GDELT news articles.")
    commands.add_parser("ingest-all", help="Ingest market candles and news.")
    classify_parser = commands.add_parser(
        "classify-news",
        help="Classify a bounded batch of stored news articles.",
    )
    classify_parser.add_argument(
        "--provider",
        choices=("openai", "fake"),
        default="openai",
        help="Classifier provider; fake is for offline development only.",
    )
    classify_parser.add_argument(
        "--limit",
        type=_classification_limit,
        default=10,
        help="Maximum articles to process (1-100, default: 10).",
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
                    timeout=settings.gdelt_request_timeout_seconds,
                    headers={"User-Agent": USER_AGENT},
                ) as http_client:
                    ingest_news(connection, GdeltClient(http_client))

            if args.command == "classify-news":
                if args.provider == "fake":
                    summary = classify_articles(
                        connection,
                        FakeSentimentClassifier(),
                        limit=args.limit,
                    )
                else:
                    if settings.openai_api_key is None:
                        raise ValueError(
                            "CRYPTOPULSE_OPENAI_API_KEY is required for OpenAI"
                        )
                    with OpenAI(
                        api_key=settings.openai_api_key.get_secret_value(),
                        timeout=settings.openai_timeout_seconds,
                        max_retries=2,
                    ) as openai_client:
                        summary = classify_articles(
                            connection,
                            OpenAISentimentClassifier(
                                openai_client,
                                model=settings.openai_model,
                            ),
                            limit=args.limit,
                        )
                LOGGER.info(
                    "Classification batch: selected=%d succeeded=%d failed=%d "
                    "relevant=%d irrelevant=%d asset_results=%d",
                    summary.selected,
                    summary.succeeded,
                    summary.failed,
                    summary.relevant,
                    summary.irrelevant,
                    summary.asset_results,
                )
    except (httpx.HTTPError, sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.error("Command %s failed: %s", args.command, error)
        return 1

    LOGGER.info("Command %s completed successfully", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

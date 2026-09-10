import argparse
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import httpx
from openai import OpenAI

from cryptopulse.config import Settings
from cryptopulse.db.database import open_database
from cryptopulse.db.schema import create_schema
from cryptopulse.evaluation.labels import validate_labeling_csv
from cryptopulse.evaluation.sampling import create_labeling_template
from cryptopulse.ingestion.binance import BinanceClient
from cryptopulse.ingestion.gdelt import ASSET_QUERIES, GdeltClient
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


def _evaluation_size(value: str) -> int:
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must be an integer") from error
    if not 10 <= size <= 500:
        raise argparse.ArgumentTypeError("size must be between 10 and 500")
    return size


def _gdelt_max_records(value: str) -> int:
    try:
        max_records = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max records must be an integer") from error
    if not 1 <= max_records <= 250:
        raise argparse.ArgumentTypeError("max records must be between 1 and 250")
    return max_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptopulse",
        description="Ingest cryptocurrency news and market data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest-market", help="Ingest Binance market candles.")
    news_parser = commands.add_parser("ingest-news", help="Ingest GDELT news articles.")
    news_parser.add_argument(
        "--assets",
        nargs="+",
        choices=tuple(ASSET_QUERIES),
        default=tuple(ASSET_QUERIES),
        help="Assets to include in the news query (default: all).",
    )
    news_parser.add_argument(
        "--max-records",
        type=_gdelt_max_records,
        default=None,
        help="Maximum articles requested from GDELT (1-250).",
    )
    news_parser.add_argument(
        "--timespan",
        choices=("1d", "3d", "7d"),
        default="1d",
        help="Recent GDELT search window (default: 1d).",
    )
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
    evaluation_parser = commands.add_parser(
        "prepare-evaluation",
        help="Create a balanced human-labeling CSV from stored articles.",
    )
    evaluation_parser.add_argument(
        "--size",
        type=_evaluation_size,
        default=100,
        help="Number of headlines to sample (10-500, default: 100).",
    )
    evaluation_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    evaluation_parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/labels_v1.csv"),
        help="Destination CSV (default: evaluation/labels_v1.csv).",
    )
    validation_parser = commands.add_parser(
        "validate-evaluation",
        help="Validate a completed human-labeling CSV.",
    )
    validation_parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/labels_v1.csv"),
        help="Human-labeling CSV (default: evaluation/labels_v1.csv).",
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
                    gdelt_client = GdeltClient(http_client)
                    if args.command == "ingest-news":
                        ingest_news(
                            connection,
                            gdelt_client,
                            assets=args.assets,
                            max_records=args.max_records,
                            timespan=args.timespan,
                        )
                    else:
                        ingest_news(connection, gdelt_client)

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

            if args.command == "prepare-evaluation":
                summary = create_labeling_template(
                    connection=connection,
                    output_path=args.output,
                    sample_size=args.size,
                    seed=args.seed,
                )
                LOGGER.info(
                    "Evaluation template: selected=%d BTC=%d ETH=%d SOL=%d "
                    "BNB=%d multi_asset=%d no_candidate=%d output=%s",
                    summary.selected,
                    summary.candidate_counts["BTC"],
                    summary.candidate_counts["ETH"],
                    summary.candidate_counts["SOL"],
                    summary.candidate_counts["BNB"],
                    summary.multi_asset,
                    summary.no_candidate,
                    args.output,
                )

            if args.command == "validate-evaluation":
                summary = validate_labeling_csv(args.input)
                LOGGER.info(
                    "Validated human labels: articles=%d relevant=%d "
                    "irrelevant=%d asset_labels=%d",
                    summary.total_articles,
                    summary.relevant,
                    summary.irrelevant,
                    summary.asset_labels,
                )
    except (httpx.HTTPError, sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.error("Command %s failed: %s", args.command, error)
        return 1

    LOGGER.info("Command %s completed successfully", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

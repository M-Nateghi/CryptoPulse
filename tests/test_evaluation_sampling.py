import csv
from datetime import UTC, datetime, timedelta

import pytest

from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article
from cryptopulse.db.repositories import insert_articles
from cryptopulse.db.schema import create_schema
from cryptopulse.evaluation.sampling import (
    LABEL_FIELDS,
    create_labeling_template,
    select_evaluation_sample,
)
from cryptopulse.sentiment.models import CryptoAsset


def _add_evaluation_articles(connection):
    titles = [
        "Bitcoin adoption increases",
        "BTC market update",
        "Ethereum upgrade announced",
        "ETH market update",
        "Solana network activity rises",
        "SOL market update",
        "BNB Chain activity increases",
        "Binance Coin market update",
        "Bitcoin and Ethereum prices diverge",
        "Global markets await inflation report",
        "Technology companies publish earnings",
        "Investors monitor interest rates",
    ]
    retrieved_at = datetime(2026, 9, 10, 10, tzinfo=UTC)
    insert_articles(
        connection,
        [
            Article(
                source="gdelt",
                external_id=None,
                title=title,
                url=f"https://example.com/article-{index}",
                published_at=retrieved_at - timedelta(hours=index),
                retrieved_at=retrieved_at,
                raw_query="evaluation-test",
            )
            for index, title in enumerate(titles, start=1)
        ],
    )


def test_evaluation_sample_is_reproducible_and_covers_every_asset(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        _add_evaluation_articles(connection)

        first = select_evaluation_sample(connection, sample_size=10, seed=7)
        second = select_evaluation_sample(connection, sample_size=10, seed=7)

    assert [article.article_id for article in first] == [
        article.article_id for article in second
    ]
    assert len(first) == 10
    for asset in CryptoAsset:
        assert sum(asset in article.candidate_assets for article in first) >= 1
    assert sum(not article.candidate_assets for article in first) >= 2


def test_evaluation_sample_rejects_insufficient_asset_coverage(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        insert_articles(
            connection,
            [
                Article(
                    source="gdelt",
                    external_id=None,
                    title=f"General market headline {index}",
                    url=f"https://example.com/general-{index}",
                    published_at=datetime(2026, 9, 10, index, tzinfo=UTC),
                    retrieved_at=datetime(2026, 9, 10, 12, tzinfo=UTC),
                    raw_query="test",
                )
                for index in range(10)
            ],
        )

        with pytest.raises(ValueError, match="insufficient coverage"):
            select_evaluation_sample(connection, sample_size=10)


def test_labeling_template_has_expected_columns_and_is_not_overwritten(tmp_path):
    output_path = tmp_path / "labels.csv"
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        _add_evaluation_articles(connection)

        summary = create_labeling_template(
            connection,
            output_path=output_path,
            sample_size=10,
            seed=7,
        )
        with pytest.raises(FileExistsError):
            create_labeling_template(
                connection,
                output_path=output_path,
                sample_size=10,
                seed=7,
            )

    with output_path.open(encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert summary.selected == 10
    assert tuple(rows[0]) == LABEL_FIELDS
    assert len(rows) == 10
    assert all(row["human_relevance"] == "" for row in rows)
    assert all(row["human_notes"] == "" for row in rows)

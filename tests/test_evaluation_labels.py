import csv

import pytest

from cryptopulse.evaluation.labels import (
    LabelValidationError,
    LabelValidationSummary,
    validate_labeling_csv,
)
from cryptopulse.evaluation.sampling import LABEL_FIELDS


def _row(**overrides):
    values = dict.fromkeys(LABEL_FIELDS, "")
    values.update(
        {
            "sample_id": "1",
            "article_id": "101",
            "title": "Bitcoin adoption increases",
            "published_at": "2026-09-10T10:00:00Z",
            "human_relevance": "relevant",
            "btc_sentiment": "bullish",
            "btc_category": "institutional_adoption",
        }
    )
    values.update(overrides)
    return values


def _write_csv(path, rows, fieldnames=LABEL_FIELDS):
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_valid_labeling_csv_returns_summary(tmp_path):
    input_path = tmp_path / "labels.csv"
    _write_csv(
        input_path,
        [
            _row(),
            _row(
                sample_id="2",
                article_id="102",
                title="General market update",
                human_relevance="irrelevant",
                btc_sentiment="",
                btc_category="",
            ),
            _row(
                sample_id="3",
                article_id="103",
                title="Ethereum and Solana update",
                btc_sentiment="",
                btc_category="",
                eth_sentiment="neutral",
                eth_category="technology",
                sol_sentiment="bearish",
                sol_category="market_movement",
            ),
        ],
    )

    summary = validate_labeling_csv(input_path)

    assert summary == LabelValidationSummary(
        total_articles=3,
        relevant=2,
        irrelevant=1,
        asset_labels=3,
        asset_counts={"BTC": 1, "ETH": 1, "SOL": 1, "BNB": 0},
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"human_relevance": ""}, "must be relevant or irrelevant"),
        (
            {
                "human_relevance": "relevant",
                "btc_sentiment": "",
                "btc_category": "",
            },
            "relevant row needs at least one asset label",
        ),
        (
            {"human_relevance": "irrelevant"},
            "irrelevant row cannot contain asset labels",
        ),
        ({"btc_sentiment": "positive"}, "invalid BTC sentiment"),
        ({"btc_category": "partnership"}, "invalid BTC category"),
        ({"btc_category": ""}, "both be filled or both be blank"),
    ],
)
def test_invalid_human_labels_are_rejected(tmp_path, overrides, message):
    input_path = tmp_path / "labels.csv"
    _write_csv(input_path, [_row(**overrides)])

    with pytest.raises(LabelValidationError, match=message):
        validate_labeling_csv(input_path)


def test_changed_csv_columns_are_rejected(tmp_path):
    input_path = tmp_path / "labels.csv"
    _write_csv(input_path, [{"sample_id": "1"}], fieldnames=("sample_id",))

    with pytest.raises(LabelValidationError, match="columns do not match"):
        validate_labeling_csv(input_path)

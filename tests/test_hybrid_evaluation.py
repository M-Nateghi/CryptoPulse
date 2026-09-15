import csv
from datetime import UTC, datetime

import pytest

from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article
from cryptopulse.db.repositories import insert_articles
from cryptopulse.db.schema import create_schema
from cryptopulse.evaluation.assisted_labels import (
    ASSISTED_LABEL_FIELDS,
    AssistedLabelItem,
    generate_assisted_labels,
)
from cryptopulse.evaluation.baselines import (
    FinBertPredictor,
    SentimentPrediction,
    vader_label,
)
from cryptopulse.evaluation.labels import (
    LabelValidationError,
    validate_relevance_csv,
)
from cryptopulse.evaluation.openai_predictions import RelevancePrediction
from cryptopulse.evaluation.runner import classification_metrics, run_hybrid_evaluation
from cryptopulse.evaluation.sampling import LABEL_FIELDS
from cryptopulse.sentiment.models import (
    AssetSentiment,
    CryptoAsset,
    NewsCategory,
    SentimentLabel,
)
from cryptopulse.sentiment.provider import ClassifierIdentity


def _write_human_labels(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            complete = dict.fromkeys(LABEL_FIELDS, "")
            complete.update(row)
            writer.writerow(complete)


class FakeAssistedProvider:
    identity = ClassifierIdentity(
        provider="openai",
        model="test-model",
        prompt_version="assisted-v1",
    )

    def label(self, rows):
        return [
            AssistedLabelItem(
                sample_id=row.sample_id,
                asset_sentiments=[
                    AssetSentiment(
                        asset=CryptoAsset.BTC,
                        sentiment=SentimentLabel.BULLISH,
                        sentiment_score=0.8,
                        confidence=0.9,
                        category=NewsCategory.INSTITUTIONAL_ADOPTION,
                        reason="Headline reports increased holdings.",
                    )
                ],
            )
            for row in rows
        ]


class FakeRelevancePredictor:
    identity = ClassifierIdentity(
        provider="openai",
        model="test-model",
        prompt_version="evaluation-v1",
    )

    def predict(self, rows):
        return [
            RelevancePrediction(
                sample_id=row.sample_id,
                is_relevant=row.sample_id == 1,
                asset_sentiments=(
                    [
                        AssetSentiment(
                            asset=CryptoAsset.BTC,
                            sentiment=SentimentLabel.BULLISH,
                            sentiment_score=0.8,
                            confidence=0.9,
                            category=NewsCategory.INSTITUTIONAL_ADOPTION,
                            reason="Headline reports increased holdings.",
                        )
                    ]
                    if row.sample_id == 1
                    else []
                ),
            )
            for row in rows
        ]


class FakeBaselinePredictor:
    name = "test-baseline"

    def predict(self, headlines):
        return [
            SentimentPrediction(label=SentimentLabel.BULLISH, score=0.75)
            for _ in headlines
        ]


def test_validate_relevance_accepts_human_relevance_only(tmp_path):
    labels_path = tmp_path / "labels.csv"
    _write_human_labels(
        labels_path,
        [
            {
                "sample_id": "1",
                "article_id": "10",
                "title": "Bitcoin holdings increase",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "relevant",
            },
            {
                "sample_id": "2",
                "article_id": "11",
                "title": "Unrelated headline",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "irrelevant",
            },
        ],
    )

    summary = validate_relevance_csv(labels_path)

    assert summary.total_articles == 2
    assert summary.relevant == 1
    assert summary.irrelevant == 1


def test_validate_relevance_rejects_machine_fields_in_human_file(tmp_path):
    labels_path = tmp_path / "labels.csv"
    _write_human_labels(
        labels_path,
        [
            {
                "sample_id": "1",
                "article_id": "10",
                "title": "Bitcoin holdings increase",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "relevant",
                "btc_sentiment": "bullish",
                "btc_category": "institutional_adoption",
            }
        ],
    )

    with pytest.raises(LabelValidationError, match="must remain blank"):
        validate_relevance_csv(labels_path)


def test_generate_assisted_labels_writes_provenance_and_database(tmp_path):
    database_path = tmp_path / "source.db"
    labels_path = tmp_path / "labels.csv"
    output_path = tmp_path / "assisted.csv"
    _write_human_labels(
        labels_path,
        [
            {
                "sample_id": "1",
                "article_id": "1",
                "title": "Bitcoin holdings increase",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "relevant",
            },
            {
                "sample_id": "2",
                "article_id": "2",
                "title": "Unrelated headline",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "irrelevant",
            },
        ],
    )
    with open_database(database_path) as connection:
        create_schema(connection)
        insert_articles(
            connection,
            [
                Article(
                    source="test",
                    external_id=None,
                    title=title,
                    url=f"https://example.com/{index}",
                    published_at=datetime(2026, 9, 15, tzinfo=UTC),
                    retrieved_at=datetime(2026, 9, 15, tzinfo=UTC),
                    raw_query="test",
                )
                for index, title in enumerate(
                    ("Bitcoin holdings increase", "Unrelated headline"), start=1
                )
            ],
        )
        summary = generate_assisted_labels(
            connection,
            FakeAssistedProvider(),
            labels_path,
            output_path,
            generated_at=datetime(2026, 9, 15, tzinfo=UTC),
        )
        stored = connection.execute(
            "SELECT provider, model, prompt_version FROM article_classifications"
        ).fetchone()

    with output_path.open(encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert summary.articles == 1
    assert summary.asset_labels == 1
    assert rows[0]["assisted_sentiment"] == "bullish"
    assert rows[0]["provider"] == "openai"
    assert tuple(rows[0]) == ASSISTED_LABEL_FIELDS
    assert tuple(stored) == ("openai", "test-model", "assisted-v1")


def test_baseline_label_mapping_and_finbert_adapter():
    assert vader_label(0.05) is SentimentLabel.BULLISH
    assert vader_label(-0.05) is SentimentLabel.BEARISH
    assert vader_label(0.0) is SentimentLabel.NEUTRAL

    classifier = lambda headlines, **_: [
        {"label": "positive", "score": 0.91} for _ in headlines
    ]
    result = FinBertPredictor(classifier=classifier).predict(["Headline"])
    assert result == [SentimentPrediction(SentimentLabel.BULLISH, 0.91)]


def test_metrics_and_hybrid_report(tmp_path):
    metrics, confusion = classification_metrics(
        ["relevant", "relevant", "irrelevant"],
        ["relevant", "irrelevant", "irrelevant"],
        ["relevant", "irrelevant"],
    )
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert confusion[("relevant", "irrelevant")] == 1

    labels_path = tmp_path / "labels.csv"
    assisted_path = tmp_path / "assisted.csv"
    output_directory = tmp_path / "results"
    _write_human_labels(
        labels_path,
        [
            {
                "sample_id": "1",
                "article_id": "10",
                "title": "Bitcoin holdings increase",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "relevant",
            },
            {
                "sample_id": "2",
                "article_id": "11",
                "title": "Unrelated headline",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "irrelevant",
            },
        ],
    )
    with assisted_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ASSISTED_LABEL_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "1",
                "article_id": "10",
                "title": "Bitcoin holdings increase",
                "published_at": "2026-09-15T00:00:00Z",
                "human_relevance": "relevant",
                "asset": "BTC",
                "assisted_sentiment": "bullish",
                "assisted_category": "institutional_adoption",
                "sentiment_score": "0.8",
                "confidence": "0.9",
                "reason": "Headline reports increased holdings.",
                "provider": "openai",
                "model": "test-model",
                "prompt_version": "assisted-v1",
                "generated_at": "2026-09-15T00:00:00Z",
            }
        )

    summary = run_hybrid_evaluation(
        labels_path,
        assisted_path,
        output_directory,
        FakeRelevancePredictor(),
        [FakeBaselinePredictor()],
    )

    assert summary.human_articles == 2
    assert summary.metrics == 2
    assert (output_directory / "model_metrics.csv").exists()
    assert (output_directory / "openai_relevance_predictions.csv").exists()

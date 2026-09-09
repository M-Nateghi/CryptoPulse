from datetime import UTC, datetime

from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article
from cryptopulse.db.repositories import insert_articles
from cryptopulse.db.schema import create_schema
from cryptopulse.sentiment import CryptoAsset, FakeSentimentClassifier
from cryptopulse.sentiment.provider import ClassificationProviderError
from cryptopulse.sentiment.service import ClassificationBatchSummary, classify_articles


def _add_articles(connection):
    insert_articles(
        connection,
        [
            Article(
                source="gdelt",
                external_id=None,
                title="Bitcoin market update",
                url="https://example.com/bitcoin",
                published_at=datetime(2026, 9, 9, 11, tzinfo=UTC),
                retrieved_at=datetime(2026, 9, 9, 12, tzinfo=UTC),
                raw_query="Bitcoin",
            ),
            Article(
                source="gdelt",
                external_id=None,
                title="General streaming market update",
                url="https://example.com/streaming",
                published_at=datetime(2026, 9, 9, 10, tzinfo=UTC),
                retrieved_at=datetime(2026, 9, 9, 12, tzinfo=UTC),
                raw_query="BTC",
            ),
        ],
    )


def test_batch_classifies_and_stores_each_selected_article(tmp_path):
    classifier = FakeSentimentClassifier()
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        _add_articles(connection)

        summary = classify_articles(connection, classifier, limit=10)
        parents = connection.execute(
            "SELECT status, is_relevant FROM article_classifications ORDER BY id"
        ).fetchall()
        asset_rows = connection.execute(
            "SELECT asset, sentiment FROM sentiment_results"
        ).fetchall()

    assert summary == ClassificationBatchSummary(
        selected=2,
        succeeded=2,
        failed=0,
        relevant=1,
        irrelevant=1,
        asset_results=1,
    )
    assert [(row["status"], row["is_relevant"]) for row in parents] == [
        ("succeeded", 1),
        ("succeeded", 0),
    ]
    assert [(row["asset"], row["sentiment"]) for row in asset_rows] == [
        ("BTC", "neutral")
    ]
    assert len(classifier.calls) == 2


def test_batch_does_not_reprocess_successful_version(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        _add_articles(connection)
        classify_articles(connection, FakeSentimentClassifier(), limit=10)
        second_classifier = FakeSentimentClassifier()

        summary = classify_articles(connection, second_classifier, limit=10)

    assert summary.selected == 0
    assert second_classifier.calls == ()


def test_batch_records_failure_and_continues(tmp_path):
    classifier = FakeSentimentClassifier(
        {"Bitcoin market update": ClassificationProviderError("simulated failure")}
    )
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        _add_articles(connection)

        summary = classify_articles(connection, classifier, limit=10)
        rows = connection.execute(
            "SELECT status, error_message FROM article_classifications ORDER BY id"
        ).fetchall()

    assert summary == ClassificationBatchSummary(
        selected=2,
        succeeded=1,
        failed=1,
        relevant=0,
        irrelevant=1,
        asset_results=0,
    )
    assert [row["status"] for row in rows] == ["failed", "succeeded"]
    assert "simulated failure" in rows[0]["error_message"]


def test_failed_article_is_selected_again_on_next_run(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        _add_articles(connection)
        classify_articles(
            connection,
            FakeSentimentClassifier(
                {
                    "Bitcoin market update": ClassificationProviderError(
                        "temporary failure"
                    )
                }
            ),
            limit=10,
        )
        retry_classifier = FakeSentimentClassifier()

        summary = classify_articles(connection, retry_classifier, limit=10)
        bitcoin_status = connection.execute(
            """
            SELECT c.status
            FROM article_classifications AS c
            JOIN articles AS a ON a.id = c.article_id
            WHERE a.title = 'Bitcoin market update'
            """
        ).fetchone()["status"]

    assert summary.selected == 1
    assert summary.succeeded == 1
    assert retry_classifier.calls[0].candidate_assets == (CryptoAsset.BTC,)
    assert bitcoin_status == "succeeded"

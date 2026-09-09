import sqlite3
from datetime import UTC, datetime, timedelta

from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article
from cryptopulse.db.repositories import (
    insert_articles,
    save_classification_failure,
    save_classification_success,
)
from cryptopulse.db.schema import create_schema
from cryptopulse.sentiment import (
    ArticleClassification,
    AssetSentiment,
    ClassifierIdentity,
)

PROCESSED_AT = datetime(2026, 9, 9, 12, tzinfo=UTC)


def _identity(**overrides):
    values = {
        "provider": "fake",
        "model": "deterministic-test-classifier",
        "prompt_version": "v1",
    }
    values.update(overrides)
    return ClassifierIdentity(**values)


def _classification(**overrides):
    values = {
        "is_relevant": True,
        "asset_sentiments": [
            AssetSentiment(
                asset="BTC",
                sentiment="bullish",
                sentiment_score=0.8,
                confidence=0.9,
                category="etf_flows",
                reason="ETF demand is positive for Bitcoin.",
            )
        ],
    }
    values.update(overrides)
    return ArticleClassification(**values)


def _insert_article(connection):
    insert_articles(
        connection,
        [
            Article(
                source="gdelt",
                external_id=None,
                title="Bitcoin ETF demand rises",
                url="https://example.com/bitcoin-etf",
                published_at=datetime(2026, 9, 9, 10, tzinfo=UTC),
                retrieved_at=datetime(2026, 9, 9, 11, tzinfo=UTC),
                raw_query="Bitcoin",
            )
        ],
    )
    return connection.execute("SELECT id FROM articles").fetchone()["id"]


def test_save_success_stores_provenance_and_asset_results(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)

        classification_id = save_classification_success(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            _classification(),
            PROCESSED_AT,
        )
        parent = connection.execute(
            "SELECT * FROM article_classifications WHERE id = ?",
            (classification_id,),
        ).fetchone()
        result = connection.execute(
            "SELECT * FROM sentiment_results WHERE classification_id = ?",
            (classification_id,),
        ).fetchone()

    assert parent["status"] == "succeeded"
    assert parent["is_relevant"] == 1
    assert parent["provider"] == "fake"
    assert parent["model"] == "deterministic-test-classifier"
    assert parent["prompt_version"] == "v1"
    assert parent["processed_at"] == "2026-09-09T12:00:00Z"
    assert parent["error_message"] is None
    assert result["asset"] == "BTC"
    assert result["sentiment"] == "bullish"
    assert result["sentiment_score"] == 0.8
    assert result["confidence"] == 0.9
    assert result["category"] == "etf_flows"


def test_save_success_stores_irrelevant_article_without_asset_rows(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        classification_id = save_classification_success(
            connection,
            article_id,
            _identity(),
            "General market update",
            ArticleClassification(is_relevant=False, asset_sentiments=[]),
            PROCESSED_AT,
        )
        parent = connection.execute(
            "SELECT is_relevant FROM article_classifications WHERE id = ?",
            (classification_id,),
        ).fetchone()
        result_count = connection.execute(
            "SELECT COUNT(*) FROM sentiment_results"
        ).fetchone()[0]

    assert parent["is_relevant"] == 0
    assert result_count == 0


def test_save_failure_records_error_without_asset_rows(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        classification_id = save_classification_failure(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            "  RuntimeError: provider unavailable  ",
            PROCESSED_AT,
        )
        parent = connection.execute(
            "SELECT * FROM article_classifications WHERE id = ?",
            (classification_id,),
        ).fetchone()

    assert parent["status"] == "failed"
    assert parent["is_relevant"] is None
    assert parent["error_message"] == "RuntimeError: provider unavailable"


def test_successful_retry_replaces_a_failed_attempt(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        failed_id = save_classification_failure(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            "temporary failure",
            PROCESSED_AT,
        )
        succeeded_id = save_classification_success(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            _classification(),
            PROCESSED_AT + timedelta(minutes=1),
        )
        parent = connection.execute(
            "SELECT status, error_message FROM article_classifications"
        ).fetchone()
        result_count = connection.execute(
            "SELECT COUNT(*) FROM sentiment_results"
        ).fetchone()[0]

    assert succeeded_id == failed_id
    assert parent["status"] == "succeeded"
    assert parent["error_message"] is None
    assert result_count == 1


def test_existing_success_is_not_silently_overwritten(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        first_id = save_classification_success(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            _classification(),
            PROCESSED_AT,
        )
        second_id = save_classification_success(
            connection,
            article_id,
            _identity(),
            "Changed input",
            _classification(
                asset_sentiments=[
                    AssetSentiment(
                        asset="BTC",
                        sentiment="bearish",
                        sentiment_score=-0.9,
                        confidence=0.8,
                        category="regulation",
                        reason="This replacement must not be stored.",
                    )
                ]
            ),
            PROCESSED_AT + timedelta(minutes=1),
        )
        stored = connection.execute(
            "SELECT sentiment, sentiment_score FROM sentiment_results"
        ).fetchone()

    assert second_id == first_id
    assert stored["sentiment"] == "bullish"
    assert stored["sentiment_score"] == 0.8


def test_failure_does_not_replace_an_existing_success(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        success_id = save_classification_success(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            _classification(),
            PROCESSED_AT,
        )
        failure_id = save_classification_failure(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            "late failure",
            PROCESSED_AT + timedelta(minutes=1),
        )
        parent = connection.execute(
            "SELECT status, error_message FROM article_classifications"
        ).fetchone()

    assert failure_id == success_id
    assert parent["status"] == "succeeded"
    assert parent["error_message"] is None


def test_model_or_prompt_change_creates_a_new_version(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        for identity in (
            _identity(),
            _identity(model="different-model"),
            _identity(prompt_version="v2"),
        ):
            save_classification_success(
                connection,
                article_id,
                identity,
                "Bitcoin ETF demand rises",
                _classification(),
                PROCESSED_AT,
            )
        classification_count = connection.execute(
            "SELECT COUNT(*) FROM article_classifications"
        ).fetchone()[0]

    assert classification_count == 3


def test_classification_rows_are_deleted_with_the_article(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        save_classification_success(
            connection,
            article_id,
            _identity(),
            "Bitcoin ETF demand rises",
            _classification(),
            PROCESSED_AT,
        )
        connection.execute("DELETE FROM articles WHERE id = ?", (article_id,))
        parent_count = connection.execute(
            "SELECT COUNT(*) FROM article_classifications"
        ).fetchone()[0]
        result_count = connection.execute(
            "SELECT COUNT(*) FROM sentiment_results"
        ).fetchone()[0]

    assert parent_count == 0
    assert result_count == 0


def test_database_constraints_reject_invalid_sentiment_result(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        article_id = _insert_article(connection)
        classification_id = save_classification_success(
            connection,
            article_id,
            _identity(),
            "General market update",
            ArticleClassification(is_relevant=False, asset_sentiments=[]),
            PROCESSED_AT,
        )
        try:
            connection.execute(
                """
                INSERT INTO sentiment_results (
                    classification_id, asset, sentiment, sentiment_score,
                    confidence, category, reason
                ) VALUES (?, 'DOGE', 'positive', 2, 3, 'unknown', '')
                """,
                (classification_id,),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Invalid sentiment result was accepted")

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from cryptopulse.db.repositories import (
    list_articles_for_classification,
    save_classification_failure,
    save_classification_success,
)
from cryptopulse.sentiment.cleaning import prepare_article_text
from cryptopulse.sentiment.models import ArticleClassification
from cryptopulse.sentiment.provider import (
    ClassificationProviderError,
    SentimentClassifier,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationBatchSummary:
    selected: int
    succeeded: int
    failed: int
    relevant: int
    irrelevant: int
    asset_results: int


def classify_articles(
    connection: sqlite3.Connection,
    classifier: SentimentClassifier,
    limit: int = 10,
) -> ClassificationBatchSummary:
    """Classify a bounded batch and persist each article outcome independently."""
    articles = list_articles_for_classification(
        connection,
        classifier.identity,
        limit,
    )
    succeeded = 0
    failed = 0
    relevant = 0
    irrelevant = 0
    asset_results = 0

    for article in articles:
        input_text = article.title
        try:
            prepared = prepare_article_text(article.title)
            input_text = prepared.text
            classification = classifier.classify(prepared)
            if not isinstance(classification, ArticleClassification):
                raise TypeError("Classifier returned an invalid result type")
        except (ClassificationProviderError, TypeError, ValueError) as error:
            save_classification_failure(
                connection=connection,
                article_id=article.id,
                identity=classifier.identity,
                input_text=input_text,
                error_message=f"{type(error).__name__}: {error}",
                processed_at=datetime.now(UTC),
            )
            connection.commit()
            failed += 1
            LOGGER.warning(
                "Classification failed for article %d: %s",
                article.id,
                error,
            )
            continue

        save_classification_success(
            connection=connection,
            article_id=article.id,
            identity=classifier.identity,
            input_text=prepared.text,
            classification=classification,
            processed_at=datetime.now(UTC),
        )
        connection.commit()
        succeeded += 1
        relevant += int(classification.is_relevant)
        irrelevant += int(not classification.is_relevant)
        asset_results += len(classification.asset_sentiments)

    return ClassificationBatchSummary(
        selected=len(articles),
        succeeded=succeeded,
        failed=failed,
        relevant=relevant,
        irrelevant=irrelevant,
        asset_results=asset_results,
    )

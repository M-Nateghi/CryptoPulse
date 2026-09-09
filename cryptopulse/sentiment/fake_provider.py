from collections.abc import Mapping

from cryptopulse.sentiment.cleaning import PreparedArticle
from cryptopulse.sentiment.models import (
    ArticleClassification,
    AssetSentiment,
    NewsCategory,
    SentimentLabel,
)
from cryptopulse.sentiment.provider import (
    ClassificationProviderError,
    ClassifierIdentity,
)

FakeOutcome = ArticleClassification | ClassificationProviderError


class FakeSentimentClassifier:
    """Deterministic classifier for offline development and tests."""

    def __init__(self, outcomes: Mapping[str, FakeOutcome] | None = None) -> None:
        self._identity = ClassifierIdentity(
            provider="fake",
            model="deterministic-test-classifier",
            prompt_version="v1",
        )
        self._outcomes = dict(outcomes or {})
        self._calls: list[PreparedArticle] = []

    @property
    def identity(self) -> ClassifierIdentity:
        return self._identity

    @property
    def calls(self) -> tuple[PreparedArticle, ...]:
        return tuple(self._calls)

    def classify(self, article: PreparedArticle) -> ArticleClassification:
        self._calls.append(article)
        outcome = self._outcomes.get(article.text)
        if isinstance(outcome, ClassificationProviderError):
            raise outcome
        if outcome is not None:
            return outcome.model_copy(deep=True)
        return self._default_classification(article)

    @staticmethod
    def _default_classification(article: PreparedArticle) -> ArticleClassification:
        asset_sentiments = [
            AssetSentiment(
                asset=asset,
                sentiment=SentimentLabel.NEUTRAL,
                sentiment_score=0.0,
                confidence=1.0,
                category=NewsCategory.OTHER,
                reason="Deterministic fake classification for offline testing.",
            )
            for asset in article.candidate_assets
        ]
        return ArticleClassification(
            is_relevant=bool(asset_sentiments),
            asset_sentiments=asset_sentiments,
        )

"""Structured sentiment classification types and services."""

from cryptopulse.sentiment.cleaning import (
    PreparedArticle,
    clean_article_text,
    find_candidate_assets,
    prepare_article_text,
)
from cryptopulse.sentiment.fake_provider import FakeSentimentClassifier
from cryptopulse.sentiment.models import (
    ArticleClassification,
    AssetSentiment,
    CryptoAsset,
    NewsCategory,
    SentimentLabel,
)
from cryptopulse.sentiment.openai_provider import OpenAISentimentClassifier
from cryptopulse.sentiment.provider import (
    ClassificationProviderError,
    ClassificationResponseError,
    ClassifierIdentity,
    SentimentClassifier,
)

__all__ = [
    "ArticleClassification",
    "AssetSentiment",
    "ClassificationProviderError",
    "ClassificationResponseError",
    "ClassifierIdentity",
    "CryptoAsset",
    "FakeSentimentClassifier",
    "NewsCategory",
    "OpenAISentimentClassifier",
    "PreparedArticle",
    "SentimentClassifier",
    "SentimentLabel",
    "clean_article_text",
    "find_candidate_assets",
    "prepare_article_text",
]

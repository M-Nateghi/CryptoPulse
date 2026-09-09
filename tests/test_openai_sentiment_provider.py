import json
from unittest.mock import Mock

import pytest
from openai import OpenAIError

from cryptopulse.sentiment import (
    ArticleClassification,
    AssetSentiment,
    ClassificationProviderError,
    ClassificationResponseError,
    CryptoAsset,
    OpenAISentimentClassifier,
    PreparedArticle,
    SentimentClassifier,
)
from cryptopulse.sentiment.openai_provider import CLASSIFICATION_INSTRUCTIONS


def _article():
    return PreparedArticle(
        text="Bitcoin ETF demand rises",
        candidate_assets=(CryptoAsset.BTC,),
    )


def _classification():
    return ArticleClassification(
        is_relevant=True,
        asset_sentiments=[
            AssetSentiment(
                asset="BTC",
                sentiment="bullish",
                sentiment_score=0.8,
                confidence=0.9,
                category="etf_flows",
                reason="Reported ETF demand is positive for Bitcoin.",
            )
        ],
    )


def test_openai_classifier_uses_structured_output_contract():
    client = Mock()
    expected = _classification()
    client.responses.parse.return_value = Mock(output_parsed=expected)
    classifier = OpenAISentimentClassifier(client)

    result = classifier.classify(_article())

    assert isinstance(classifier, SentimentClassifier)
    assert result is expected
    request = client.responses.parse.call_args.kwargs
    assert request["model"] == "gpt-5.6-luna"
    assert request["text_format"] is ArticleClassification
    assert request["instructions"] == CLASSIFICATION_INSTRUCTIONS
    assert request["store"] is False
    assert request["max_output_tokens"] == 1_000
    assert json.loads(request["input"]) == {
        "candidate_assets": ["BTC"],
        "article_text": "Bitcoin ETF demand rises",
    }


def test_openai_classifier_exposes_versioned_identity():
    classifier = OpenAISentimentClassifier(Mock(), model="custom-model")

    assert classifier.identity.provider == "openai"
    assert classifier.identity.model == "custom-model"
    assert classifier.identity.prompt_version == "openai-sentiment-v1"


def test_openai_classifier_rejects_missing_parsed_output():
    client = Mock()
    client.responses.parse.return_value = Mock(output_parsed=None)

    with pytest.raises(ClassificationResponseError, match="no parsed"):
        OpenAISentimentClassifier(client).classify(_article())


def test_openai_classifier_rejects_wrong_parsed_output_type():
    client = Mock()
    client.responses.parse.return_value = Mock(output_parsed={"unexpected": True})

    with pytest.raises(ClassificationResponseError, match="no parsed"):
        OpenAISentimentClassifier(client).classify(_article())


def test_openai_classifier_wraps_sdk_errors():
    client = Mock()
    client.responses.parse.side_effect = OpenAIError("connection failed")

    with pytest.raises(ClassificationProviderError, match="OpenAI request failed"):
        OpenAISentimentClassifier(client).classify(_article())


def test_prompt_defines_domain_rules_without_article_content():
    assert "Keyword candidates" in CLASSIFICATION_INSTRUCTIONS
    assert "hints only" in CLASSIFICATION_INSTRUCTIONS
    assert "etf_flows" in CLASSIFICATION_INSTRUCTIONS
    assert "untrusted content" in CLASSIFICATION_INSTRUCTIONS
    assert _article().text not in CLASSIFICATION_INSTRUCTIONS

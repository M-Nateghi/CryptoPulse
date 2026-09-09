import pytest
from pydantic import ValidationError

from cryptopulse.sentiment.models import (
    ArticleClassification,
    CryptoAsset,
    NewsCategory,
    SentimentLabel,
)


def _asset_sentiment(**overrides):
    values = {
        "asset": "BTC",
        "sentiment": "bullish",
        "sentiment_score": 0.78,
        "confidence": 0.91,
        "category": "etf_flows",
        "reason": "ETF inflows indicate increased institutional demand for BTC.",
    }
    values.update(overrides)
    return values


def test_valid_single_asset_classification_is_typed():
    result = ArticleClassification.model_validate(
        {
            "is_relevant": True,
            "asset_sentiments": [_asset_sentiment(reason="  Strong BTC demand.  ")],
        }
    )

    asset_result = result.asset_sentiments[0]
    assert asset_result.asset is CryptoAsset.BTC
    assert asset_result.sentiment is SentimentLabel.BULLISH
    assert asset_result.category is NewsCategory.ETF_FLOWS
    assert asset_result.reason == "Strong BTC demand."


def test_valid_multi_asset_classification_can_hold_different_sentiment():
    result = ArticleClassification.model_validate(
        {
            "is_relevant": True,
            "asset_sentiments": [
                _asset_sentiment(),
                _asset_sentiment(
                    asset="ETH",
                    sentiment="bearish",
                    sentiment_score=-0.42,
                    category="regulation",
                ),
            ],
        }
    )

    assert [item.asset for item in result.asset_sentiments] == [
        CryptoAsset.BTC,
        CryptoAsset.ETH,
    ]
    assert result.asset_sentiments[1].sentiment is SentimentLabel.BEARISH


def test_irrelevant_article_has_an_explicit_empty_asset_list():
    result = ArticleClassification.model_validate(
        {"is_relevant": False, "asset_sentiments": []}
    )

    assert result.asset_sentiments == []


@pytest.mark.parametrize(
    ("is_relevant", "asset_sentiments"),
    [
        (True, []),
        (False, [_asset_sentiment()]),
    ],
)
def test_relevance_must_agree_with_asset_results(is_relevant, asset_sentiments):
    with pytest.raises(ValidationError, match="is_relevant must be true exactly"):
        ArticleClassification.model_validate(
            {
                "is_relevant": is_relevant,
                "asset_sentiments": asset_sentiments,
            }
        )


def test_duplicate_assets_are_rejected():
    with pytest.raises(ValidationError, match="duplicate assets"):
        ArticleClassification.model_validate(
            {
                "is_relevant": True,
                "asset_sentiments": [
                    _asset_sentiment(),
                    _asset_sentiment(sentiment="neutral", sentiment_score=0.0),
                ],
            }
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("asset", "DOGE"),
        ("sentiment", "positive"),
        ("category", "partnership"),
        ("sentiment_score", -1.01),
        ("sentiment_score", 1.01),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("reason", "   "),
    ],
)
def test_invalid_asset_result_fields_are_rejected(field, invalid_value):
    with pytest.raises(ValidationError):
        ArticleClassification.model_validate(
            {
                "is_relevant": True,
                "asset_sentiments": [_asset_sentiment(**{field: invalid_value})],
            }
        )


def test_numeric_strings_are_not_silently_converted():
    with pytest.raises(ValidationError):
        ArticleClassification.model_validate(
            {
                "is_relevant": True,
                "asset_sentiments": [_asset_sentiment(confidence="0.91")],
            }
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ArticleClassification.model_validate(
            {
                "is_relevant": True,
                "asset_sentiments": [
                    _asset_sentiment(summary="This field is not in the contract")
                ],
            }
        )

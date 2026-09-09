import pytest

from cryptopulse.sentiment import (
    ArticleClassification,
    AssetSentiment,
    ClassificationProviderError,
    CryptoAsset,
    FakeSentimentClassifier,
    NewsCategory,
    PreparedArticle,
    SentimentClassifier,
    SentimentLabel,
)


def test_fake_classifier_satisfies_provider_protocol():
    classifier = FakeSentimentClassifier()

    assert isinstance(classifier, SentimentClassifier)
    assert classifier.identity.provider == "fake"
    assert classifier.identity.model == "deterministic-test-classifier"
    assert classifier.identity.prompt_version == "v1"


def test_fake_classifier_returns_irrelevant_for_no_candidates():
    classifier = FakeSentimentClassifier()
    article = PreparedArticle(text="General market update", candidate_assets=())

    result = classifier.classify(article)

    assert result == ArticleClassification(
        is_relevant=False,
        asset_sentiments=[],
    )


def test_fake_classifier_returns_neutral_results_for_candidates():
    classifier = FakeSentimentClassifier()
    article = PreparedArticle(
        text="Bitcoin and Ethereum market update",
        candidate_assets=(CryptoAsset.BTC, CryptoAsset.ETH),
    )

    result = classifier.classify(article)

    assert result.is_relevant is True
    assert [item.asset for item in result.asset_sentiments] == [
        CryptoAsset.BTC,
        CryptoAsset.ETH,
    ]
    assert all(
        item.sentiment is SentimentLabel.NEUTRAL
        and item.sentiment_score == 0.0
        and item.confidence == 1.0
        and item.category is NewsCategory.OTHER
        for item in result.asset_sentiments
    )


def test_fake_classifier_can_return_a_scripted_result():
    article = PreparedArticle(
        text="Bitcoin ETF demand rises",
        candidate_assets=(CryptoAsset.BTC,),
    )
    scripted = ArticleClassification(
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
    classifier = FakeSentimentClassifier({article.text: scripted})

    result = classifier.classify(article)

    assert result == scripted
    assert result is not scripted


def test_fake_classifier_can_raise_a_scripted_error():
    article = PreparedArticle(
        text="Provider failure example",
        candidate_assets=(),
    )
    classifier = FakeSentimentClassifier(
        {article.text: ClassificationProviderError("simulated provider failure")}
    )

    with pytest.raises(ClassificationProviderError, match="simulated provider failure"):
        classifier.classify(article)


def test_fake_classifier_records_calls_in_order():
    classifier = FakeSentimentClassifier()
    first = PreparedArticle(text="First article", candidate_assets=())
    second = PreparedArticle(text="Second article", candidate_assets=())

    classifier.classify(first)
    classifier.classify(second)

    assert classifier.calls == (first, second)

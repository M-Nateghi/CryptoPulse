from datetime import UTC, datetime, timedelta

from cryptopulse.analytics.queries import load_market_data, load_sentiment_data
from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article, MarketCandle
from cryptopulse.db.repositories import (
    insert_articles,
    insert_market_candles,
    save_classification_success,
)
from cryptopulse.db.schema import create_schema
from cryptopulse.sentiment import (
    ArticleClassification,
    AssetSentiment,
    ClassifierIdentity,
)


def test_load_market_data_parses_utc_and_orders_assets(tmp_path):
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        insert_market_candles(
            connection,
            [
                MarketCandle(
                    asset="ETH",
                    symbol="ETHUSDT",
                    candle_timestamp=datetime(2026, 9, 1, tzinfo=UTC),
                    open=100,
                    high=110,
                    low=90,
                    close=105,
                    volume=10,
                    quote_volume=1_000,
                    trade_count=20,
                )
            ],
        )

        result = load_market_data(connection)

    assert result.iloc[0]["asset"] == "ETH"
    assert str(result["candle_timestamp"].dt.tz) == "UTC"


def test_load_sentiment_data_selects_latest_success_for_provider(tmp_path):
    published_at = datetime(2026, 9, 1, 10, tzinfo=UTC)
    with open_database(tmp_path / "test.db") as connection:
        create_schema(connection)
        insert_articles(
            connection,
            [
                Article(
                    source="gdelt",
                    external_id=None,
                    title="Bitcoin adoption grows",
                    url="https://example.com/btc",
                    published_at=published_at,
                    retrieved_at=published_at,
                    raw_query="Bitcoin",
                )
            ],
        )
        article_id = connection.execute("SELECT id FROM articles").fetchone()["id"]
        for prompt_version, score, processed_offset in (
            ("v1", -0.3, 0),
            ("v2", 0.7, 1),
        ):
            save_classification_success(
                connection,
                article_id,
                ClassifierIdentity(
                    provider="openai",
                    model="test-model",
                    prompt_version=prompt_version,
                ),
                "Bitcoin adoption grows",
                ArticleClassification(
                    is_relevant=True,
                    asset_sentiments=[
                        AssetSentiment(
                            asset="BTC",
                            sentiment="bullish" if score > 0 else "bearish",
                            sentiment_score=score,
                            confidence=0.9,
                            category="institutional_adoption",
                            reason="Test evidence.",
                        )
                    ],
                ),
                published_at + timedelta(hours=processed_offset),
            )

        result = load_sentiment_data(connection)
        fake_result = load_sentiment_data(connection, provider="fake")
        fake_result = load_sentiment_data(connection, provider="fake")

    assert len(result) == 1
    assert result.iloc[0]["prompt_version"] == "v2"
    assert result.iloc[0]["sentiment_score"] == 0.7
    assert str(result["published_at"].dtype) == "datetime64[us, UTC]"
    assert fake_result.empty

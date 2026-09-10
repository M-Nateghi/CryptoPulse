from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from cryptopulse.analytics.features import (
    add_market_features,
    aggregate_category_drivers,
    aggregate_sentiment,
    build_sentiment_market_history,
)


def _market_frame() -> pd.DataFrame:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    rows = []
    for asset, starting_price in (("BTC", 100.0), ("ETH", 50.0)):
        for hour in range(25):
            rows.append(
                {
                    "asset": asset,
                    "symbol": f"{asset}USDT",
                    "candle_timestamp": start + timedelta(hours=hour),
                    "open": starting_price + hour,
                    "high": starting_price + hour + 1,
                    "low": starting_price + hour - 1,
                    "close": starting_price + hour,
                    "volume": 1_000.0 + hour * 10,
                    "quote_volume": 100_000.0,
                    "trade_count": 100,
                }
            )
    return pd.DataFrame(rows)


def _sentiment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": 1,
                "published_at": pd.Timestamp("2026-09-01T10:00:00Z"),
                "asset": "BTC",
                "sentiment": "bullish",
                "sentiment_score": 0.8,
                "confidence": 0.75,
                "category": "etf_flows",
            },
            {
                "article_id": 2,
                "published_at": pd.Timestamp("2026-09-01T11:00:00Z"),
                "asset": "BTC",
                "sentiment": "bearish",
                "sentiment_score": -0.4,
                "confidence": 0.25,
                "category": "regulation",
            },
        ]
    )


def test_add_market_features_calculates_each_asset_independently():
    result = add_market_features(_market_frame())

    btc = result[result["asset"] == "BTC"].reset_index(drop=True)
    eth = result[result["asset"] == "ETH"].reset_index(drop=True)
    assert np.isnan(btc.loc[0, "simple_return_pct"])
    assert np.isnan(eth.loc[0, "simple_return_pct"])
    assert btc.loc[1, "simple_return_pct"] == pytest.approx(1.0)
    assert btc.loc[24, "price_change_24h_pct"] == pytest.approx(24.0)
    assert btc.loc[23, "rolling_volatility_24h_pct"] >= 0


def test_add_market_features_preserves_expected_columns_for_empty_input():
    result = add_market_features(
        pd.DataFrame(columns=["asset", "candle_timestamp", "close", "volume"])
    )

    assert "price_change_24h_pct" in result.columns
    assert "volume_zscore_24h" in result.columns
    assert result.empty


def test_aggregate_sentiment_uses_confidence_weighting_and_percentages():
    result = aggregate_sentiment(_sentiment_frame()).iloc[0]

    assert result["story_count"] == 2
    assert result["weighted_sentiment"] == pytest.approx(0.5)
    assert result["sentiment_score_0_100"] == pytest.approx(75.0)
    assert result["bullish_pct"] == pytest.approx(50.0)
    assert result["bearish_pct"] == pytest.approx(50.0)
    assert result["neutral_pct"] == 0


def test_aggregate_sentiment_falls_back_when_confidence_is_zero():
    frame = _sentiment_frame()
    frame["confidence"] = 0.0

    result = aggregate_sentiment(frame).iloc[0]

    assert result["weighted_sentiment"] == pytest.approx(0.2)


def test_category_drivers_aggregate_separately():
    result = aggregate_category_drivers(_sentiment_frame())

    assert set(result["category"]) == {"etf_flows", "regulation"}
    assert set(result["story_count"]) == {1}


def test_sentiment_and_market_history_align_on_asset_and_day():
    market = add_market_features(_market_frame())
    sentiment = aggregate_sentiment(_sentiment_frame())

    result = build_sentiment_market_history(sentiment, market)

    assert len(result) == 1
    assert result.iloc[0]["asset"] == "BTC"
    assert result.iloc[0]["close"] == pytest.approx(123.0)

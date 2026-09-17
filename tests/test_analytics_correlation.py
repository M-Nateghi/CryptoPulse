from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pandas as pd
import pytest

from cryptopulse.analytics.correlation import build_sentiment_price_correlation


def _market_frame(period_returns: list[float]) -> pd.DataFrame:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    rows = []
    for period, period_return in enumerate(period_returns):
        period_start = start + timedelta(hours=period * 6)
        for hour in range(6):
            rows.append(
                {
                    "asset": "BTC",
                    "candle_timestamp": period_start + timedelta(hours=hour),
                    "open": 100.0,
                    "close": (
                        100.0 * (1 + period_return / 100)
                        if hour == 5
                        else 100.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sentiment_frame(scores: list[float]) -> pd.DataFrame:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    return pd.DataFrame(
        [
            {
                "article_id": index + 1,
                "asset": "BTC",
                "published_at": start + timedelta(hours=index * 6 + 1),
                "sentiment_score": score,
                "confidence": 1.0,
            }
            for index, score in enumerate(scores)
        ]
    )


def test_same_period_alignment_and_rolling_spearman_correlation():
    scores = [0.0, 0.1, -0.1, 0.3, 0.0, 0.4, 0.2, 0.5, 0.1, 0.6, 0.0]
    score_changes = [0.0] + [
        (current - previous) * 50
        for previous, current in pairwise(scores)
    ]

    result = build_sentiment_price_correlation(
        _sentiment_frame(scores),
        _market_frame(score_changes),
        timing="same_6h",
    )

    latest = result.iloc[-1]
    assert latest["rolling_observations"] == 10
    assert latest["rolling_correlation"] == pytest.approx(1.0, abs=0.01)
    assert latest["sentiment_change"] == pytest.approx(-30.0)
    assert latest["price_change_pct"] == pytest.approx(-30.0)


def test_sentiment_change_does_not_bridge_a_missing_six_hour_period():
    sentiment = _sentiment_frame([0.1, 0.2, 0.3]).drop(index=1).reset_index(drop=True)

    result = build_sentiment_price_correlation(
        sentiment,
        _market_frame([1.0, 2.0, 3.0]),
        timing="same_6h",
        min_observations=2,
    )

    final_period = result.iloc[-1]
    assert pd.isna(final_period["sentiment_change"])


def test_next_period_price_return_uses_exact_future_bucket():
    result = build_sentiment_price_correlation(
        _sentiment_frame([0.0, 0.5, 0.2]),
        _market_frame([1.0, 3.0, 6.0]),
        timing="next_6h",
        min_observations=2,
    )

    first_period = result.iloc[0]
    assert first_period["price_change_pct"] == pytest.approx((103 / 101 - 1) * 100)
    assert pd.isna(result.iloc[-1]["price_change_pct"])

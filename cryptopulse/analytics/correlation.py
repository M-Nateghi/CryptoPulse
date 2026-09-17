from typing import Literal

import numpy as np
import pandas as pd

CorrelationTiming = Literal["same_6h", "next_6h", "next_24h"]
CORRELATION_TIMINGS: tuple[CorrelationTiming, ...] = (
    "same_6h",
    "next_6h",
    "next_24h",
)


def _aggregate_sentiment(
    sentiment_data: pd.DataFrame,
    bucket_hours: int,
) -> pd.DataFrame:
    frame = sentiment_data.copy()
    frame["period_start"] = frame["published_at"].dt.floor(f"{bucket_hours}h")
    frame["weighted_component"] = frame["sentiment_score"] * frame["confidence"]
    grouped = frame.groupby(["asset", "period_start"], observed=True, sort=True)
    result = grouped.agg(
        story_count=("article_id", "nunique"),
        mean_sentiment=("sentiment_score", "mean"),
        weighted_total=("weighted_component", "sum"),
        confidence_total=("confidence", "sum"),
    ).reset_index()
    result["sentiment_level"] = np.where(
        result["confidence_total"].gt(0),
        result["weighted_total"] / result["confidence_total"],
        result["mean_sentiment"],
    )
    result["sentiment_score_0_100"] = (
        (result["sentiment_level"] + 1) * 50
    ).clip(0, 100)

    previous = result[
        ["asset", "period_start", "sentiment_score_0_100"]
    ].copy()
    previous["period_start"] += pd.Timedelta(hours=bucket_hours)
    previous = previous.rename(
        columns={"sentiment_score_0_100": "previous_sentiment_score"}
    )
    result = result.merge(
        previous,
        on=["asset", "period_start"],
        how="left",
        validate="one_to_one",
    )
    result["sentiment_change"] = (
        result["sentiment_score_0_100"] - result["previous_sentiment_score"]
    )
    return result[
        [
            "asset",
            "period_start",
            "story_count",
            "sentiment_score_0_100",
            "sentiment_change",
        ]
    ]


def _aggregate_market(
    market_data: pd.DataFrame,
    bucket_hours: int,
    timing: CorrelationTiming,
) -> pd.DataFrame:
    frame = market_data.copy()
    frame["period_start"] = frame["candle_timestamp"].dt.floor(
        f"{bucket_hours}h"
    )
    grouped = frame.groupby(["asset", "period_start"], observed=True, sort=True)
    result = grouped.agg(
        period_open=("open", "first"),
        period_close=("close", "last"),
        candle_count=("candle_timestamp", "nunique"),
    ).reset_index()
    result = result[result["candle_count"].eq(bucket_hours)].copy()

    if timing == "same_6h":
        result["price_change_pct"] = (
            result["period_close"] / result["period_open"] - 1
        ) * 100
        return result

    horizon_hours = 6 if timing == "next_6h" else 24
    future = result[["asset", "period_start", "period_close"]].copy()
    future["period_start"] -= pd.Timedelta(hours=horizon_hours)
    future = future.rename(columns={"period_close": "future_close"})
    result = result.merge(
        future,
        on=["asset", "period_start"],
        how="left",
        validate="one_to_one",
    )
    result["price_change_pct"] = (
        result["future_close"] / result["period_close"] - 1
    ) * 100
    return result


def _add_rolling_correlation(
    frame: pd.DataFrame,
    window_periods: int,
    min_observations: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["rolling_observations"] = 0
    result["rolling_correlation"] = np.nan

    for _, asset_frame in result.groupby("asset", sort=False):
        ordered_indices = asset_frame.sort_values("period_start").index.tolist()
        for position, row_index in enumerate(ordered_indices):
            window_indices = ordered_indices[
                max(0, position - window_periods + 1) : position + 1
            ]
            window = result.loc[
                window_indices, ["sentiment_change", "price_change_pct"]
            ].dropna()
            observation_count = len(window)
            result.at[row_index, "rolling_observations"] = observation_count
            if (
                observation_count < min_observations
                or window["sentiment_change"].nunique() < 2
                or window["price_change_pct"].nunique() < 2
            ):
                continue
            result.at[row_index, "rolling_correlation"] = window[
                "sentiment_change"
            ].rank().corr(window["price_change_pct"].rank())

    return result


def build_sentiment_price_correlation(
    sentiment_data: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    timing: CorrelationTiming = "next_6h",
    bucket_hours: int = 6,
    rolling_days: int = 7,
    min_observations: int = 10,
) -> pd.DataFrame:
    """Build aligned sentiment/price changes and a rolling Spearman correlation."""
    if timing not in CORRELATION_TIMINGS:
        raise ValueError(f"Unsupported correlation timing: {timing}")
    if bucket_hours <= 0 or 24 % bucket_hours:
        raise ValueError("Bucket hours must be a positive divisor of 24")
    if rolling_days <= 0:
        raise ValueError("Rolling days must be positive")
    window_periods = rolling_days * 24 // bucket_hours
    if not 2 <= min_observations <= window_periods:
        raise ValueError("Minimum observations must fit inside the rolling window")

    columns = [
        "asset",
        "period_start",
        "period_end",
        "story_count",
        "sentiment_score_0_100",
        "sentiment_change",
        "price_change_pct",
        "rolling_observations",
        "rolling_correlation",
    ]
    if sentiment_data.empty or market_data.empty:
        return pd.DataFrame(columns=columns)

    sentiment = sentiment_data.copy()
    sentiment["published_at"] = pd.to_datetime(
        sentiment["published_at"], utc=True
    )
    market = market_data.copy()
    market["candle_timestamp"] = pd.to_datetime(
        market["candle_timestamp"], utc=True
    )

    sentiment_periods = _aggregate_sentiment(sentiment, bucket_hours)
    market_periods = _aggregate_market(market, bucket_hours, timing)
    aligned = market_periods.merge(
        sentiment_periods,
        on=["asset", "period_start"],
        how="left",
        validate="one_to_one",
    )
    aligned["period_end"] = aligned["period_start"] + pd.Timedelta(
        hours=bucket_hours
    )
    aligned = _add_rolling_correlation(
        aligned,
        window_periods=window_periods,
        min_observations=min_observations,
    )
    return aligned[columns].sort_values(["asset", "period_start"]).reset_index(
        drop=True
    )

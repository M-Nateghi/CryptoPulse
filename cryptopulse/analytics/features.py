import math

import numpy as np
import pandas as pd

MARKET_FEATURE_COLUMNS = (
    "simple_return_pct",
    "log_return",
    "price_change_24h_pct",
    "rolling_volatility_24h_pct",
    "volume_change_pct",
    "volume_zscore_24h",
)


def add_market_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns, volatility, and volume movement per asset."""
    frame = market_data.copy()
    for column in MARKET_FEATURE_COLUMNS:
        frame[column] = pd.Series(dtype="float64")
    if frame.empty:
        return frame

    frame = frame.sort_values(["asset", "candle_timestamp"]).reset_index(drop=True)
    grouped_close = frame.groupby("asset", sort=False)["close"]
    grouped_volume = frame.groupby("asset", sort=False)["volume"]

    frame["simple_return_pct"] = grouped_close.transform(
        lambda values: values.pct_change(fill_method=None) * 100
    )
    frame["log_return"] = grouped_close.transform(
        lambda values: np.log(values.replace(0, np.nan)).diff()
    )
    frame["price_change_24h_pct"] = grouped_close.transform(
        lambda values: values.pct_change(periods=24, fill_method=None) * 100
    )
    frame["rolling_volatility_24h_pct"] = frame.groupby(
        "asset", sort=False
    )["log_return"].transform(
        lambda values: values.rolling(window=24, min_periods=6).std()
        * math.sqrt(24)
        * 100
    )
    frame["volume_change_pct"] = grouped_volume.transform(
        lambda values: values.pct_change(fill_method=None) * 100
    )

    rolling_mean = grouped_volume.transform(
        lambda values: values.rolling(window=24, min_periods=6).mean()
    )
    rolling_std = grouped_volume.transform(
        lambda values: values.rolling(window=24, min_periods=6).std()
    )
    frame["volume_zscore_24h"] = (frame["volume"] - rolling_mean) / rolling_std
    frame.loc[rolling_std.eq(0), "volume_zscore_24h"] = np.nan
    return frame


def aggregate_sentiment(
    sentiment_data: pd.DataFrame,
    frequency: str = "D",
) -> pd.DataFrame:
    """Aggregate article sentiment by asset and time period."""
    columns = [
        "asset",
        "period",
        "story_count",
        "mean_sentiment",
        "weighted_sentiment",
        "sentiment_score_0_100",
        "mean_confidence",
        "bullish_pct",
        "neutral_pct",
        "bearish_pct",
    ]
    if sentiment_data.empty:
        return pd.DataFrame(columns=columns)

    frame = sentiment_data.copy()
    frame["period"] = frame["published_at"].dt.floor(frequency)
    frame["weighted_component"] = frame["sentiment_score"] * frame["confidence"]

    grouped = frame.groupby(["asset", "period"], observed=True, sort=True)
    result = grouped.agg(
        story_count=("article_id", "nunique"),
        mean_sentiment=("sentiment_score", "mean"),
        weighted_total=("weighted_component", "sum"),
        confidence_total=("confidence", "sum"),
        mean_confidence=("confidence", "mean"),
        bullish_count=("sentiment", lambda values: values.eq("bullish").sum()),
        neutral_count=("sentiment", lambda values: values.eq("neutral").sum()),
        bearish_count=("sentiment", lambda values: values.eq("bearish").sum()),
    ).reset_index()

    result["weighted_sentiment"] = np.where(
        result["confidence_total"].gt(0),
        result["weighted_total"] / result["confidence_total"],
        result["mean_sentiment"],
    )
    result["sentiment_score_0_100"] = (
        (result["weighted_sentiment"] + 1) * 50
    ).clip(0, 100)
    result_count = (
        result["bullish_count"] + result["neutral_count"] + result["bearish_count"]
    )
    for label in ("bullish", "neutral", "bearish"):
        result[f"{label}_pct"] = result[f"{label}_count"] / result_count * 100
    return result[columns]


def aggregate_category_drivers(sentiment_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize which news categories contribute to each asset's sentiment."""
    columns = [
        "asset",
        "category",
        "story_count",
        "weighted_sentiment",
        "sentiment_score_0_100",
        "mean_confidence",
    ]
    if sentiment_data.empty:
        return pd.DataFrame(columns=columns)

    frame = sentiment_data.copy()
    frame["weighted_component"] = frame["sentiment_score"] * frame["confidence"]
    grouped = frame.groupby(["asset", "category"], observed=True, sort=True)
    result = grouped.agg(
        story_count=("article_id", "nunique"),
        mean_sentiment=("sentiment_score", "mean"),
        weighted_total=("weighted_component", "sum"),
        confidence_total=("confidence", "sum"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    result["weighted_sentiment"] = np.where(
        result["confidence_total"].gt(0),
        result["weighted_total"] / result["confidence_total"],
        result["mean_sentiment"],
    )
    result["sentiment_score_0_100"] = (
        (result["weighted_sentiment"] + 1) * 50
    ).clip(0, 100)
    return result[columns]


def build_sentiment_market_history(
    sentiment_history: pd.DataFrame,
    market_features: pd.DataFrame,
    frequency: str = "D",
) -> pd.DataFrame:
    """Align sentiment periods with the final market observation in each period."""
    market_columns = [
        "asset",
        "period",
        "close",
        "price_change_24h_pct",
        "rolling_volatility_24h_pct",
        "volume",
        "volume_zscore_24h",
    ]
    if market_features.empty:
        return sentiment_history.merge(
            pd.DataFrame(columns=market_columns),
            on=["asset", "period"],
            how="left",
        )

    frame = market_features.copy()
    frame["period"] = frame["candle_timestamp"].dt.floor(frequency)
    market_history = (
        frame.sort_values(["asset", "candle_timestamp"])
        .groupby(["asset", "period"], observed=True, sort=True)
        .tail(1)[market_columns]
    )
    return sentiment_history.merge(
        market_history,
        on=["asset", "period"],
        how="left",
        validate="one_to_one",
    )

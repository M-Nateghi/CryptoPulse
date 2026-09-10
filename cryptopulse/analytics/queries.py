import sqlite3

import pandas as pd


def load_sentiment_data(
    connection: sqlite3.Connection,
    provider: str = "openai",
) -> pd.DataFrame:
    """Load the latest successful classification for each article and provider."""
    frame = pd.read_sql_query(
        """
        WITH ranked_classifications AS (
            SELECT
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.article_id, c.provider
                    ORDER BY c.processed_at DESC, c.id DESC
                ) AS version_rank
            FROM article_classifications AS c
            WHERE c.status = 'succeeded'
              AND c.provider = ?
        )
        SELECT
            a.id AS article_id,
            a.title,
            a.url,
            a.published_at,
            c.provider,
            c.model,
            c.prompt_version,
            c.processed_at,
            s.asset,
            s.sentiment,
            s.sentiment_score,
            s.confidence,
            s.category,
            s.reason
        FROM ranked_classifications AS c
        JOIN articles AS a ON a.id = c.article_id
        JOIN sentiment_results AS s ON s.classification_id = c.id
        WHERE c.version_rank = 1
          AND c.is_relevant = 1
        ORDER BY a.published_at, a.id, s.asset
        """,
        connection,
        params=(provider,),
    )
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["processed_at"] = pd.to_datetime(frame["processed_at"], utc=True)
    return frame


def load_market_data(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load validated hourly market candles in chronological order."""
    frame = pd.read_sql_query(
        """
        SELECT
            asset,
            symbol,
            candle_timestamp,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trade_count
        FROM market_data
        ORDER BY asset, candle_timestamp
        """,
        connection,
    )
    frame["candle_timestamp"] = pd.to_datetime(
        frame["candle_timestamp"], utc=True
    )
    return frame

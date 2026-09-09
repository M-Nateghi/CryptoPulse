import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    external_id TEXT,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    raw_query TEXT NOT NULL,
    UNIQUE (source, external_id),
    UNIQUE (source, url)
);

CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY,
    asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH', 'SOL', 'BNB')),
    symbol TEXT NOT NULL,
    candle_timestamp TEXT NOT NULL,
    open REAL NOT NULL CHECK (open >= 0),
    high REAL NOT NULL CHECK (high >= 0),
    low REAL NOT NULL CHECK (low >= 0),
    close REAL NOT NULL CHECK (close >= 0),
    volume REAL NOT NULL CHECK (volume >= 0),
    quote_volume REAL NOT NULL CHECK (quote_volume >= 0),
    trade_count INTEGER NOT NULL CHECK (trade_count >= 0),
    CHECK (high >= low),
    UNIQUE (symbol, candle_timestamp)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    records_received INTEGER NOT NULL DEFAULT 0 CHECK (records_received >= 0),
    records_inserted INTEGER NOT NULL DEFAULT 0 CHECK (records_inserted >= 0),
    records_skipped INTEGER NOT NULL DEFAULT 0 CHECK (records_skipped >= 0),
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS article_classifications (
    id INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL,
    provider TEXT NOT NULL CHECK (provider <> ''),
    model TEXT NOT NULL CHECK (model <> ''),
    prompt_version TEXT NOT NULL CHECK (prompt_version <> ''),
    input_text TEXT NOT NULL CHECK (input_text <> ''),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    is_relevant INTEGER CHECK (is_relevant IN (0, 1)),
    processed_at TEXT NOT NULL,
    error_message TEXT,
    FOREIGN KEY (article_id) REFERENCES articles (id) ON DELETE CASCADE,
    UNIQUE (article_id, provider, model, prompt_version),
    CHECK (
        (status = 'succeeded' AND is_relevant IS NOT NULL AND error_message IS NULL)
        OR
        (status = 'failed' AND is_relevant IS NULL AND error_message IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS sentiment_results (
    id INTEGER PRIMARY KEY,
    classification_id INTEGER NOT NULL,
    asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH', 'SOL', 'BNB')),
    sentiment TEXT NOT NULL CHECK (sentiment IN ('bullish', 'neutral', 'bearish')),
    sentiment_score REAL NOT NULL CHECK (
        sentiment_score >= -1 AND sentiment_score <= 1
    ),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    category TEXT NOT NULL CHECK (
        category IN (
            'etf_flows',
            'institutional_adoption',
            'regulation',
            'technology',
            'security_hacks',
            'market_movement',
            'macro',
            'exchange_activity',
            'other'
        )
    ),
    reason TEXT NOT NULL CHECK (reason <> ''),
    FOREIGN KEY (classification_id)
        REFERENCES article_classifications (id) ON DELETE CASCADE,
    UNIQUE (classification_id, asset)
);

CREATE INDEX IF NOT EXISTS idx_articles_published_at
ON articles (published_at);

CREATE INDEX IF NOT EXISTS idx_market_data_asset_timestamp
ON market_data (asset, candle_timestamp);

CREATE INDEX IF NOT EXISTS idx_article_classifications_status
ON article_classifications (status);

CREATE INDEX IF NOT EXISTS idx_sentiment_results_asset
ON sentiment_results (asset);
"""

BNB_ASSET_MIGRATION_SQL = """
BEGIN IMMEDIATE;

CREATE TABLE market_data_bnb (
    id INTEGER PRIMARY KEY,
    asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH', 'SOL', 'BNB')),
    symbol TEXT NOT NULL,
    candle_timestamp TEXT NOT NULL,
    open REAL NOT NULL CHECK (open >= 0),
    high REAL NOT NULL CHECK (high >= 0),
    low REAL NOT NULL CHECK (low >= 0),
    close REAL NOT NULL CHECK (close >= 0),
    volume REAL NOT NULL CHECK (volume >= 0),
    quote_volume REAL NOT NULL CHECK (quote_volume >= 0),
    trade_count INTEGER NOT NULL CHECK (trade_count >= 0),
    CHECK (high >= low),
    UNIQUE (symbol, candle_timestamp)
);

INSERT INTO market_data_bnb
SELECT * FROM market_data;

CREATE TABLE sentiment_results_bnb (
    id INTEGER PRIMARY KEY,
    classification_id INTEGER NOT NULL,
    asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH', 'SOL', 'BNB')),
    sentiment TEXT NOT NULL CHECK (sentiment IN ('bullish', 'neutral', 'bearish')),
    sentiment_score REAL NOT NULL CHECK (
        sentiment_score >= -1 AND sentiment_score <= 1
    ),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    category TEXT NOT NULL CHECK (
        category IN (
            'etf_flows',
            'institutional_adoption',
            'regulation',
            'technology',
            'security_hacks',
            'market_movement',
            'macro',
            'exchange_activity',
            'other'
        )
    ),
    reason TEXT NOT NULL CHECK (reason <> ''),
    FOREIGN KEY (classification_id)
        REFERENCES article_classifications (id) ON DELETE CASCADE,
    UNIQUE (classification_id, asset)
);

INSERT INTO sentiment_results_bnb
SELECT * FROM sentiment_results;

DROP TABLE sentiment_results;
DROP TABLE market_data;
ALTER TABLE sentiment_results_bnb RENAME TO sentiment_results;
ALTER TABLE market_data_bnb RENAME TO market_data;

COMMIT;
"""


def _asset_tables_support_bnb(connection: sqlite3.Connection) -> bool:
    for table_name in ("market_data", "sentiment_results"):
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if row is None or "'BNB'" not in row[0]:
            return False
    return True


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the current CryptoPulse database tables and indexes."""
    connection.executescript(SCHEMA_SQL)
    if not _asset_tables_support_bnb(connection):
        try:
            connection.executescript(BNB_ASSET_MIGRATION_SQL)
        except sqlite3.Error:
            connection.rollback()
            raise
        connection.executescript(SCHEMA_SQL)

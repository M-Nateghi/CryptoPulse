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
    asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH', 'SOL')),
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

CREATE INDEX IF NOT EXISTS idx_articles_published_at
ON articles (published_at);

CREATE INDEX IF NOT EXISTS idx_market_data_asset_timestamp
ON market_data (asset, candle_timestamp);
"""


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the Stage 1 database tables and indexes."""
    connection.executescript(SCHEMA_SQL)

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Article:
    source: str
    external_id: str | None
    title: str
    url: str
    published_at: datetime
    retrieved_at: datetime
    raw_query: str


@dataclass(frozen=True)
class MarketCandle:
    asset: str
    symbol: str
    candle_timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int


@dataclass(frozen=True)
class InsertSummary:
    received: int
    inserted: int
    skipped: int


@dataclass(frozen=True)
class ArticleForClassification:
    id: int
    title: str

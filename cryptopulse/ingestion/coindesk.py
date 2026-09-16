from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx

from cryptopulse.db.models import Article
from cryptopulse.ingestion.gdelt import ASSET_QUERY_TERMS
from cryptopulse.sentiment.cleaning import clean_article_text, find_candidate_assets

SOURCE_NAME = "coindesk"
FEED_PATH = "/arc/outboundfeeds/rss"
ALLOWED_TIMESPANS = {"1d": 1, "3d": 3, "7d": 7}


def _required_text(item: ElementTree.Element, field: str, index: int) -> str:
    value = item.findtext(field)
    if value is None or not value.strip():
        raise ValueError(f"Missing CoinDesk {field} at index {index}")
    return value.strip()


def _published_at(value: str, index: int) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid CoinDesk pubDate at index {index}: {value}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"CoinDesk pubDate at index {index} has no timezone")
    return parsed.astimezone(UTC)


def _optional_summary(value: str | None, title: str) -> str | None:
    if value is None:
        return None
    try:
        summary = clean_article_text(value)
    except ValueError:
        return None
    return None if summary.casefold() == title.casefold() else summary


def parse_coindesk_articles(
    content: bytes,
    *,
    retrieved_at: datetime,
    raw_query: str,
) -> list[Article]:
    """Convert the official CoinDesk RSS feed into canonical articles."""
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("CoinDesk retrieval time must include a timezone")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise RuntimeError("CoinDesk returned invalid RSS XML") from error

    articles = []
    for index, item in enumerate(root.findall("./channel/item")):
        title = clean_article_text(_required_text(item, "title", index))
        link = _required_text(item, "link", index)
        external_id = (item.findtext("guid") or link).strip()
        articles.append(
            Article(
                source=SOURCE_NAME,
                external_id=external_id,
                title=title,
                summary=_optional_summary(item.findtext("description"), title),
                url=link,
                published_at=_published_at(
                    _required_text(item, "pubDate", index),
                    index,
                ),
                retrieved_at=retrieved_at.astimezone(UTC),
                raw_query=raw_query,
            )
        )
    return articles


class CoinDeskRssClient:
    source_name = SOURCE_NAME

    def __init__(self, http_client: httpx.Client) -> None:
        self._http_client = http_client

    def fetch_recent_articles_for_assets(
        self,
        assets: Iterable[str],
        max_records: int = 50,
        timespan: str = "1d",
        retrieved_at: datetime | None = None,
    ) -> list[Article]:
        """Fetch recent CoinDesk RSS items mentioning selected assets."""
        normalized_assets = tuple(dict.fromkeys(asset.upper() for asset in assets))
        if not normalized_assets:
            raise ValueError("At least one CoinDesk asset is required")
        unsupported_assets = [
            asset for asset in normalized_assets if asset not in ASSET_QUERY_TERMS
        ]
        if unsupported_assets:
            raise ValueError(f"Unsupported CoinDesk asset: {unsupported_assets[0]}")
        if not 1 <= max_records <= 250:
            raise ValueError("CoinDesk max_records must be between 1 and 250")
        if timespan not in ALLOWED_TIMESPANS:
            raise ValueError("CoinDesk timespan must be 1d, 3d, or 7d")

        observed_at = retrieved_at or datetime.now(UTC)
        response = self._http_client.get(FEED_PATH)
        response.raise_for_status()
        articles = parse_coindesk_articles(
            response.content,
            retrieved_at=observed_at,
            raw_query=", ".join(normalized_assets),
        )
        selected_assets = set(normalized_assets)
        cutoff = observed_at - timedelta(days=ALLOWED_TIMESPANS[timespan])
        matching = [
            article
            for article in articles
            if article.published_at >= cutoff
            and selected_assets.intersection(
                asset.value
                for asset in find_candidate_assets(
                    " ".join(
                        part
                        for part in (article.title, article.summary)
                        if part is not None
                    )
                )
            )
        ]
        return matching[:max_records]

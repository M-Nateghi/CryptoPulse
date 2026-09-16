from collections.abc import Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx

from cryptopulse.db.models import Article
from cryptopulse.ingestion.gdelt import ASSET_QUERY_TERMS
from cryptopulse.sentiment.cleaning import clean_article_text

SOURCE_NAME = "google_news"
ALLOWED_TIMESPANS = {"1d", "3d", "7d"}
SUMMARY_BOUNDARY_CHARACTERS = " -|:;"


def _required_text(item: ElementTree.Element, field: str, index: int) -> str:
    value = item.findtext(field)
    if value is None or not value.strip():
        raise ValueError(f"Missing Google News {field} at index {index}")
    return value.strip()


def _published_at(value: str, index: int) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid Google News pubDate at index {index}: {value}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Google News pubDate at index {index} has no timezone")
    return parsed.astimezone(UTC)


def clean_rss_summary(
    value: str | None,
    *,
    title: str,
    publisher: str = "",
) -> str | None:
    """Return useful RSS description text or None for empty duplicates."""
    if value is None:
        return None
    try:
        cleaned = clean_article_text(value)
    except ValueError:
        return None

    for prefix in (title, f"{title} - {publisher}" if publisher else ""):
        if prefix and cleaned.casefold().startswith(prefix.casefold()):
            cleaned = cleaned[len(prefix) :].lstrip(SUMMARY_BOUNDARY_CHARACTERS)
            break
    if publisher and cleaned.casefold().endswith(publisher.casefold()):
        cleaned = cleaned[: -len(publisher)].rstrip(SUMMARY_BOUNDARY_CHARACTERS)
    if not cleaned or cleaned.casefold() == title.casefold():
        return None
    return cleaned


def parse_rss_articles(
    content: bytes,
    query: str,
    retrieved_at: datetime,
) -> list[Article]:
    """Convert Google News RSS metadata into canonical article records."""
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("Google News retrieval time must include a timezone")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise RuntimeError("Google News returned invalid RSS XML") from error

    articles = []
    for index, item in enumerate(root.findall("./channel/item")):
        title = _required_text(item, "title", index)
        link = _required_text(item, "link", index)
        published = _required_text(item, "pubDate", index)
        publisher = (item.findtext("source") or "").strip()
        publisher_suffix = f" - {publisher}"
        if publisher and title.endswith(publisher_suffix):
            title = title[: -len(publisher_suffix)].strip()
        summary = clean_rss_summary(
            item.findtext("description"),
            title=title,
            publisher=publisher,
        )

        external_id = (item.findtext("guid") or link).strip()
        articles.append(
            Article(
                source=SOURCE_NAME,
                external_id=external_id,
                title=title,
                url=link,
                published_at=_published_at(published, index),
                retrieved_at=retrieved_at.astimezone(UTC),
                raw_query=query,
                summary=summary,
            )
        )
    return articles


class GoogleNewsRssClient:
    source_name = SOURCE_NAME

    def __init__(self, http_client: httpx.Client) -> None:
        self._http_client = http_client

    def fetch_recent_articles_for_assets(
        self,
        assets: Iterable[str],
        max_records: int = 100,
        timespan: str = "1d",
        retrieved_at: datetime | None = None,
    ) -> list[Article]:
        """Fetch recent English Google News RSS results for selected assets."""
        normalized_assets = tuple(dict.fromkeys(asset.upper() for asset in assets))
        if not normalized_assets:
            raise ValueError("At least one Google News asset is required")
        unsupported_assets = [
            asset for asset in normalized_assets if asset not in ASSET_QUERY_TERMS
        ]
        if unsupported_assets:
            raise ValueError(f"Unsupported Google News asset: {unsupported_assets[0]}")
        if not 1 <= max_records <= 250:
            raise ValueError("Google News max_records must be between 1 and 250")
        if timespan not in ALLOWED_TIMESPANS:
            raise ValueError("Google News timespan must be 1d, 3d, or 7d")

        query_terms = [
            term
            for asset in normalized_assets
            for term in ASSET_QUERY_TERMS[asset]
        ]
        query = f"({' OR '.join(query_terms)}) crypto when:{timespan}"
        response = self._http_client.get(
            "/rss/search",
            params={
                "q": query,
                "hl": "en-GB",
                "gl": "GB",
                "ceid": "GB:en",
            },
        )
        response.raise_for_status()
        articles = parse_rss_articles(
            response.content,
            query=query,
            retrieved_at=retrieved_at or datetime.now(UTC),
        )
        return articles[:max_records]

from datetime import UTC, datetime

import httpx

from cryptopulse.db.models import Article

ASSET_QUERIES = {
    "BTC": '(Bitcoin OR "BTC")',
    "ETH": '(Ethereum OR "ETH")',
    "SOL": '(Solana OR "SOL")',
}


def _parse_seen_date(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("GDELT seendate must be a string")
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"Invalid GDELT seendate: {value}") from error


def parse_articles(
    payload: object,
    query: str,
    retrieved_at: datetime,
) -> list[Article]:
    """Convert a GDELT response into article records."""
    if not isinstance(payload, dict):
        raise TypeError("GDELT response must be an object")
    article_items = payload.get("articles")
    if not isinstance(article_items, list):
        raise TypeError("GDELT response must contain an articles list")
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("GDELT retrieval time must include timezone information")

    articles = []
    for index, item in enumerate(article_items):
        if not isinstance(item, dict):
            raise TypeError(f"GDELT article at index {index} must be an object")

        title = item.get("title")
        url = item.get("url")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Missing GDELT title at index {index}")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Missing GDELT URL at index {index}")

        articles.append(
            Article(
                source="gdelt",
                external_id=None,
                title=title.strip(),
                url=url.strip(),
                published_at=_parse_seen_date(item.get("seendate")),
                retrieved_at=retrieved_at.astimezone(UTC),
                raw_query=query,
            )
        )

    return articles


class GdeltClient:
    def __init__(self, http_client: httpx.Client) -> None:
        self._http_client = http_client

    def fetch_recent_articles(
        self,
        asset: str,
        max_records: int = 50,
        timespan: str = "1d",
        retrieved_at: datetime | None = None,
    ) -> list[Article]:
        """Fetch recent English-language articles for one supported asset."""
        normalized_asset = asset.upper()
        if normalized_asset not in ASSET_QUERIES:
            raise ValueError(f"Unsupported GDELT asset: {asset}")
        if not 1 <= max_records <= 250:
            raise ValueError("GDELT max_records must be between 1 and 250")

        query = f"{ASSET_QUERIES[normalized_asset]} sourcelang:english"
        response = self._http_client.get(
            "/api/v2/doc/doc",
            params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": max_records,
                "timespan": timespan,
                "sort": "datedesc",
            },
        )
        response.raise_for_status()
        return parse_articles(
            response.json(),
            query=query,
            retrieved_at=retrieved_at or datetime.now(UTC),
        )

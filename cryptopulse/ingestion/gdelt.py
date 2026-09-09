import logging
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

import httpx

from cryptopulse.db.models import Article

LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

ASSET_QUERY_TERMS = {
    "BTC": ("Bitcoin", "BTC"),
    "ETH": ("Ethereum", "ETH"),
    "SOL": ("Solana", "SOL"),
}
ASSET_QUERIES = {
    asset: f"({' OR '.join(terms)})"
    for asset, terms in ASSET_QUERY_TERMS.items()
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
    def __init__(
        self,
        http_client: httpx.Client,
        *,
        max_retries: int = 4,
        retry_delay_seconds: float = 5.0,
        min_request_interval_seconds: float = 5.0,
        sleep_func: Callable[[float], None] = time.sleep,
        monotonic_func: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_retries < 0:
            raise ValueError("GDELT max_retries cannot be negative")
        if retry_delay_seconds < 0:
            raise ValueError("GDELT retry delay cannot be negative")
        if min_request_interval_seconds < 0:
            raise ValueError("GDELT request interval cannot be negative")

        self._http_client = http_client
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._min_request_interval_seconds = min_request_interval_seconds
        self._sleep = sleep_func
        self._monotonic = monotonic_func
        self._last_request_started_at: float | None = None

    def _wait_for_request_slot(self) -> None:
        if self._last_request_started_at is not None:
            elapsed = self._monotonic() - self._last_request_started_at
            wait_seconds = self._min_request_interval_seconds - elapsed
            if wait_seconds > 0:
                self._sleep(wait_seconds)
        self._last_request_started_at = self._monotonic()

    def _retry_delay(self, attempt: int) -> float:
        return self._retry_delay_seconds * (2**attempt)

    @staticmethod
    def _is_rate_limit_response(response: httpx.Response) -> bool:
        return (
            response.status_code == 429
            or "limit requests" in response.text.lower()
        )

    def _request_payload(self, params: dict[str, object]) -> object:
        for attempt in range(self._max_retries + 1):
            self._wait_for_request_slot()
            try:
                response = self._http_client.get("/api/v2/doc/doc", params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                if attempt == self._max_retries:
                    raise
                delay = self._retry_delay(attempt)
                LOGGER.warning(
                    "GDELT request failed with %s; retrying in %.1f seconds",
                    type(error).__name__,
                    delay,
                )
                self._sleep(delay)
                continue

            retryable = (
                response.status_code in RETRYABLE_STATUS_CODES
                or self._is_rate_limit_response(response)
            )
            if retryable and attempt < self._max_retries:
                delay = self._retry_delay(attempt)
                LOGGER.warning(
                    "GDELT returned HTTP %d; retrying in %.1f seconds",
                    response.status_code,
                    delay,
                )
                self._sleep(delay)
                continue

            response.raise_for_status()
            try:
                return response.json()
            except ValueError as error:
                response_preview = " ".join(response.text.split())[:200]
                detail = response_preview or "empty response"
                raise RuntimeError(
                    f"GDELT returned non-JSON data: {detail}"
                ) from error

        raise RuntimeError("GDELT request retry loop ended unexpectedly")

    def fetch_recent_articles(
        self,
        asset: str,
        max_records: int = 50,
        timespan: str = "1d",
        retrieved_at: datetime | None = None,
    ) -> list[Article]:
        """Fetch recent English-language articles for one supported asset."""
        return self.fetch_recent_articles_for_assets(
            (asset,),
            max_records=max_records,
            timespan=timespan,
            retrieved_at=retrieved_at,
        )

    def fetch_recent_articles_for_assets(
        self,
        assets: Iterable[str],
        max_records: int = 150,
        timespan: str = "1d",
        retrieved_at: datetime | None = None,
    ) -> list[Article]:
        """Fetch recent English-language articles with one combined asset query."""
        normalized_assets = tuple(dict.fromkeys(asset.upper() for asset in assets))
        if not normalized_assets:
            raise ValueError("At least one GDELT asset is required")
        unsupported_assets = [
            asset for asset in normalized_assets if asset not in ASSET_QUERY_TERMS
        ]
        if unsupported_assets:
            raise ValueError(f"Unsupported GDELT asset: {unsupported_assets[0]}")
        if not 1 <= max_records <= 250:
            raise ValueError("GDELT max_records must be between 1 and 250")

        query_terms = [
            term
            for asset in normalized_assets
            for term in ASSET_QUERY_TERMS[asset]
        ]
        query = f"({' OR '.join(query_terms)}) sourcelang:english"
        payload = self._request_payload(
            {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": max_records,
                "timespan": timespan,
                "sort": "datedesc",
            }
        )
        return parse_articles(
            payload,
            query=query,
            retrieved_at=retrieved_at or datetime.now(UTC),
        )

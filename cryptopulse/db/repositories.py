import re
import sqlite3
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cryptopulse.db.models import (
    Article,
    ArticleForClassification,
    InsertSummary,
    MarketCandle,
)
from cryptopulse.sentiment.models import ArticleClassification
from cryptopulse.sentiment.provider import ClassifierIdentity

CROSS_SOURCE_DEDUP_WINDOW = timedelta(hours=72)
NEAR_DUPLICATE_MIN_TOKENS = 5
NEAR_DUPLICATE_TOKEN_OVERLAP = 0.85
TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "ref", "source"}


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Database timestamps must include timezone information")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in TRACKING_QUERY_KEYS
        )
    )
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(("", hostname, path, query, ""))


def _normalized_title_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return set(TITLE_TOKEN_PATTERN.findall(normalized))


def _titles_are_near_duplicates(left: str, right: str) -> bool:
    return _title_token_sets_are_near_duplicates(
        _normalized_title_tokens(left),
        _normalized_title_tokens(right),
    )


def _title_token_sets_are_near_duplicates(
    left_tokens: set[str],
    right_tokens: set[str],
) -> bool:
    if left_tokens == right_tokens:
        return True
    if min(len(left_tokens), len(right_tokens)) < NEAR_DUPLICATE_MIN_TOKENS:
        return False
    overlap = len(left_tokens & right_tokens) / max(
        len(left_tokens),
        len(right_tokens),
    )
    return overlap >= NEAR_DUPLICATE_TOKEN_OVERLAP


def _stored_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _find_cross_source_duplicate(
    article: Article,
    stored_articles: list[sqlite3.Row],
) -> sqlite3.Row | None:
    article_url = _canonical_url(article.url)
    if article.published_at.tzinfo is None or article.published_at.utcoffset() is None:
        raise ValueError("Database timestamps must include timezone information")
    published_at = article.published_at.astimezone(UTC)
    for stored in stored_articles:
        if stored["source"] == article.source:
            continue
        if _canonical_url(stored["url"]) == article_url:
            return stored
        published_distance = abs(
            published_at - _stored_datetime(stored["published_at"])
        )
        if (
            published_distance <= CROSS_SOURCE_DEDUP_WINDOW
            and _titles_are_near_duplicates(article.title, stored["title"])
        ):
            return stored
    return None


def _article_survival_rank(article: sqlite3.Row) -> tuple[int, int, int]:
    return (
        int(article["successful_classifications"] > 0),
        int(bool((article["summary"] or "").strip())),
        -article["id"],
    )


def deduplicate_cross_source_articles(connection: sqlite3.Connection) -> int:
    """Merge previously stored cross-source duplicates conservatively."""
    stored_articles = connection.execute(
        """
        SELECT
            a.id, a.source, a.title, a.summary, a.url, a.published_at,
            COUNT(CASE WHEN c.status = 'succeeded' THEN 1 END)
                AS successful_classifications
        FROM articles AS a
        LEFT JOIN article_classifications AS c ON c.article_id = a.id
        GROUP BY a.id
        ORDER BY a.published_at, a.id
        """
    ).fetchall()
    parents = {article["id"]: article["id"] for article in stored_articles}
    fingerprints = {
        article["id"]: (
            _canonical_url(article["url"]),
            _normalized_title_tokens(article["title"]),
            _stored_datetime(article["published_at"]),
        )
        for article in stored_articles
    }

    def find(article_id: int) -> int:
        while parents[article_id] != article_id:
            parents[article_id] = parents[parents[article_id]]
            article_id = parents[article_id]
        return article_id

    def union(left_id: int, right_id: int) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parents[right_root] = left_root

    for index, left in enumerate(stored_articles):
        left_url, left_tokens, left_published = fingerprints[left["id"]]
        for right in stored_articles[index + 1 :]:
            if left["source"] == right["source"]:
                continue
            right_url, right_tokens, right_published = fingerprints[right["id"]]
            if left_url == right_url or (
                abs(left_published - right_published) <= CROSS_SOURCE_DEDUP_WINDOW
                and _title_token_sets_are_near_duplicates(left_tokens, right_tokens)
            ):
                union(left["id"], right["id"])

    groups: dict[int, list[sqlite3.Row]] = {}
    for article in stored_articles:
        groups.setdefault(find(article["id"]), []).append(article)

    duplicates_removed = 0
    for group in groups.values():
        if len(group) == 1:
            continue
        keeper = max(group, key=_article_survival_rank)
        if not (keeper["summary"] or "").strip():
            richer_article = next(
                (article for article in group if (article["summary"] or "").strip()),
                None,
            )
            if richer_article is not None:
                connection.execute(
                    "UPDATE articles SET summary = ? WHERE id = ?",
                    (richer_article["summary"], keeper["id"]),
                )
        for duplicate in group:
            if duplicate["id"] == keeper["id"]:
                continue
            connection.execute(
                "DELETE FROM articles WHERE id = ?",
                (duplicate["id"],),
            )
            duplicates_removed += 1
    return duplicates_removed


def insert_articles(
    connection: sqlite3.Connection,
    articles: Iterable[Article],
) -> InsertSummary:
    article_list = list(articles)
    stored_articles = connection.execute(
        """
        SELECT id, source, title, summary, url, published_at
        FROM articles
        """
    ).fetchall()
    articles_to_insert = []
    for article in article_list:
        duplicate = _find_cross_source_duplicate(article, stored_articles)
        if duplicate is None:
            articles_to_insert.append(article)
            continue
        if article.summary is not None and not (duplicate["summary"] or "").strip():
            connection.execute(
                "UPDATE articles SET summary = ? WHERE id = ?",
                (article.summary, duplicate["id"]),
            )

    rows = [
        (
            article.source,
            article.external_id,
            article.title,
            article.summary,
            article.url,
            _utc_text(article.published_at),
            _utc_text(article.retrieved_at),
            article.raw_query,
        )
        for article in articles_to_insert
    ]

    changes_before = connection.total_changes
    connection.executemany(
        """
        INSERT INTO articles (
            source, external_id, title, summary, url, published_at, retrieved_at,
            raw_query
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    inserted = connection.total_changes - changes_before
    for article in articles_to_insert:
        if article.summary is None:
            continue
        connection.execute(
            """
            UPDATE articles
            SET summary = ?
            WHERE source = ?
              AND COALESCE(TRIM(summary), '') = ''
              AND (
                  (? IS NOT NULL AND external_id = ?)
                  OR url = ?
              )
            """,
            (
                article.summary,
                article.source,
                article.external_id,
                article.external_id,
                article.url,
            ),
        )
    return InsertSummary(
        received=len(article_list),
        inserted=inserted,
        skipped=len(article_list) - inserted,
    )


def insert_market_candles(
    connection: sqlite3.Connection,
    candles: Iterable[MarketCandle],
) -> InsertSummary:
    candle_list = list(candles)
    rows = [
        (
            candle.asset,
            candle.symbol,
            _utc_text(candle.candle_timestamp),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.quote_volume,
            candle.trade_count,
        )
        for candle in candle_list
    ]

    changes_before = connection.total_changes
    connection.executemany(
        """
        INSERT INTO market_data (
            asset, symbol, candle_timestamp, open, high, low, close,
            volume, quote_volume, trade_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    inserted = connection.total_changes - changes_before
    return InsertSummary(
        received=len(candle_list),
        inserted=inserted,
        skipped=len(candle_list) - inserted,
    )


def start_ingestion_run(
    connection: sqlite3.Connection,
    source: str,
    started_at: datetime,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO ingestion_runs (source, started_at, status)
        VALUES (?, ?, 'running')
        """,
        (source, _utc_text(started_at)),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an ingestion run ID")
    return cursor.lastrowid


def finish_ingestion_run(
    connection: sqlite3.Connection,
    run_id: int,
    completed_at: datetime,
    status: Literal["succeeded", "failed"],
    summary: InsertSummary,
    error_message: str | None = None,
) -> None:
    cursor = connection.execute(
        """
        UPDATE ingestion_runs
        SET completed_at = ?,
            status = ?,
            records_received = ?,
            records_inserted = ?,
            records_skipped = ?,
            error_message = ?
        WHERE id = ?
        """,
        (
            _utc_text(completed_at),
            status,
            summary.received,
            summary.inserted,
            summary.skipped,
            error_message,
            run_id,
        ),
    )
    if cursor.rowcount != 1:
        raise ValueError(f"Ingestion run {run_id} does not exist")


def _find_classification(
    connection: sqlite3.Connection,
    article_id: int,
    identity: ClassifierIdentity,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT id, status
        FROM article_classifications
        WHERE article_id = ?
          AND provider = ?
          AND model = ?
          AND prompt_version = ?
        """,
        (
            article_id,
            identity.provider,
            identity.model,
            identity.prompt_version,
        ),
    ).fetchone()


def save_classification_success(
    connection: sqlite3.Connection,
    article_id: int,
    identity: ClassifierIdentity,
    input_text: str,
    classification: ArticleClassification,
    processed_at: datetime,
) -> int:
    """Store one validated classification without replacing an earlier success."""
    if not input_text.strip():
        raise ValueError("Classification input text cannot be blank")

    existing = _find_classification(connection, article_id, identity)
    if existing is not None and existing["status"] == "succeeded":
        return existing["id"]

    values = (
        input_text,
        int(classification.is_relevant),
        _utc_text(processed_at),
    )
    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO article_classifications (
                article_id, provider, model, prompt_version, input_text,
                status, is_relevant, processed_at, error_message
            ) VALUES (?, ?, ?, ?, ?, 'succeeded', ?, ?, NULL)
            """,
            (
                article_id,
                identity.provider,
                identity.model,
                identity.prompt_version,
                *values,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a classification ID")
        classification_id = cursor.lastrowid
    else:
        classification_id = existing["id"]
        connection.execute(
            """
            UPDATE article_classifications
            SET input_text = ?,
                status = 'succeeded',
                is_relevant = ?,
                processed_at = ?,
                error_message = NULL
            WHERE id = ?
            """,
            (*values, classification_id),
        )
        connection.execute(
            "DELETE FROM sentiment_results WHERE classification_id = ?",
            (classification_id,),
        )

    connection.executemany(
        """
        INSERT INTO sentiment_results (
            classification_id, asset, sentiment, sentiment_score,
            confidence, category, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                classification_id,
                result.asset.value,
                result.sentiment.value,
                result.sentiment_score,
                result.confidence,
                result.category.value,
                result.reason,
            )
            for result in classification.asset_sentiments
        ],
    )
    return classification_id


def save_classification_failure(
    connection: sqlite3.Connection,
    article_id: int,
    identity: ClassifierIdentity,
    input_text: str,
    error_message: str,
    processed_at: datetime,
) -> int:
    """Store a failed attempt unless a successful result already exists."""
    if not input_text.strip():
        raise ValueError("Classification input text cannot be blank")
    normalized_error = error_message.strip()[:1_000]
    if not normalized_error:
        raise ValueError("Classification failure message cannot be blank")

    existing = _find_classification(connection, article_id, identity)
    if existing is not None and existing["status"] == "succeeded":
        return existing["id"]

    values = (
        input_text,
        _utc_text(processed_at),
        normalized_error,
    )
    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO article_classifications (
                article_id, provider, model, prompt_version, input_text,
                status, is_relevant, processed_at, error_message
            ) VALUES (?, ?, ?, ?, ?, 'failed', NULL, ?, ?)
            """,
            (
                article_id,
                identity.provider,
                identity.model,
                identity.prompt_version,
                *values,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a classification ID")
        return cursor.lastrowid

    connection.execute(
        """
        UPDATE article_classifications
        SET input_text = ?,
            status = 'failed',
            is_relevant = NULL,
            processed_at = ?,
            error_message = ?
        WHERE id = ?
        """,
        (*values, existing["id"]),
    )
    return existing["id"]


def list_articles_for_classification(
    connection: sqlite3.Connection,
    identity: ClassifierIdentity,
    limit: int,
) -> list[ArticleForClassification]:
    """Return newest articles not yet successfully processed by this version."""
    if not 1 <= limit <= 200:
        raise ValueError("Classification limit must be between 1 and 200")

    rows = connection.execute(
        """
        SELECT a.id, a.title, a.summary
        FROM articles AS a
        LEFT JOIN article_classifications AS c
          ON c.article_id = a.id
         AND c.provider = ?
         AND c.model = ?
         AND c.prompt_version = ?
        WHERE c.id IS NULL OR c.status = 'failed'
        ORDER BY a.published_at DESC, a.id DESC
        LIMIT ?
        """,
        (
            identity.provider,
            identity.model,
            identity.prompt_version,
            limit,
        ),
    ).fetchall()
    return [
        ArticleForClassification(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
        )
        for row in rows
    ]

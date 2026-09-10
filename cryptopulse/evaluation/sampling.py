import csv
import random
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cryptopulse.sentiment.cleaning import find_candidate_assets
from cryptopulse.sentiment.models import CryptoAsset

ASSET_TARGET_SHARE = 0.15
MULTI_ASSET_TARGET_SHARE = 0.10
NO_CANDIDATE_TARGET_SHARE = 0.25

LABEL_FIELDS = (
    "sample_id",
    "article_id",
    "title",
    "published_at",
    "human_relevance",
    "btc_sentiment",
    "btc_category",
    "eth_sentiment",
    "eth_category",
    "sol_sentiment",
    "sol_category",
    "bnb_sentiment",
    "bnb_category",
    "human_notes",
)


@dataclass(frozen=True)
class EvaluationArticle:
    article_id: int
    title: str
    published_at: str
    candidate_assets: tuple[CryptoAsset, ...]


@dataclass(frozen=True)
class EvaluationSampleSummary:
    selected: int
    candidate_counts: dict[str, int]
    multi_asset: int
    no_candidate: int


def _add_unselected(
    selected: list[EvaluationArticle],
    selected_ids: set[int],
    candidates: Iterable[EvaluationArticle],
    count: int,
    sample_size: int,
) -> None:
    added = 0
    for article in candidates:
        if len(selected) >= sample_size or added >= count:
            return
        if article.article_id in selected_ids:
            continue
        selected.append(article)
        selected_ids.add(article.article_id)
        added += 1


def select_evaluation_sample(
    connection: sqlite3.Connection,
    sample_size: int = 100,
    seed: int = 42,
) -> list[EvaluationArticle]:
    """Select a reproducible sample with minimum coverage for every asset."""
    if not 1 <= sample_size <= 500:
        raise ValueError("Evaluation sample size must be between 1 and 500")

    rows = connection.execute(
        """
        SELECT id, title, published_at
        FROM articles
        ORDER BY published_at DESC, id DESC
        """
    ).fetchall()
    if len(rows) < sample_size:
        raise ValueError(
            f"Evaluation requires {sample_size} articles but only {len(rows)} are stored"
        )

    articles = [
        EvaluationArticle(
            article_id=row["id"],
            title=row["title"],
            published_at=row["published_at"],
            candidate_assets=find_candidate_assets(row["title"]),
        )
        for row in rows
    ]
    by_asset = {
        asset: [article for article in articles if asset in article.candidate_assets]
        for asset in CryptoAsset
    }
    minimum_per_asset = max(1, sample_size // 10)
    insufficient = {
        asset.value: len(candidates)
        for asset, candidates in by_asset.items()
        if len(candidates) < minimum_per_asset
    }
    if insufficient:
        available = ", ".join(
            f"{asset}={count}" for asset, count in insufficient.items()
        )
        raise ValueError(
            f"Evaluation needs at least {minimum_per_asset} candidate headlines per "
            f"asset; insufficient coverage: {available}"
        )

    rng = random.Random(seed)
    for candidates in by_asset.values():
        rng.shuffle(candidates)
    multi_asset = [article for article in articles if len(article.candidate_assets) > 1]
    no_candidate = [article for article in articles if not article.candidate_assets]
    rng.shuffle(multi_asset)
    rng.shuffle(no_candidate)

    selected: list[EvaluationArticle] = []
    selected_ids: set[int] = set()
    target_per_asset = max(minimum_per_asset, round(sample_size * ASSET_TARGET_SHARE))
    for asset in sorted(CryptoAsset, key=lambda item: len(by_asset[item])):
        current_count = sum(asset in article.candidate_assets for article in selected)
        _add_unselected(
            selected,
            selected_ids,
            by_asset[asset],
            max(0, target_per_asset - current_count),
            sample_size,
        )

    current_multi = sum(len(article.candidate_assets) > 1 for article in selected)
    _add_unselected(
        selected,
        selected_ids,
        multi_asset,
        max(0, round(sample_size * MULTI_ASSET_TARGET_SHARE) - current_multi),
        sample_size,
    )
    _add_unselected(
        selected,
        selected_ids,
        no_candidate,
        round(sample_size * NO_CANDIDATE_TARGET_SHARE),
        sample_size,
    )

    remaining = [
        article for article in articles if article.article_id not in selected_ids
    ]
    rng.shuffle(remaining)
    _add_unselected(
        selected,
        selected_ids,
        remaining,
        sample_size - len(selected),
        sample_size,
    )
    rng.shuffle(selected)
    return selected


def create_labeling_template(
    connection: sqlite3.Connection,
    output_path: Path,
    sample_size: int = 100,
    seed: int = 42,
) -> EvaluationSampleSummary:
    """Create a new human-labeling CSV without overwriting existing work."""
    sample = select_evaluation_sample(connection, sample_size=sample_size, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for sample_id, article in enumerate(sample, start=1):
            row = dict.fromkeys(LABEL_FIELDS, "")
            row.update(
                {
                    "sample_id": sample_id,
                    "article_id": article.article_id,
                    "title": article.title,
                    "published_at": article.published_at,
                }
            )
            writer.writerow(row)

    return EvaluationSampleSummary(
        selected=len(sample),
        candidate_counts={
            asset.value: sum(
                asset in article.candidate_assets for article in sample
            )
            for asset in CryptoAsset
        },
        multi_asset=sum(len(article.candidate_assets) > 1 for article in sample),
        no_candidate=sum(not article.candidate_assets for article in sample),
    )

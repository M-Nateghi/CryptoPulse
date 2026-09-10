import csv
from dataclasses import dataclass
from pathlib import Path

from cryptopulse.evaluation.sampling import LABEL_FIELDS
from cryptopulse.sentiment.models import (
    CryptoAsset,
    NewsCategory,
    SentimentLabel,
)

FINAL_RELEVANCE_VALUES = {"relevant", "irrelevant"}


class LabelValidationError(ValueError):
    """One or more human-label rows are incomplete or inconsistent."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        preview = "; ".join(errors[:10])
        if len(errors) > 10:
            preview = f"{preview}; and {len(errors) - 10} more"
        super().__init__(preview)


@dataclass(frozen=True)
class LabelValidationSummary:
    total_articles: int
    relevant: int
    irrelevant: int
    asset_labels: int
    asset_counts: dict[str, int]


def _positive_integer(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"row {row_number}: {field} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"row {row_number}: {field} must be positive")
    return parsed


def validate_labeling_csv(input_path: Path) -> LabelValidationSummary:
    """Validate a completed human-labeling sheet before model evaluation."""
    with input_path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != LABEL_FIELDS:
            raise LabelValidationError(
                ["CSV columns do not match the generated labeling template"]
            )
        rows = list(reader)

    if not rows:
        raise LabelValidationError(["The labeling CSV contains no article rows"])

    errors: list[str] = []
    sample_ids: set[int] = set()
    article_ids: set[int] = set()
    relevant_count = 0
    irrelevant_count = 0
    asset_counts = {asset.value: 0 for asset in CryptoAsset}

    for row_number, row in enumerate(rows, start=2):
        try:
            sample_id = _positive_integer(row["sample_id"], "sample_id", row_number)
            article_id = _positive_integer(row["article_id"], "article_id", row_number)
        except ValueError as error:
            errors.append(str(error))
            continue

        if sample_id in sample_ids:
            errors.append(f"row {row_number}: duplicate sample_id {sample_id}")
        if article_id in article_ids:
            errors.append(f"row {row_number}: duplicate article_id {article_id}")
        sample_ids.add(sample_id)
        article_ids.add(article_id)

        relevance = row["human_relevance"].strip()
        if relevance not in FINAL_RELEVANCE_VALUES:
            errors.append(
                f"row {row_number}: human_relevance must be relevant or irrelevant"
            )

        labeled_assets = 0
        for asset in CryptoAsset:
            prefix = asset.value.lower()
            sentiment = row[f"{prefix}_sentiment"].strip()
            category = row[f"{prefix}_category"].strip()
            if bool(sentiment) != bool(category):
                errors.append(
                    f"row {row_number}: {asset.value} sentiment and category must "
                    "both be filled or both be blank"
                )
                continue
            if not sentiment:
                continue

            labeled_assets += 1
            asset_counts[asset.value] += 1
            try:
                SentimentLabel(sentiment)
            except ValueError:
                errors.append(
                    f"row {row_number}: invalid {asset.value} sentiment {sentiment!r}"
                )
            try:
                NewsCategory(category)
            except ValueError:
                errors.append(
                    f"row {row_number}: invalid {asset.value} category {category!r}"
                )

        if relevance == "relevant":
            relevant_count += 1
            if labeled_assets == 0:
                errors.append(
                    f"row {row_number}: relevant row needs at least one asset label"
                )
        elif relevance == "irrelevant":
            irrelevant_count += 1
            if labeled_assets:
                errors.append(
                    f"row {row_number}: irrelevant row cannot contain asset labels"
                )

    if errors:
        raise LabelValidationError(errors)

    return LabelValidationSummary(
        total_articles=len(rows),
        relevant=relevant_count,
        irrelevant=irrelevant_count,
        asset_labels=sum(asset_counts.values()),
        asset_counts=asset_counts,
    )

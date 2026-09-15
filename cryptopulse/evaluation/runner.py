import csv
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptopulse.evaluation.assisted_labels import (
    ASSISTED_LABEL_FIELDS,
    HumanRelevanceRow,
    load_human_relevance_rows,
)
from cryptopulse.evaluation.baselines import SentimentPrediction
from cryptopulse.evaluation.openai_predictions import RelevancePrediction
from cryptopulse.sentiment.models import SentimentLabel
from cryptopulse.sentiment.provider import ClassifierIdentity

METRIC_FIELDS = (
    "task",
    "reference",
    "model",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "support",
    "interpretation",
)
CONFUSION_FIELDS = ("task", "model", "reference", "prediction", "count")
ERROR_FIELDS = (
    "sample_id",
    "article_id",
    "title",
    "asset",
    "task",
    "model",
    "reference",
    "prediction",
)


class RelevancePredictor(Protocol):
    @property
    def identity(self) -> ClassifierIdentity: ...

    def predict(
        self,
        rows: Sequence[HumanRelevanceRow],
    ) -> list[RelevancePrediction]: ...


class BaselinePredictor(Protocol):
    name: str

    def predict(self, headlines: Sequence[str]) -> list[SentimentPrediction]: ...


@dataclass(frozen=True)
class EvaluationSummary:
    human_articles: int
    assisted_asset_labels: int
    metrics: int
    errors: int
    output_directory: Path


def classification_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    labels: Sequence[str],
) -> tuple[dict[str, float], Counter[tuple[str, str]]]:
    if not references or len(references) != len(predictions):
        raise ValueError("Metric inputs must be non-empty and have equal lengths")
    confusion = Counter(zip(references, predictions, strict=True))
    correct = sum(reference == prediction for reference, prediction in zip(
        references, predictions, strict=True
    ))
    per_label: list[tuple[float, float, float, int]] = []
    for label in labels:
        true_positive = confusion[(label, label)]
        false_positive = sum(
            count for (reference, prediction), count in confusion.items()
            if prediction == label and reference != label
        )
        false_negative = sum(
            count for (reference, prediction), count in confusion.items()
            if reference == label and prediction != label
        )
        support = sum(
            count for (reference, _), count in confusion.items() if reference == label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label.append((precision, recall, f1, support))

    total = len(references)
    metrics = {
        "accuracy": correct / total,
        "macro_precision": sum(item[0] for item in per_label) / len(labels),
        "macro_recall": sum(item[1] for item in per_label) / len(labels),
        "macro_f1": sum(item[2] for item in per_label) / len(labels),
        "weighted_f1": sum(item[2] * item[3] for item in per_label) / total,
    }
    return metrics, confusion


def _read_assisted_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != ASSISTED_LABEL_FIELDS:
            raise ValueError("AI-assisted label columns do not match the expected schema")
        rows = list(reader)
    if not rows:
        raise ValueError("AI-assisted label file contains no asset labels")
    return rows


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _validated_relevance_predictions(
    rows: Sequence[HumanRelevanceRow],
    predictions: Sequence[RelevancePrediction],
) -> dict[int, RelevancePrediction]:
    expected_ids = {row.sample_id for row in rows}
    actual_ids = [prediction.sample_id for prediction in predictions]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError(
            "OpenAI relevance predictions must return every sample ID exactly once"
        )
    return {prediction.sample_id: prediction for prediction in predictions}


def run_hybrid_evaluation(
    human_path: Path,
    assisted_path: Path,
    output_directory: Path,
    relevance_predictor: RelevancePredictor,
    baseline_predictors: Sequence[BaselinePredictor],
    *,
    force: bool = False,
    batch_size: int = 20,
) -> EvaluationSummary:
    """Evaluate human relevance and AI-assisted sentiment as separate tasks."""
    if not 1 <= batch_size <= 25:
        raise ValueError("Evaluation batch size must be between 1 and 25")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "metrics": output_directory / "model_metrics.csv",
        "confusion": output_directory / "confusion_matrices.csv",
        "errors": output_directory / "model_errors.csv",
        "relevance": output_directory / "openai_relevance_predictions.csv",
        "sentiment": output_directory / "baseline_sentiment_predictions.csv",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Evaluation output already exists: {existing[0]}")

    human_rows = load_human_relevance_rows(human_path)
    assisted_rows = _read_assisted_labels(assisted_path)
    relevant_ids = {
        row.sample_id for row in human_rows if row.human_relevance == "relevant"
    }
    assisted_ids = {int(row["sample_id"]) for row in assisted_rows}
    if assisted_ids != relevant_ids:
        raise ValueError(
            "AI-assisted labels must cover every human-relevant sample and no others"
        )

    all_relevance_predictions: list[RelevancePrediction] = []
    for start in range(0, len(human_rows), batch_size):
        batch = human_rows[start : start + batch_size]
        batch_predictions = relevance_predictor.predict(batch)
        _validated_relevance_predictions(batch, batch_predictions)
        all_relevance_predictions.extend(batch_predictions)
    relevance_by_id = _validated_relevance_predictions(
        human_rows, all_relevance_predictions
    )

    reference_relevance = [row.human_relevance for row in human_rows]
    predicted_relevance = [
        "relevant" if relevance_by_id[row.sample_id].is_relevant else "irrelevant"
        for row in human_rows
    ]
    relevance_metrics, relevance_confusion = classification_metrics(
        reference_relevance,
        predicted_relevance,
        ("relevant", "irrelevant"),
    )

    metric_rows = [
        {
            "task": "relevance",
            "reference": "human",
            "model": f"OpenAI ({relevance_predictor.identity.model})",
            **{key: f"{value:.4f}" for key, value in relevance_metrics.items()},
            "support": len(human_rows),
            "interpretation": "Accuracy against independent human relevance labels.",
        }
    ]
    confusion_rows = [
        {
            "task": "relevance",
            "model": f"OpenAI ({relevance_predictor.identity.model})",
            "reference": reference,
            "prediction": prediction,
            "count": count,
        }
        for (reference, prediction), count in sorted(relevance_confusion.items())
    ]
    error_rows = [
        {
            "sample_id": row.sample_id,
            "article_id": row.article_id,
            "title": row.title,
            "asset": "",
            "task": "relevance",
            "model": f"OpenAI ({relevance_predictor.identity.model})",
            "reference": reference,
            "prediction": prediction,
        }
        for row, reference, prediction in zip(
            human_rows, reference_relevance, predicted_relevance, strict=True
        )
        if reference != prediction
    ]
    relevance_output_rows = [
        {
            "sample_id": row.sample_id,
            "article_id": row.article_id,
            "title": row.title,
            "human_relevance": row.human_relevance,
            "predicted_relevance": (
                "relevant" if relevance_by_id[row.sample_id].is_relevant else "irrelevant"
            ),
            "provider": relevance_predictor.identity.provider,
            "model": relevance_predictor.identity.model,
            "prompt_version": relevance_predictor.identity.prompt_version,
        }
        for row in human_rows
    ]

    title_by_id = {
        row.sample_id: row.title for row in human_rows if row.sample_id in assisted_ids
    }
    unique_ids = sorted(assisted_ids)
    headlines = [title_by_id[sample_id] for sample_id in unique_ids]
    sentiment_output_rows: list[dict] = []
    sentiment_labels = tuple(label.value for label in SentimentLabel)
    for predictor in baseline_predictors:
        baseline_results = predictor.predict(headlines)
        if len(baseline_results) != len(unique_ids):
            raise ValueError(f"{predictor.name} returned an unexpected result count")
        result_by_id = dict(zip(unique_ids, baseline_results, strict=True))
        references = [row["assisted_sentiment"] for row in assisted_rows]
        predictions = [
            result_by_id[int(row["sample_id"])].label.value for row in assisted_rows
        ]
        metrics, confusion = classification_metrics(
            references,
            predictions,
            sentiment_labels,
        )
        metric_rows.append(
            {
                "task": "sentiment_agreement",
                "reference": "OpenAI-assisted",
                "model": predictor.name,
                **{key: f"{value:.4f}" for key, value in metrics.items()},
                "support": len(assisted_rows),
                "interpretation": (
                    "Agreement with AI-assisted labels; not human-grounded accuracy."
                ),
            }
        )
        confusion_rows.extend(
            {
                "task": "sentiment_agreement",
                "model": predictor.name,
                "reference": reference,
                "prediction": prediction,
                "count": count,
            }
            for (reference, prediction), count in sorted(confusion.items())
        )
        for row, prediction in zip(assisted_rows, predictions, strict=True):
            result = result_by_id[int(row["sample_id"])]
            sentiment_output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "asset": row["asset"],
                    "reference_sentiment": row["assisted_sentiment"],
                    "reference_source": "OpenAI-assisted",
                    "model": predictor.name,
                    "predicted_sentiment": prediction,
                    "model_score": result.score,
                }
            )
            if row["assisted_sentiment"] != prediction:
                error_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "article_id": row["article_id"],
                        "title": row["title"],
                        "asset": row["asset"],
                        "task": "sentiment_agreement",
                        "model": predictor.name,
                        "reference": row["assisted_sentiment"],
                        "prediction": prediction,
                    }
                )

    _write_csv(output_paths["metrics"], METRIC_FIELDS, metric_rows)
    _write_csv(output_paths["confusion"], CONFUSION_FIELDS, confusion_rows)
    _write_csv(output_paths["errors"], ERROR_FIELDS, error_rows)
    _write_csv(
        output_paths["relevance"],
        tuple(relevance_output_rows[0]),
        relevance_output_rows,
    )
    _write_csv(
        output_paths["sentiment"],
        tuple(sentiment_output_rows[0]),
        sentiment_output_rows,
    )
    return EvaluationSummary(
        human_articles=len(human_rows),
        assisted_asset_labels=len(assisted_rows),
        metrics=len(metric_rows),
        errors=len(error_rows),
        output_directory=output_directory,
    )

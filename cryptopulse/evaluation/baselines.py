from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cryptopulse.sentiment.models import SentimentLabel

VADER_MODEL_NAME = "VADER 3.3.2"
FINBERT_MODEL_NAME = "ProsusAI/finbert"


@dataclass(frozen=True)
class SentimentPrediction:
    label: SentimentLabel
    score: float


def vader_label(compound_score: float) -> SentimentLabel:
    if compound_score >= 0.05:
        return SentimentLabel.BULLISH
    if compound_score <= -0.05:
        return SentimentLabel.BEARISH
    return SentimentLabel.NEUTRAL


class VaderPredictor:
    name = VADER_MODEL_NAME

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as error:
            raise RuntimeError(
                "VADER is not installed. Run pip install -r requirements-evaluation.txt"
            ) from error
        self._analyzer = SentimentIntensityAnalyzer()

    def predict(self, headlines: Sequence[str]) -> list[SentimentPrediction]:
        predictions = []
        for headline in headlines:
            compound = float(self._analyzer.polarity_scores(headline)["compound"])
            predictions.append(
                SentimentPrediction(label=vader_label(compound), score=compound)
            )
        return predictions


class FinBertPredictor:
    name = FINBERT_MODEL_NAME

    def __init__(
        self,
        classifier: Any | None = None,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._classifier = classifier
        if classifier is not None:
            self._tokenizer = None
            self._model = None
            self._torch = None
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "FinBERT is not installed. Run pip install -r "
                "requirements-evaluation.txt"
            ) from error
        self._torch = torch
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    FINBERT_MODEL_NAME, local_files_only=True
                )
            except OSError:
                tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        if model is None:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    FINBERT_MODEL_NAME, local_files_only=True
                )
            except OSError:
                model = AutoModelForSequenceClassification.from_pretrained(
                    FINBERT_MODEL_NAME
                )
        self._tokenizer = tokenizer
        self._model = model
        self._model.eval()

    def predict(self, headlines: Sequence[str]) -> list[SentimentPrediction]:
        mapping = {
            "positive": SentimentLabel.BULLISH,
            "neutral": SentimentLabel.NEUTRAL,
            "negative": SentimentLabel.BEARISH,
        }
        if self._classifier is not None:
            outputs = self._classifier(
                list(headlines),
                truncation=True,
                batch_size=16,
            )
            return [
                SentimentPrediction(
                    label=mapping[str(output["label"]).lower()],
                    score=float(output["score"]),
                )
                for output in outputs
            ]

        encoded = self._tokenizer(
            list(headlines),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits
            probabilities = self._torch.softmax(logits, dim=-1)
        scores, indices = probabilities.max(dim=-1)
        id_to_label = self._model.config.id2label
        predictions = []
        for index, score in zip(indices.tolist(), scores.tolist(), strict=True):
            label = str(id_to_label[index]).lower()
            if label not in mapping:
                raise ValueError(f"Unexpected FinBERT label: {label}")
            predictions.append(
                SentimentPrediction(label=mapping[label], score=float(score))
            )
        return predictions

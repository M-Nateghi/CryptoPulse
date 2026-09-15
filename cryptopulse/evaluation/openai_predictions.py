import json
from collections.abc import Sequence

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cryptopulse.evaluation.assisted_labels import HumanRelevanceRow
from cryptopulse.sentiment.models import AssetSentiment
from cryptopulse.sentiment.openai_provider import CLASSIFICATION_INSTRUCTIONS
from cryptopulse.sentiment.provider import (
    ClassificationProviderError,
    ClassifierIdentity,
)

EVALUATION_PROMPT_VERSION = "openai-evaluation-batch-v1"


class RelevancePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: int = Field(gt=0)
    is_relevant: bool = Field(strict=True)
    asset_sentiments: list[AssetSentiment] = Field(max_length=4)

    @model_validator(mode="after")
    def validate_assets(self) -> "RelevancePrediction":
        if self.is_relevant != bool(self.asset_sentiments):
            raise ValueError(
                "is_relevant must be true exactly when asset_sentiments is non-empty"
            )
        assets = [result.asset for result in self.asset_sentiments]
        if len(assets) != len(set(assets)):
            raise ValueError("asset_sentiments cannot contain duplicate assets")
        return self


class RelevancePredictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[RelevancePrediction] = Field(min_length=1, max_length=25)


class OpenAIRelevancePredictor:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._identity = ClassifierIdentity(
            provider="openai",
            model=model,
            prompt_version=EVALUATION_PROMPT_VERSION,
        )

    @property
    def identity(self) -> ClassifierIdentity:
        return self._identity

    def predict(
        self,
        rows: Sequence[HumanRelevanceRow],
    ) -> list[RelevancePrediction]:
        payload = [
            {"sample_id": row.sample_id, "headline": row.title} for row in rows
        ]
        instructions = (
            f"{CLASSIFICATION_INSTRUCTIONS}\n\nReturn every supplied sample_id exactly "
            "once. Keep sample IDs unchanged."
        )
        try:
            response = self._client.responses.parse(
                model=self.identity.model,
                instructions=instructions,
                input=json.dumps({"headlines": payload}, ensure_ascii=False),
                text_format=RelevancePredictionBatch,
                max_output_tokens=8_000,
                store=False,
            )
        except OpenAIError as error:
            raise ClassificationProviderError(
                f"OpenAI evaluation prediction failed: {type(error).__name__}: {error}"
            ) from error
        parsed = response.output_parsed
        if not isinstance(parsed, RelevancePredictionBatch):
            raise ClassificationProviderError(
                "OpenAI evaluation prediction returned no parsed result"
            )
        return parsed.predictions

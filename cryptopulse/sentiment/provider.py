from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from cryptopulse.sentiment.cleaning import PreparedArticle
from cryptopulse.sentiment.models import ArticleClassification


class ClassificationProviderError(RuntimeError):
    """A provider request failed before producing a usable classification."""


class ClassificationResponseError(ClassificationProviderError):
    """A provider response did not contain the required classification."""


class ClassifierIdentity(BaseModel):
    """Reproducibility metadata owned by a classifier implementation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    prompt_version: str = Field(
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )


@runtime_checkable
class SentimentClassifier(Protocol):
    """Provider-neutral capability required by the sentiment workflow."""

    @property
    def identity(self) -> ClassifierIdentity: ...

    def classify(self, article: PreparedArticle) -> ArticleClassification: ...

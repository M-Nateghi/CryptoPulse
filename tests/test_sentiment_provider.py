import pytest
from pydantic import ValidationError

from cryptopulse.sentiment import (
    ArticleClassification,
    ClassifierIdentity,
    PreparedArticle,
    SentimentClassifier,
)


class ExampleClassifier:
    @property
    def identity(self):
        return ClassifierIdentity(
            provider="example",
            model="example-model-1",
            prompt_version="v1",
        )

    def classify(self, article):
        return ArticleClassification(
            is_relevant=False,
            asset_sentiments=[],
        )


def test_classifier_identity_is_normalized_and_immutable():
    identity = ClassifierIdentity(
        provider="  example  ",
        model="  example-model-1  ",
        prompt_version="v1.0",
    )

    assert identity.provider == "example"
    assert identity.model == "example-model-1"
    assert identity.prompt_version == "v1.0"
    with pytest.raises(ValidationError):
        identity.model = "different-model"


@pytest.mark.parametrize("field", ["provider", "model", "prompt_version"])
def test_classifier_identity_rejects_blank_required_fields(field):
    values = {
        "provider": "example",
        "model": "example-model-1",
        "prompt_version": "v1",
    }
    values[field] = "   "

    with pytest.raises(ValidationError):
        ClassifierIdentity(**values)


def test_classifier_identity_rejects_unsafe_prompt_version():
    with pytest.raises(ValidationError):
        ClassifierIdentity(
            provider="example",
            model="example-model-1",
            prompt_version="version 1",
        )


def test_classifier_identity_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ClassifierIdentity(
            provider="example",
            model="example-model-1",
            prompt_version="v1",
            api_key="must-not-be-stored",
        )


def test_protocol_accepts_any_object_with_the_required_shape():
    classifier = ExampleClassifier()
    article = PreparedArticle(text="General market update", candidate_assets=())

    assert isinstance(classifier, SentimentClassifier)
    assert classifier.identity.provider == "example"
    assert classifier.classify(article) == ArticleClassification(
        is_relevant=False,
        asset_sentiments=[],
    )


def test_protocol_rejects_an_object_without_classify():
    assert not isinstance(object(), SentimentClassifier)

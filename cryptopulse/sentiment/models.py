from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CryptoAsset(StrEnum):
    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"


class SentimentLabel(StrEnum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class NewsCategory(StrEnum):
    ETF_FLOWS = "etf_flows"
    INSTITUTIONAL_ADOPTION = "institutional_adoption"
    REGULATION = "regulation"
    TECHNOLOGY = "technology"
    SECURITY_HACKS = "security_hacks"
    MARKET_MOVEMENT = "market_movement"
    MACRO = "macro"
    EXCHANGE_ACTIVITY = "exchange_activity"
    OTHER = "other"


class AssetSentiment(BaseModel):
    """Validated sentiment evidence for one affected cryptocurrency."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    asset: CryptoAsset
    sentiment: SentimentLabel
    sentiment_score: float = Field(ge=-1, le=1, strict=True)
    confidence: float = Field(ge=0, le=1, strict=True)
    category: NewsCategory
    reason: str = Field(min_length=1, max_length=500)


class ArticleClassification(BaseModel):
    """Complete structured output expected from the sentiment classifier."""

    model_config = ConfigDict(extra="forbid")

    is_relevant: bool = Field(strict=True)
    asset_sentiments: list[AssetSentiment] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_relevance_and_assets(self) -> "ArticleClassification":
        has_asset_sentiments = bool(self.asset_sentiments)
        if self.is_relevant != has_asset_sentiments:
            raise ValueError(
                "is_relevant must be true exactly when asset_sentiments is non-empty"
            )

        assets = [result.asset for result in self.asset_sentiments]
        if len(assets) != len(set(assets)):
            raise ValueError("asset_sentiments cannot contain duplicate assets")

        return self

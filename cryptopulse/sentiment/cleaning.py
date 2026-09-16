import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser

from cryptopulse.sentiment.models import CryptoAsset

WHITESPACE_PATTERN = re.compile(r"\s+")
MAX_SUMMARY_CHARACTERS = 1_500

ASSET_NAME_PATTERNS = {
    CryptoAsset.BTC: re.compile(r"\bbitcoin\b", re.IGNORECASE),
    CryptoAsset.ETH: re.compile(r"\b(?:ethereum|ether)\b", re.IGNORECASE),
    CryptoAsset.SOL: re.compile(r"\bsolana\b", re.IGNORECASE),
    CryptoAsset.BNB: re.compile(r"\b(?:binance coin|bnb chain)\b", re.IGNORECASE),
}

ASSET_TICKER_PATTERNS = {
    asset: re.compile(rf"(?<![A-Za-z0-9])\$?{asset.value}(?![A-Za-z0-9])")
    for asset in CryptoAsset
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


@dataclass(frozen=True)
class PreparedArticle:
    title: str
    candidate_assets: tuple[CryptoAsset, ...]
    summary: str | None = None

    @property
    def text(self) -> str:
        if self.summary is None:
            return self.title
        return f"Headline: {self.title}\nSummary: {self.summary}"


def clean_article_text(text: str) -> str:
    """Normalize one article title or text snippet for classification."""
    if not isinstance(text, str):
        raise TypeError("Article text must be a string")

    extractor = _TextExtractor()
    extractor.feed(unicodedata.normalize("NFKC", text))
    extractor.close()
    extracted_text = " ".join(extractor.parts)

    cleaned_characters = (
        " "
        if character == "\ufffd"
        or character.isspace()
        or unicodedata.category(character).startswith("C")
        else character
        for character in extracted_text
    )
    cleaned_text = WHITESPACE_PATTERN.sub(" ", "".join(cleaned_characters)).strip()
    if not cleaned_text:
        raise ValueError("Article text is empty after cleaning")
    return cleaned_text


def _find_candidates_in_clean_text(text: str) -> tuple[CryptoAsset, ...]:
    return tuple(
        asset
        for asset in CryptoAsset
        if ASSET_NAME_PATTERNS[asset].search(text)
        or ASSET_TICKER_PATTERNS[asset].search(text)
    )


def find_candidate_assets(text: str) -> tuple[CryptoAsset, ...]:
    """Return possible in-scope assets mentioned in article text."""
    return _find_candidates_in_clean_text(clean_article_text(text))


def _prepare_summary(summary: str | None, title: str) -> str | None:
    if summary is None:
        return None
    if not isinstance(summary, str):
        raise TypeError("Article summary must be a string or None")
    try:
        cleaned_summary = clean_article_text(summary)
    except ValueError:
        return None
    if cleaned_summary.casefold() == title.casefold():
        return None
    if len(cleaned_summary) <= MAX_SUMMARY_CHARACTERS:
        return cleaned_summary
    shortened = cleaned_summary[:MAX_SUMMARY_CHARACTERS].rsplit(" ", 1)[0].strip()
    return shortened or cleaned_summary[:MAX_SUMMARY_CHARACTERS]


def prepare_article_text(title: str, summary: str | None = None) -> PreparedArticle:
    """Clean an article headline and optional summary for classification."""
    cleaned_title = clean_article_text(title)
    cleaned_summary = _prepare_summary(summary, cleaned_title)
    candidate_text = " ".join(
        part for part in (cleaned_title, cleaned_summary) if part is not None
    )
    return PreparedArticle(
        title=cleaned_title,
        summary=cleaned_summary,
        candidate_assets=_find_candidates_in_clean_text(candidate_text),
    )

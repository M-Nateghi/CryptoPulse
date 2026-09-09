import pytest

from cryptopulse.sentiment.cleaning import (
    clean_article_text,
    find_candidate_assets,
    prepare_article_text,
)
from cryptopulse.sentiment.models import CryptoAsset


def test_clean_article_text_normalizes_markup_entities_and_whitespace():
    text = "  <strong>Bitcoin</strong>\t&amp;\n Ethereum rally  "

    assert clean_article_text(text) == "Bitcoin & Ethereum rally"


def test_clean_article_text_removes_control_and_replacement_characters():
    text = "Bitcoin\x00 price � rises"

    assert clean_article_text(text) == "Bitcoin price rises"


@pytest.mark.parametrize("value", ["", "   ", "<span></span>", "�"])
def test_clean_article_text_rejects_empty_results(value):
    with pytest.raises(ValueError, match="empty after cleaning"):
        clean_article_text(value)


def test_clean_article_text_rejects_non_string_values():
    with pytest.raises(TypeError, match="must be a string"):
        clean_article_text(None)


def test_candidate_assets_include_names_and_uppercase_tickers():
    candidates = find_candidate_assets(
        "Bitcoin gains while ETH and $SOL react to the announcement"
    )

    assert candidates == (CryptoAsset.BTC, CryptoAsset.ETH, CryptoAsset.SOL)


def test_candidate_assets_are_unique_and_have_stable_order():
    candidates = find_candidate_assets("ETH, Ethereum, BTC and Bitcoin")

    assert candidates == (CryptoAsset.BTC, CryptoAsset.ETH)


def test_candidate_assets_do_not_match_inside_larger_words():
    candidates = find_candidate_assets("Bethany discussed solvent technology")

    assert candidates == ()


def test_lowercase_short_words_are_not_treated_as_tickers():
    candidates = find_candidate_assets("A singer performs sol while using eth tools")

    assert candidates == ()


def test_prepare_article_text_returns_clean_text_and_candidates():
    prepared = prepare_article_text(" <b>Solana</b>  upgrade &amp; BTC reaction ")

    assert prepared.text == "Solana upgrade & BTC reaction"
    assert prepared.candidate_assets == (CryptoAsset.BTC, CryptoAsset.SOL)


def test_prepare_article_text_allows_no_candidate_asset():
    prepared = prepare_article_text("Streaming prices continue to rise")

    assert prepared.candidate_assets == ()

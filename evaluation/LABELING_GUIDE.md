# CryptoPulse Human Relevance Labelling Guide

## Purpose

This file records the independent human reference used to evaluate whether the
OpenAI classifier correctly identifies relevant cryptocurrency headlines. The
asset sentiment and category task is AI-assisted and is stored separately in
`ai_assisted_labels_v1.csv`.

Label only what the headline itself supports. Do not open the article, search for
outside context, or view model predictions before deciding.

## Supported assets

- BTC: Bitcoin
- ETH: Ethereum or Ether
- SOL: Solana
- BNB: BNB, BNB Chain, or Binance Coin

A mention of the Binance exchange does not automatically mean the article
affects BNB.

## What to fill

Do not change `sample_id`, `article_id`, `title`, or `published_at`.

Fill only `human_relevance` with one of:

- `relevant`: at least one supported asset is genuinely affected.
- `irrelevant`: none of the supported assets is genuinely affected.

Leave all asset sentiment and category columns blank. They remain in the stable
template for backwards compatibility, but the project deliberately prevents AI
outputs from being written into the human reference file. `human_notes` is
optional and may record a difficult decision.

## Decision process

1. Read the headline without looking at any model output.
2. Decide whether BTC, ETH, SOL, or BNB is genuinely affected.
3. Enter `relevant` or `irrelevant` in `human_relevance`.
4. Add a human note only when useful.

## Important rules

- Mention does not always mean impact. A list of token prices may still be relevant,
  but its directional sentiment is a separate question.
- General cryptocurrency news is not automatically relevant to every asset.
- Judge only the supported assets and only evidence present in the headline.
- Do not change labels after seeing model predictions unless the change is recorded
  as a separate adjudication.

## Validation

Run:

```powershell
python -m cryptopulse.cli validate-evaluation
```

The completed file must contain only `relevant` or `irrelevant`, with no blank
rows and no machine-generated asset labels. AI-assisted sentiment is generated
separately with `label-evaluation` and records its provider, model, prompt
version, confidence, reason, and generation time.

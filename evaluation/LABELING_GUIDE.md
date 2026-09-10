# CryptoPulse Human Labelling Guide

## Purpose

This guide defines the human reference labels used to evaluate VADER, FinBERT,
and the OpenAI classifier. These labels are ground truth for this project, so
they must be based on your judgment rather than copied from a model.

Label only what the headline itself supports. Do not open the article or use
outside market knowledge. This keeps the human task aligned with the current
classifier, which also receives only the headline.

## Supported assets

- BTC: Bitcoin
- ETH: Ethereum or Ether
- SOL: Solana
- BNB: BNB, BNB Chain, or Binance Coin

A mention of the Binance exchange does not automatically mean the article
affects BNB.

## CSV fields

Do not change `sample_id`, `article_id`, `title`, or `published_at`.

Fill `human_relevance` with one of:

- `relevant`: at least one supported asset is genuinely affected.
- `irrelevant`: none of the supported assets is genuinely affected.
- `review`: you cannot decide confidently and want to revisit the row.

For each affected asset, fill its sentiment and category columns. Leave both
columns blank when that asset is not affected.

Allowed sentiment values:

- `bullish`: the headline has a positive directional implication for the asset.
- `neutral`: the implication is balanced, unclear, or purely descriptive.
- `bearish`: the headline has a negative directional implication for the asset.

Allowed categories:

- `etf_flows`
- `institutional_adoption`
- `regulation`
- `technology`
- `security_hacks`
- `market_movement`
- `macro`
- `exchange_activity`
- `other`

Use `human_notes` to explain ambiguity, mixed evidence, or a decision that may
need discussion.

## Decision process

1. Read the headline once without looking at any model output.
2. Decide whether BTC, ETH, SOL, or BNB is genuinely affected.
3. If no supported asset is affected, enter `irrelevant` and leave all asset
   fields blank.
4. If one or more assets are affected, enter `relevant` and label each affected
   asset independently.
5. Choose the closest category for each affected asset.
6. Use `review` and add a note when the headline does not provide enough evidence.

## Important rules

- Mention does not always mean impact. A list of token prices may be neutral.
- General cryptocurrency news is not automatically relevant to every asset.
- Label each asset independently in a multi-asset headline.
- Judge likely directional impact, not whether the writing sounds emotional.
- Do not infer facts that are absent from the headline.
- Do not change earlier labels after seeing model predictions unless you record
  and justify an adjudication.

## Synthetic examples

| Headline | Human decision |
|---|---|
| Bitcoin ETF inflows reach a new weekly high | BTC bullish, `etf_flows` |
| Ethereum developers publish an upgrade schedule | ETH neutral, `technology` |
| Solana network outage disrupts transactions | SOL bearish, `technology` |
| BNB rises 8% after increased BNB Chain activity | BNB bullish, `market_movement` |
| Major exchange publishes its annual report | Irrelevant unless a supported asset is clearly affected |

## Completion check

Before the dataset is evaluated:

- Every row must use `relevant` or `irrelevant`; no blank or `review` rows remain.
- Relevant rows must contain at least one asset sentiment and category.
- Irrelevant rows must have all asset fields blank.
- Every sentiment and category must use the exact controlled spelling above.

# CryptoPulse

[![CI](https://github.com/M-Nateghi/CryptoPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/M-Nateghi/CryptoPulse/actions/workflows/ci.yml)
[![Refresh data snapshot](https://github.com/M-Nateghi/CryptoPulse/actions/workflows/refresh-data.yml/badge.svg)](https://github.com/M-Nateghi/CryptoPulse/actions/workflows/refresh-data.yml)

[Open the live CryptoPulse dashboard](https://m-nateghi-cryptopulse.streamlit.app)

CryptoPulse is a production-style cryptocurrency data project that combines
news sentiment with market activity for Bitcoin (BTC), Ethereum (ETH), Solana
(SOL), and BNB.

All six implementation stages are complete. Stage 3 uses independent human
relevance labels and explicitly identified AI-assisted sentiment labels. The
project currently collects recent cryptocurrency news metadata from GDELT,
Google News RSS, and CoinDesk RSS plus hourly OHLCV market data from Binance,
validates the responses, and stores them in SQLite without creating duplicate
records. It can also clean useful RSS descriptions, classify headline-plus-summary inputs into
validated asset-specific sentiment records with OpenAI Structured Outputs,
calculate time-series features, and present them in an interactive Streamlit
dashboard.

## Current Features

- Public Binance ingestion for `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, and `BNBUSDT`
- Public GDELT news searches for BTC, ETH, SOL, and BNB
- Google News RSS ingestion with automatic source fallback
- Official CoinDesk RSS ingestion for genuine article descriptions
- Optional cleaned RSS summaries with headline-only fallback
- Typed configuration through Pydantic settings
- SQLite tables for articles, market candles, ingestion audits, and classifications
- Idempotent inserts that safely skip previously stored data
- Cross-source deduplication using canonical URLs and conservative title overlap,
  including cleanup of previously stored duplicates during news ingestion
- UTC timestamps and structured application logging
- Command-line interface for market, news, or combined ingestion
- Mocked API tests that do not depend on live external services
- Automated tests that run without live API dependencies
- Strict Pydantic models for article relevance and asset-level sentiment
- Provider-independent batch classification with failure tracking and safe retries
- Real OpenAI classification plus a deterministic fake provider for offline testing
- Versioned, bounded headline-plus-summary classification inputs
- Per-asset returns, rolling volatility, volume movement, and daily sentiment metrics
- Streamlit dashboard with asset, date, sentiment, and news-category filters
- Interactive sentiment, category-driver, story, price, and volatility views
- Rolling sentiment-price correlation using fixed six-hour UTC periods
- A versioned, read-only demo snapshot for deployments without private data or keys
- GitHub Actions checks for linting and the complete automated test suite
- Six-hour GitHub Actions refreshes with bounded OpenAI usage and deduplication
- Human-grounded relevance evaluation plus VADER and FinBERT sentiment baselines

## Architecture

```text
CLI command
    |
    v
Configuration and logging
    |
    v
Binance / GDELT / Google News clients
    |
    v
Validated Python models
    |
    v
Ingestion service
    |
    v
SQLite repositories and database
    |
    +--> Reproducible read-only demo snapshot
    |
    `--> Streamlit analytics dashboard
```

The layers are kept separate so that API access, validation, storage, and
orchestration can be tested and changed independently.

## Project Structure

```text
cryptopulse/
|-- analytics/
|   |-- features.py         # Time-series features and aggregations
|   `-- queries.py          # Dashboard database queries
|-- dashboard/
|   `-- data_source.py      # Local database and demo fallback selection
|-- demo/
|   `-- export.py           # Sanitized, reproducible snapshot exporter
|-- evaluation/
|   |-- assisted_labels.py  # Provenance-rich AI-assisted label workflow
|   |-- baselines.py        # VADER and FinBERT adapters
|   |-- labels.py           # Human relevance validation
|   `-- runner.py           # Metrics, confusion matrices, and error analysis
|-- cli.py                  # Command-line entry point
|-- config.py               # Environment-based settings
|-- logging_config.py       # Application logging
|-- db/
|   |-- database.py         # SQLite connection management
|   |-- models.py           # Canonical data records
|   |-- repositories.py     # Database reads and writes
|   `-- schema.py           # Tables, constraints, and indexes
|-- ingestion/
|   |-- binance.py          # Binance market-data client
|   |-- coindesk.py         # Summary-bearing publisher RSS client
|   |-- gdelt.py            # GDELT news client
|   |-- google_news.py      # Google News RSS and summary parser
|   `-- service.py          # Ingestion workflow coordination
`-- sentiment/
    |-- cleaning.py         # Text normalization and asset candidates
    |-- fake_provider.py    # Deterministic offline classifier
    |-- models.py           # Structured sentiment output contract
    |-- openai_provider.py  # Real OpenAI Structured Outputs adapter
    |-- provider.py         # Provider-independent classifier interface
    `-- service.py          # Bounded classification batch workflow

tests/                      # Unit and integration tests
demo/cryptopulse_demo.db    # Versioned, read-only deployment data
streamlit_app.py            # Interactive analytics dashboard
requirements.txt            # Pinned Python dependencies
requirements-evaluation.txt # Optional local VADER and FinBERT dependencies
.github/workflows/ci.yml    # Automated lint and test checks
.github/workflows/refresh-data.yml # Six-hour data refresh and safe publishing
```

Operational databases, virtual environments, secrets, caches, and teaching notes
are excluded from Git through `.gitignore`. The generated demo snapshot is the
single intentional database exception.

## Setup

The project has been tested with Python 3.13 on Windows.

1. Clone the repository and enter its directory:

   ```powershell
   git clone https://github.com/M-Nateghi/CryptoPulse.git
   cd CryptoPulse
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

No API keys are required for Stage 1 or for the offline test suite. Live OpenAI
classification requires an API key. For one PowerShell session, set:

```powershell
$env:CRYPTOPULSE_OPENAI_API_KEY = "your-key-here"
```

Alternatively, store it in a local `.env` file:

```dotenv
CRYPTOPULSE_OPENAI_API_KEY=your-key-here
CRYPTOPULSE_OPENAI_MODEL=gpt-5.6-luna
```

Never commit the `.env` file or paste the key into Python source code.

## Usage

Run market-data ingestion:

```powershell
python -m cryptopulse.cli ingest-market
```

Run news ingestion:

```powershell
python -m cryptopulse.cli ingest-news
```

Run both pipelines:

```powershell
python -m cryptopulse.cli ingest-all
```

Exercise the complete classification workflow without API cost:

```powershell
python -m cryptopulse.cli classify-news --provider fake --limit 10
```

Run real OpenAI classification after configuring the API key:

```powershell
python -m cryptopulse.cli classify-news --provider openai --limit 10
```

Google News articles may include a cleaned RSS description. CryptoPulse stores
that description as an optional summary and supplies it alongside the headline
to the `openai-sentiment-v3` classifier. Empty descriptions, publisher markup,
and descriptions that only repeat the headline are discarded. Summaries are
limited to 1,500 characters, and articles without a useful summary continue
through the original headline-only fallback.

Google News descriptions currently tend to repeat the headline and publisher,
so the scheduled workflow also ingests the official CoinDesk RSS feed as a
summary-bearing source. CoinDesk ingestion is optional: if it is unavailable,
the workflow continues with Google News or GDELT headline data.

Collect additional news for selected assets when preparing evaluation coverage:

```powershell
python -m cryptopulse.cli ingest-news --assets BNB --timespan 7d --max-records 100
```

If GDELT is temporarily rate-limited, use the no-key Google News RSS metadata
source without changing the downstream article schema:

```powershell
python -m cryptopulse.cli ingest-news --source google-news --assets SOL BNB --timespan 7d --max-records 100
```

Create the versioned human-relevance sheet after the database has sufficient
coverage for every supported asset:

```powershell
python -m cryptopulse.cli prepare-evaluation --size 100 --seed 42
```

Follow `evaluation/LABELING_GUIDE.md` and fill only `human_relevance` in
`evaluation/labels_v1.csv`. The command refuses to overwrite an existing sheet.

Validate the completed human labels before running any model evaluation:

```powershell
python -m cryptopulse.cli validate-evaluation
```

Install the optional local baseline dependencies, generate traceable
AI-assisted sentiment for human-relevant rows, and run the hybrid evaluation:

```powershell
python -m pip install -r requirements-evaluation.txt
python -m cryptopulse.cli label-evaluation
python -m cryptopulse.cli run-evaluation
```

The evaluation uses the independent human labels only for relevance. Assisted
sentiment is stored separately with provider, model, prompt version, confidence,
reason, and timestamp. Sentiment results for VADER and FinBERT are therefore
reported as agreement with an AI-assisted reference, not as human-grounded
accuracy. These fixed Stage 3 metrics evaluated headline-only inputs; Stage 6
does not claim an accuracy improvement until summary-aware evaluation is run.

Start the local analytics dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

The dashboard uses `data/cryptopulse.db` when that local database exists. If it
does not, it automatically falls back to the versioned, read-only
`demo/cryptopulse_demo.db` snapshot. Market views work as soon as candles exist;
sentiment views remain explicitly empty until relevant OpenAI classifications
have been stored.

Create or refresh the deployment snapshot from the local database:

```powershell
python -m cryptopulse.cli export-demo --market-hours 720 --article-days 30 --force
```

The exporter keeps the requested recent market window and 30 days of the
latest successful OpenAI classification state. Retaining both relevant and
irrelevant decisions prevents scheduled runs from paying to classify the same
rejected headline again; dashboard queries still display relevant stories only.
It writes to a temporary database first and replaces the tracked snapshot only
after the export succeeds. Fake-provider results, API keys, run logs, and the
human-labelling sheet are never copied into the demo database.

### Analytics formulas

The dashboard preserves the model's `-1` to `+1` sentiment score and calculates
a daily confidence-weighted mean for each asset:

```text
weighted sentiment = sum(sentiment score * confidence) / sum(confidence)
display score       = (weighted sentiment + 1) * 50
```

The display score therefore runs from 0 to 100, with 50 as neutral. Every score
is shown with its story count. Market features are calculated independently per
asset from historical hourly candles: simple and log returns, 24-hour price
change, rolling 24-hour volatility, volume change, and a rolling volume z-score.
All calculations use UTC. Rolling market features use only the current and earlier
candles, while daily sentiment and market values are presented as same-period
associations rather than causal evidence.

### Sentiment-price correlation

The dashboard's **Sentiment-price correlation** tab places sentiment and market
data into fixed six-hour UTC periods. Sentiment change is the difference between
adjacent confidence-weighted sentiment scores. Missing sentiment periods remain
missing rather than being carried forward across a time gap.

Price movement can be compared with sentiment during the same six-hour period,
the following six hours, or the following 24 hours. For each asset, the dashboard
calculates a seven-day rolling Spearman correlation from up to 28 aligned periods
and requires at least 10 valid observations. It displays the current value, its
observation count, its history, and the underlying period records. Correlation is
an exploratory association and does not establish causation or profitability.

The default provider is `openai`. The limit must be between 1 and 200 to keep
each run bounded. Fake classifications are for development and testing only and
must not be presented as analytical results.

The database is created automatically at `data/cryptopulse.db`. Every run is
recorded in the `ingestion_runs` table with its status and record counts.

GDELT can be slower than the market-data endpoint and limits request frequency.
CryptoPulse searches for all four assets in one combined request, uses a separate
45-second timeout, spaces requests, and retries temporary rate-limit, server,
timeout, and network failures with bounded backoff. Permanent failures are
recorded in the audit table and return a non-zero status.

## Configuration

Settings can be overridden with environment variables using the
`CRYPTOPULSE_` prefix. For example:

```powershell
$env:CRYPTOPULSE_LOG_LEVEL = "DEBUG"
$env:CRYPTOPULSE_REQUEST_TIMEOUT_SECONDS = "20"
$env:CRYPTOPULSE_GDELT_REQUEST_TIMEOUT_SECONDS = "60"
$env:CRYPTOPULSE_OPENAI_MODEL = "gpt-5.6-luna"
$env:CRYPTOPULSE_OPENAI_TIMEOUT_SECONDS = "30"
```

The same values may be placed in a local `.env` file, which is excluded from
Git. Never commit API keys or other credentials.

## Tests and Quality Checks

```powershell
python -m pytest
python -m pytest --cov=cryptopulse --cov-report=term-missing
python -m ruff check .
```

The tests use temporary databases and mocked HTTP responses, making the normal
test suite fast, repeatable, and independent of live API availability.

Every push to `main` and every pull request runs Ruff and the full test suite in
GitHub Actions with Python 3.13. No API keys are required by CI.

A separate scheduled workflow refreshes the public data snapshot every six
hours. It fetches recent Binance candles and runs a separate Google News RSS
search for each supported asset so BTC cannot consume the whole news allowance,
automatically falling back to GDELT when a Google News search is unavailable. It
classifies only headlines not already processed by the current model and prompt,
audits the export for secret-shaped text, and commits the refreshed demo
database. The workflow requires the repository secret
`CRYPTOPULSE_OPENAI_API_KEY`. GitHub scheduled workflows use UTC and may begin a
few minutes after the stated time.

### Automated refresh schedule

| Setting | Value |
|---|---|
| Frequency | Every 6 hours |
| Scheduled times | 00:17, 06:17, 12:17, and 18:17 UTC |
| Market source | Binance public API |
| Primary news source | Google News RSS |
| Google News allowance | Up to 50 articles per asset per run |
| News fallback | GDELT DOC API |
| OpenAI limit | At most 100 unseen articles per scheduled run |
| Deduplication window | 7 days of processed headline state |
| Dashboard retention | 30 days of hourly prices and classified news |
| Published artifact | `demo/cryptopulse_demo.db` |

The workflow can also be started manually from the
[Refresh data snapshot workflow](https://github.com/M-Nateghi/CryptoPulse/actions/workflows/refresh-data.yml).
Successful runs commit only the sanitized demo database. Streamlit then
redeploys from `main`; the public app itself never receives the OpenAI key.

## Deployment

CryptoPulse is deployed on Streamlit Community Cloud as a public, read-only
portfolio application at
[m-nateghi-cryptopulse.streamlit.app](https://m-nateghi-cryptopulse.streamlit.app).
The deployed app does not run ingestion or call OpenAI. The API key is used only
by the scheduled GitHub Actions runner and is never placed in Streamlit.

Use these deployment settings:

| Setting | Value |
|---|---|
| Repository | `M-Nateghi/CryptoPulse` |
| Branch | `main` |
| Main file | `streamlit_app.py` |
| Python version | `3.13` |

The root `requirements.txt`, `.streamlit/config.toml`, application entry point,
and demo database are committed for the cloud build. Do not add a Streamlit
secret unless a future deployed feature genuinely requires one.

Snapshot size, data-through time, and story counts advance whenever the scheduled
refresh publishes new data. The initial deployed snapshot contained 672 hourly
market rows and 45 asset-level sentiment results for 27 independently
human-relevant headlines.

### Evaluation results

| Task | Reference | Model | Accuracy/agreement | Macro F1 |
|---|---|---|---:|---:|
| Relevance | Human | OpenAI `gpt-5.6-luna` | 79.0% | 77.6% |
| Sentiment agreement | OpenAI-assisted | VADER 3.3.2 | 64.4% | 57.6% |
| Sentiment agreement | OpenAI-assisted | ProsusAI/finbert | 57.8% | 54.0% |

Only the relevance row is an accuracy result against independent human labels.
The sentiment rows quantify model agreement and must not be interpreted as
human-grounded accuracy.

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| 1 | Foundation, database, API clients, and ingestion | Complete |
| 2 | Article cleaning and structured LLM sentiment | Complete |
| 3 | Human relevance evaluation and assisted sentiment comparison | Complete |
| 4 | Time-series analytics and Streamlit dashboard | Complete |
| 5 | CI, deployment, and portfolio polish | Complete |
| 6 | RSS-enriched sentiment with safe headline-only fallback | Complete |
| 7 | Rolling sentiment-price correlation dashboard | Complete |

The finished application will compare asset-specific news sentiment with price,
volume, returns, and volatility while distinguishing statistical association
from unsupported causal claims.

## Data Sources

- [Binance public market data](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
- [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- [Google News RSS](https://news.google.com/rss)
- [CoinDesk official RSS feed](https://www.coindesk.com/arc/outboundfeeds/rss/)

CryptoPulse is an independent educational portfolio project and is not
affiliated with Binance or GDELT. It does not provide financial advice.

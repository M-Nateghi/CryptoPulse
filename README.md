# CryptoPulse

CryptoPulse is a production-style cryptocurrency data project that will combine
news sentiment with market activity for Bitcoin (BTC), Ethereum (ETH), and
Solana (SOL).

Stages 1 and 2 are complete. The project currently collects
recent cryptocurrency news metadata from GDELT and hourly OHLCV market data from
Binance, validates the responses, and stores them in SQLite without creating
duplicate records. It can also classify stored articles into validated,
asset-specific sentiment records with OpenAI Structured Outputs.

## Current Features

- Public Binance ingestion for `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`
- Public GDELT news searches for BTC, ETH, and SOL
- Typed configuration through Pydantic settings
- SQLite tables for articles, market candles, ingestion audits, and classifications
- Idempotent inserts that safely skip previously stored data
- UTC timestamps and structured application logging
- Command-line interface for market, news, or combined ingestion
- Mocked API tests that do not depend on live external services
- Automated tests that run without live API dependencies
- Strict Pydantic models for article relevance and asset-level sentiment
- Provider-independent batch classification with failure tracking and safe retries
- Real OpenAI classification plus a deterministic fake provider for offline testing

## Architecture

```text
CLI command
    |
    v
Configuration and logging
    |
    v
Binance / GDELT API clients
    |
    v
Validated Python models
    |
    v
Ingestion service
    |
    v
SQLite repositories and database
```

The layers are kept separate so that API access, validation, storage, and
orchestration can be tested and changed independently.

## Project Structure

```text
cryptopulse/
|-- cli.py                  # Command-line entry point
|-- config.py               # Environment-based settings
|-- logging_config.py       # Application logging
|-- db/
|   |-- database.py         # SQLite connection management
|   |-- models.py           # Canonical data records
|   |-- repositories.py     # Database reads and writes
|   `-- schema.py           # Tables, constraints, and indexes
|-- ingestion/
    |-- binance.py          # Binance market-data client
    |-- gdelt.py            # GDELT news client
|   `-- service.py          # Ingestion workflow coordination
`-- sentiment/
    |-- cleaning.py         # Text normalization and asset candidates
    |-- fake_provider.py    # Deterministic offline classifier
    |-- models.py           # Structured sentiment output contract
    |-- openai_provider.py  # Real OpenAI Structured Outputs adapter
    |-- provider.py         # Provider-independent classifier interface
    `-- service.py          # Bounded classification batch workflow

tests/                      # Unit and integration tests
requirements.txt            # Pinned Python dependencies
```

Local databases, virtual environments, secrets, caches, and teaching notes are
excluded from Git through `.gitignore`.

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

The default provider is `openai`. The limit must be between 1 and 100 to keep
each run bounded. Fake classifications are for development and testing only and
must not be presented as analytical results.

The database is created automatically at `data/cryptopulse.db`. Every run is
recorded in the `ingestion_runs` table with its status and record counts.

GDELT can be slower than the market-data endpoint and limits request frequency.
CryptoPulse searches for all three assets in one combined request, uses a separate
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

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| 1 | Foundation, database, API clients, and ingestion | Complete |
| 2 | Article cleaning and structured LLM sentiment | Complete |
| 3 | VADER/FinBERT baselines and model evaluation | Planned |
| 4 | Time-series analytics and Streamlit dashboard | Planned |
| 5 | CI, deployment, and portfolio polish | Planned |

The finished application will compare asset-specific news sentiment with price,
volume, returns, and volatility while distinguishing statistical association
from unsupported causal claims.

## Data Sources

- [Binance public market data](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
- [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)

CryptoPulse is an independent educational portfolio project and is not
affiliated with Binance or GDELT. It does not provide financial advice.

# Demo Data

`cryptopulse_demo.db` is the versioned, read-only data source used when the local
`data/cryptopulse.db` database is unavailable, including public deployments.

Regenerate it from the local database with:

```powershell
python -m cryptopulse.cli export-demo --market-hours 168 --force
```

The export includes recent market candles and only the latest successful,
relevant OpenAI classification for each article. It excludes secrets, ingestion
audits, fake-provider classifications, and the human-labelling sheet.

Before committing a refreshed snapshot, run:

```powershell
python -m pytest
python -m ruff check .
```

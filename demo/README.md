# Demo Data

`cryptopulse_demo.db` is the versioned, read-only data source used when the local
`data/cryptopulse.db` database is unavailable, including public deployments.

Regenerate it from the local database with:

```powershell
python -m cryptopulse.cli export-demo --market-hours 168 --article-days 7 --force
```

The export includes recent market candles and seven days of the latest successful
OpenAI classification state. Relevant and irrelevant decisions are retained for
deduplication, while dashboard queries display relevant stories only. The current
sentiment records combine independent human relevance decisions with traceable
OpenAI-assisted asset labels. The export excludes secrets, ingestion audits,
fake-provider classifications, and the human-labelling sheet itself.

Before committing a refreshed snapshot, run:

```powershell
python -m pytest
python -m ruff check .
```

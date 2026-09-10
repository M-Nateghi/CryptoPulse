from datetime import UTC, datetime

from cryptopulse import cli
from cryptopulse.db.database import open_database
from cryptopulse.db.models import Article, InsertSummary
from cryptopulse.db.repositories import insert_articles
from cryptopulse.db.schema import create_schema


def test_market_command_runs_ingestion_with_temporary_database(monkeypatch, tmp_path):
    database_path = tmp_path / "cli.db"
    calls = []

    def record_market_ingestion(connection, client):
        calls.append((connection, client))
        return InsertSummary(received=1, inserted=1, skipped=0)

    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(database_path))
    monkeypatch.setattr(cli, "ingest_market", record_market_ingestion)

    exit_code = cli.main(["ingest-market"])

    assert exit_code == 0
    assert database_path.exists()
    assert len(calls) == 1


def test_command_returns_failure_exit_code(monkeypatch, tmp_path):
    database_path = tmp_path / "cli.db"

    def fail_market_ingestion(connection, client):
        raise RuntimeError("test failure")

    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(database_path))
    monkeypatch.setattr(cli, "ingest_market", fail_market_ingestion)

    exit_code = cli.main(["ingest-market"])

    assert exit_code == 1


def test_fake_classify_command_runs_end_to_end(monkeypatch, tmp_path):
    database_path = tmp_path / "cli.db"
    with open_database(database_path) as connection:
        create_schema(connection)
        insert_articles(
            connection,
            [
                Article(
                    source="gdelt",
                    external_id=None,
                    title="Bitcoin market update",
                    url="https://example.com/bitcoin",
                    published_at=datetime(2026, 9, 9, 10, tzinfo=UTC),
                    retrieved_at=datetime(2026, 9, 9, 11, tzinfo=UTC),
                    raw_query="Bitcoin",
                )
            ],
        )
    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(database_path))

    first_exit_code = cli.main(
        ["classify-news", "--provider", "fake", "--limit", "1"]
    )
    second_exit_code = cli.main(
        ["classify-news", "--provider", "fake", "--limit", "1"]
    )

    with open_database(database_path) as connection:
        stored = connection.execute(
            "SELECT provider, status FROM article_classifications"
        ).fetchone()
        classification_count = connection.execute(
            "SELECT COUNT(*) FROM article_classifications"
        ).fetchone()[0]
        result_count = connection.execute(
            "SELECT COUNT(*) FROM sentiment_results"
        ).fetchone()[0]

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert stored["provider"] == "fake"
    assert stored["status"] == "succeeded"
    assert classification_count == 1
    assert result_count == 1


def test_openai_classify_command_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(tmp_path / "cli.db"))
    monkeypatch.delenv("CRYPTOPULSE_OPENAI_API_KEY", raising=False)

    exit_code = cli.main(["classify-news", "--provider", "openai", "--limit", "1"])

    assert exit_code == 1


def test_news_command_passes_selected_assets(monkeypatch, tmp_path):
    selected_assets = []

    def record_news_ingestion(connection, client, assets, max_records, timespan):
        selected_assets.extend(assets)
        assert max_records == 120
        assert timespan == "7d"
        return InsertSummary(received=0, inserted=0, skipped=0)

    monkeypatch.setenv(
        "CRYPTOPULSE_DATABASE_PATH",
        str(tmp_path / "cli.db"),
    )
    monkeypatch.setattr(cli, "ingest_news", record_news_ingestion)

    exit_code = cli.main(
        [
            "ingest-news",
            "--assets",
            "SOL",
            "BNB",
            "--max-records",
            "120",
            "--timespan",
            "7d",
        ]
    )

    assert exit_code == 0
    assert selected_assets == ["SOL", "BNB"]


def test_prepare_evaluation_command_creates_labeling_csv(monkeypatch, tmp_path):
    database_path = tmp_path / "cli.db"
    output_path = tmp_path / "labels.csv"
    titles = [
        "Bitcoin update",
        "Ethereum update",
        "Solana update",
        "BNB Chain update",
        "General market one",
        "General market two",
        "General market three",
        "General market four",
        "General market five",
        "General market six",
    ]
    with open_database(database_path) as connection:
        create_schema(connection)
        insert_articles(
            connection,
            [
                Article(
                    source="gdelt",
                    external_id=None,
                    title=title,
                    url=f"https://example.com/evaluation-{index}",
                    published_at=datetime(2026, 9, 10, 10, tzinfo=UTC),
                    retrieved_at=datetime(2026, 9, 10, 11, tzinfo=UTC),
                    raw_query="test",
                )
                for index, title in enumerate(titles)
            ],
        )
    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(database_path))

    exit_code = cli.main(
        [
            "prepare-evaluation",
            "--size",
            "10",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert len(output_path.read_text(encoding="utf-8-sig").splitlines()) == 11

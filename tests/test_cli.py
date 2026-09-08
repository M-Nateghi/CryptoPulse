from cryptopulse import cli
from cryptopulse.db.models import InsertSummary


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

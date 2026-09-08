import sqlite3
from contextlib import closing

import pytest

from cryptopulse.db.database import open_database


def test_open_database_creates_parent_directory_and_commits(tmp_path):
    database_path = tmp_path / "nested" / "test.db"

    with open_database(database_path) as connection:
        connection.execute("CREATE TABLE example (value TEXT NOT NULL)")
        connection.execute("INSERT INTO example (value) VALUES (?)", ("saved",))

    assert database_path.exists()
    with closing(sqlite3.connect(database_path)) as connection:
        result = connection.execute("SELECT value FROM example").fetchone()

    assert result == ("saved",)


def test_open_database_rolls_back_failed_transaction(tmp_path):
    database_path = tmp_path / "test.db"

    with open_database(database_path) as connection:
        connection.execute("CREATE TABLE example (value TEXT NOT NULL)")

    with (
        pytest.raises(RuntimeError, match="force rollback"),
        open_database(database_path) as connection,
    ):
        connection.execute("INSERT INTO example (value) VALUES (?)", ("lost",))
        raise RuntimeError("force rollback")

    with closing(sqlite3.connect(database_path)) as connection:
        row_count = connection.execute("SELECT COUNT(*) FROM example").fetchone()[0]

    assert row_count == 0


def test_open_database_enables_foreign_keys_and_closes_connection(tmp_path):
    database_path = tmp_path / "test.db"

    with open_database(database_path) as connection:
        foreign_keys_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert foreign_keys_enabled == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")

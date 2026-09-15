import argparse
import re
import sqlite3
from pathlib import Path

SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_]{20,}"
)


def _quoted_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def find_secret_locations(database_path: Path) -> list[tuple[str, str, int]]:
    """Return table and column names containing key-shaped text, never the values."""
    connection = sqlite3.connect(database_path)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        ]
        matches = []
        for table in tables:
            table_identifier = _quoted_identifier(table)
            columns = connection.execute(
                f"PRAGMA table_info({table_identifier})"
            ).fetchall()
            for column in columns:
                column_name = column[1]
                column_type = str(column[2]).upper()
                if column_type not in {"", "TEXT"}:
                    continue
                column_identifier = _quoted_identifier(column_name)
                values = connection.execute(
                    f"SELECT rowid, {column_identifier} FROM {table_identifier}"
                )
                for row_id, value in values:
                    if value is not None and SECRET_PATTERN.search(str(value)):
                        matches.append((table, column_name, row_id))
        return matches
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit demo database text fields for OpenAI key-shaped values."
    )
    parser.add_argument(
        "database",
        type=Path,
        nargs="?",
        default=Path("demo/cryptopulse_demo.db"),
    )
    args = parser.parse_args()
    matches = find_secret_locations(args.database)
    if matches:
        print(f"Secret-shaped text found in {len(matches)} field(s): {matches}")
        return 1
    print("No secret-shaped text found in demo database fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

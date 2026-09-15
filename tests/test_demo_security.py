import sqlite3

from scripts.audit_demo_secrets import find_secret_locations


def test_demo_secret_audit_distinguishes_url_slugs_from_keys(tmp_path):
    database_path = tmp_path / "demo.db"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE examples (value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO examples (value) VALUES (?)",
        [
            ("https://example.com/bitcoin-risk-grows-after-rejection",),
            (f"sk-proj-{'A' * 24}",),
        ],
    )
    connection.commit()
    connection.close()

    assert find_secret_locations(database_path) == [("examples", "value", 2)]

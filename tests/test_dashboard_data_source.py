from cryptopulse.dashboard import resolve_dashboard_database


def test_dashboard_prefers_existing_local_database(tmp_path):
    local_path = tmp_path / "local.db"
    demo_path = tmp_path / "demo.db"
    local_path.touch()
    demo_path.touch()

    selected = resolve_dashboard_database(local_path, demo_path)

    assert selected.path == local_path
    assert selected.mode == "local"
    assert not selected.read_only


def test_dashboard_uses_readonly_demo_when_local_database_is_missing(tmp_path):
    local_path = tmp_path / "missing.db"
    demo_path = tmp_path / "demo.db"
    demo_path.touch()

    selected = resolve_dashboard_database(local_path, demo_path)

    assert selected.path == demo_path
    assert selected.mode == "demo"
    assert selected.read_only


def test_dashboard_reports_empty_mode_when_no_database_exists(tmp_path):
    selected = resolve_dashboard_database(
        tmp_path / "missing-local.db",
        tmp_path / "missing-demo.db",
    )

    assert selected.mode == "empty"
    assert not selected.read_only

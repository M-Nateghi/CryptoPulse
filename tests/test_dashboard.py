from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_honest_empty_states(tmp_path, monkeypatch):
    project_root = Path(__file__).parents[1]
    database_path = tmp_path / "empty.db"
    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(database_path))
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(project_root / "streamlit_app.py").run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "CryptoPulse AI"
    messages = [message.value for message in app.info]
    assert "No OpenAI sentiment results match the current filters." in messages
    assert "No market observations match the current filters." in messages
    assert "Evaluation results are pending the completed human-labelled dataset." in (
        messages
    )

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_DEMO_DATABASE_PATH = Path("demo/cryptopulse_demo.db")


@dataclass(frozen=True)
class DashboardDatabase:
    path: Path
    mode: Literal["local", "demo", "empty"]

    @property
    def read_only(self) -> bool:
        return self.mode == "demo"


def resolve_dashboard_database(
    local_path: Path,
    demo_path: Path = DEFAULT_DEMO_DATABASE_PATH,
) -> DashboardDatabase:
    """Prefer local operational data, then the committed portfolio snapshot."""
    if local_path.exists():
        return DashboardDatabase(path=local_path, mode="local")
    if demo_path.exists():
        return DashboardDatabase(path=demo_path, mode="demo")
    return DashboardDatabase(path=local_path, mode="empty")

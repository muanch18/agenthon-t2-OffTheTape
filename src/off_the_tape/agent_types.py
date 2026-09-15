"""Small structural types shared without importing the orchestration module."""

from typing import Protocol


class TaskSpecLike(Protocol):
    assets: tuple[str, ...]
    horizons: tuple[int, ...]
    asset_panels: dict[str, str]

"""Few-Shot Learning Plugin for FiftyOne Labs."""

from typing import Any

from .panel import FewShotLearningPanel


def register(p: Any) -> None:
    """Register the panel plugin with FiftyOne."""
    p.register(FewShotLearningPanel)

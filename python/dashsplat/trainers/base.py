from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


ProgressCb = Callable[[str, float], None]


@dataclass
class TrainerConfig:
    iterations: int = 2500
    max_splats: int = 300_000
    masks_dir: Path | None = None
    force_fallback: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    trainer: str
    ply: str
    spz: str | None
    cameras: str | None
    exit_code: int
    num_gaussians: int | None = None
    log: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Trainer(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def train(
        self,
        project_dir: Path,
        out_dir: Path,
        *,
        config: TrainerConfig,
        progress: ProgressCb | None = None,
    ) -> TrainResult: ...

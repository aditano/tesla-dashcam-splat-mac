from __future__ import annotations

import os
from typing import Any

import numpy as np

from dashsplat.trainers.base import Trainer, TrainerConfig, TrainResult
from dashsplat.trainers.brush import BrushTrainer
from dashsplat.trainers.msplat_trainer import MSplatTrainer
from dashsplat.trainers.seeded import SeededTrainer


def available_trainers() -> list[str]:
    names = []
    if BrushTrainer().is_available():
        names.append("brush")
    if MSplatTrainer().is_available():
        names.append("msplat")
    names.append("seeded")
    return names


def resolve_trainer_name(preferred: str | None = None, *, force_fallback: bool = False) -> str:
    if force_fallback:
        return "seeded"
    env = preferred or os.environ.get("DASHSPLAT_TRAINER") or "brush"
    env = env.lower().strip()
    if env == "opensplat":
        raise ValueError(
            "OpenSplat is AGPL-3.0 and was removed. Use brush (default), msplat, or seeded."
        )
    avail = available_trainers()
    if env in avail:
        return env
    # Prefer brush, then msplat, then seeded
    for cand in ("brush", "msplat", "seeded"):
        if cand in avail:
            return cand
    return "seeded"


def get_trainer(
    name: str | None = None,
    *,
    means: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    cameras: list[dict[str, Any]] | None = None,
    force_fallback: bool = False,
) -> Trainer:
    resolved = resolve_trainer_name(name, force_fallback=force_fallback)
    if resolved == "brush":
        return BrushTrainer()
    if resolved == "msplat":
        return MSplatTrainer()
    seeded = SeededTrainer()
    if means is not None and colors is not None:
        seeded.bind(means, colors, cameras or [])
    return seeded
